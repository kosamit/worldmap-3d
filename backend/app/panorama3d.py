"""単一視点パノラマ生成補完（C: PanoDreamer / 3D Pano Inpainting 系）。

1視点の透視ビュー群(DA3 深度つき)を equirect の「カラー＋半径距離(深度)」に統合し、
穴（空・ビュー間ギャップ）を生成AI(LaMa)＋近傍補間で埋めてから、隙間のない球面
メッシュにする。中心付近を歩ける。

注意: 単一中心の equirect は1方向あたり1深度しか持てないため、前景の真裏（アモーダル）
までは埋まらない。空・取りこぼし等の穴埋めが主目的。前景裏まで埋めるのは複数視点
growth(A) で対応する。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .reconstruct_da3 import (
    _as_homogeneous44,
    _decimate_mesh,
    _drop_small_components,
    _level_ground,
    _metric_scale,
    _noop,
)


def _build_equirect_rgbd(prediction, view_index, vp_idx, eq_w=1536):
    """指定視点の透視ビュー群を equirect(カラー/半径距離/有効マスク) に z-buffer 統合。

    返り値: color(H,W,3 uint8), radius(H,W float, 中心からの距離), valid(H,W bool)。
    equirect 規約: x=lon(0=+Z前方, 右回り), y=lat(上+)。
    """
    depth = prediction["depth"].astype(np.float64)
    K = prediction["intrinsics"].astype(np.float64)
    ext = prediction["extrinsics"]
    imgs = prediction["processed_images"]
    n, h, w = depth.shape
    eq_h = eq_w // 2

    color = np.zeros((eq_h, eq_w, 3), np.uint8)
    radius = np.zeros((eq_h, eq_w), np.float64)
    rbest = np.full((eq_h, eq_w), np.inf)  # z-buffer（手前優先）

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([us.ravel(), vs.ravel(), np.ones(h * w)], 0)  # (3, hw)

    views = [i for i in range(n) if view_index[i] == vp_idx]
    # 視点中心（このグループ平均）を原点に
    c2ws = [np.linalg.inv(_as_homogeneous44(ext[i])) for i in views]
    center = np.mean([c[:3, 3] for c in c2ws], axis=0)

    for k, i in enumerate(views):
        c2w = c2ws[k]
        R = c2w[:3, :3]
        z = depth[i].ravel()
        valid = np.isfinite(z) & (z > 0)
        rays = np.linalg.inv(K[i]) @ pix  # (3,hw) カメラ座標方向
        cam = rays * z[None, :]  # (3,hw)
        world = (R @ cam).T + (c2w[:3, 3] - center)  # (hw,3) 中心相対
        r = np.linalg.norm(world, axis=1)
        d = world / (r[:, None] + 1e-9)
        lon = np.arctan2(d[:, 0], d[:, 2])  # -pi..pi
        lat = np.arcsin(np.clip(d[:, 1], -1, 1))  # -pi/2..pi/2
        px = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.int64) % eq_w
        py = ((0.5 - lat / np.pi) * eq_h).astype(np.int64).clip(0, eq_h - 1)
        col = imgs[i].reshape(-1, 3)
        sel = valid & (r < 1e6)
        # z-buffer: 同一equirect画素は手前(r小)を採用
        idx = py * eq_w + px
        order = np.argsort(-r[sel])  # 遠い順に書く→最後に手前が残る
        si = np.flatnonzero(sel)[order]
        flat_r = rbest.ravel(); flat_rad = radius.ravel(); flat_col = color.reshape(-1, 3)
        write = r[si] < flat_r[idx[si]]
        ii = idx[si][write]
        flat_r[ii] = r[si][write]
        flat_rad[ii] = r[si][write]
        flat_col[ii] = col[si][write]
    valid_eq = np.isfinite(rbest)
    return color, radius, valid_eq, center


def _inpaint_equirect(color, radius, valid, use_lama=True):
    """equirect の無効画素（穴）を補完。カラーは LaMa、半径は近傍からの拡散で埋める。"""
    import cv2

    hole = ~valid
    if not hole.any():
        return color, radius
    mask = (hole.astype(np.uint8)) * 255
    # カラー: LaMa（無ければ OpenCV inpaint にフォールバック）
    filled_color = color.copy()
    if use_lama:
        try:
            from . import segment
            out = segment.inpaint_masked(color[None], hole[None])
            filled_color = out[0]
        except Exception:
            filled_color = cv2.inpaint(color, mask, 5, cv2.INPAINT_TELEA)
    else:
        filled_color = cv2.inpaint(color, mask, 5, cv2.INPAINT_TELEA)
    # 半径: 既知画素から最近傍の値で埋める（穴は背景＝遠めになりやすい）
    r = radius.copy().astype(np.float32)
    known = valid.astype(np.uint8)
    # distance transform で各穴に最近傍既知画素のインデックスを得る
    _, labels = cv2.distanceTransformWithLabels(
        (1 - known), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    known_idx = np.flatnonzero(known.ravel())
    # labels は「最近傍のゼロでない(=既知)画素」の連番ラベル → 値マップを作る
    lab_known = labels.ravel()[known_idx]
    order = np.argsort(lab_known)
    lut = np.zeros(lab_known.max() + 1, np.float32)
    lut[lab_known[order]] = r.ravel()[known_idx][order]
    rf = r.ravel()
    h = hole.ravel()
    rf[h] = lut[labels.ravel()[h]]
    return filled_color, rf.reshape(radius.shape)


def build_panorama_mesh(prediction, view_index, viewpoints, vp_idx=0, eq_w=1536,
                        inpaint=True, discontinuity_ratio=0.15, edge_factor=0.7,
                        max_faces=1_200_000, progress=None):
    """単一視点 equirect RGBD → 穴埋め → 隙間のない球面メッシュ(trimesh.Scene)。"""
    progress = progress or _noop
    progress("mesh", 0, 1, "パノラマ統合（equirect RGBD）...")
    color, radius, valid, _ = _build_equirect_rgbd(prediction, view_index, vp_idx, eq_w)

    filled_frac = float((~valid).mean())
    if inpaint:
        progress("mesh", 0, 1, "穴を生成補完（LaMa）...")
        color, radius = _inpaint_equirect(color, radius, valid)
        valid_mesh = np.ones_like(valid)
    else:
        valid_mesh = valid

    progress("mesh", 0, 1, "球面メッシュ生成 ...")
    eq_h, eq_w2 = radius.shape
    lon = (np.arange(eq_w2) / eq_w2 - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(eq_h) / eq_h) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    verts = (dirs * radius[..., None]).reshape(-1, 3)

    # スケール（メートル化 or 半径正規化）
    n = prediction["depth"].shape[0]
    ext = prediction["extrinsics"]
    centers = np.array([np.linalg.inv(_as_homogeneous44(ext[i]))[:3, 3] for i in range(n)])
    scale = _metric_scale(centers, list(view_index), viewpoints)
    if not scale or not np.isfinite(scale) or scale <= 0:
        rad95 = float(np.percentile(radius[valid_mesh], 95)) or 1.0
        scale = 15.0 / rad95
    verts = verts * scale

    # グリッド三角形化（経度方向は wrap）。深度不連続は分断。
    idx = np.arange(eq_h * eq_w2).reshape(eq_h, eq_w2)
    j = np.arange(eq_w2); jr = (j + 1) % eq_w2
    tl = idx[:-1, j].ravel(); tr = idx[:-1, jr].ravel()
    bl = idx[1:, j].ravel(); br = idx[1:, jr].ravel()
    rr = (radius * scale).ravel()
    rq = np.stack([rr[tl], rr[tr], rr[bl], rr[br]], 1)
    rmean = rq.mean(1) + 1e-9
    cont = (rq.max(1) - rq.min(1)) / rmean < discontinuity_ratio
    vflat = valid_mesh.ravel()
    keep = cont & vflat[tl] & vflat[tr] & vflat[bl] & vflat[br]
    tri1 = np.stack([tl, bl, tr], 1)[keep]
    tri2 = np.stack([tr, bl, br], 1)[keep]
    faces = np.concatenate([tri1, tri2], 0)

    verts, _ = _level_ground(verts.astype(np.float32))
    ref = np.zeros(len(verts), bool); ref[np.unique(faces)] = True
    verts = (verts - np.median(verts[ref], axis=0)).astype(np.float32)

    rgba = np.concatenate([color.reshape(-1, 3), np.full((len(verts), 1), 255, np.uint8)], 1)
    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
    m.remove_unreferenced_vertices()
    m = _drop_small_components(m)
    if max_faces and len(m.faces) > max_faces:
        m = _decimate_mesh(m, max_faces)
    scene = trimesh.Scene(); scene.add_geometry(m)
    info = {"representation": "mesh", "method": "panorama", "vertex_count": int(len(m.vertices)),
            "face_count": int(len(m.faces)), "viewpoints": 1, "eq_w": eq_w,
            "hole_filled_frac": round(filled_frac, 3), "inpaint": bool(inpaint)}
    return scene, info
