"""単一視点パノラマ生成補完（C: PanoDreamer / 3D Pano Inpainting 系）。

1視点の透視ビュー群(DA3 深度つき)を equirect の「カラー＋半径距離(深度)」に統合し、
穴（空・ビュー間ギャップ）を生成AI(LaMa)＋近傍補間で埋めてから、隙間のない球面
メッシュにする。中心付近を歩ける。

注意: 単一中心の equirect は1方向あたり1深度しか持てないため、前景の真裏（アモーダル）
までは埋まらない。空・取りこぼし等の穴埋めが主目的。前景裏まで埋めるのは複数視点
growth(A) で対応する。
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

from .reconstruct_da3 import (
    _as_homogeneous44,
    _decimate_mesh,
    _drop_small_components,
    _level_ground,
    _metric_scale,
    _noop,
    _umeyama_2d,
    _viewpoint_enu,
)


def _build_equirect_rgbd(prediction, view_index, vp_idx, eq_w=1536, center=None):
    """指定視点の透視ビュー群を equirect(カラー/半径距離/有効マスク) に z-buffer 統合。

    返り値: color(H,W,3 uint8), radius(H,W float, 中心からの距離), valid(H,W bool), center。
    equirect 規約: x=lon(0=+Z前方, 右回り), y=lat(上+)。
    center 指定時は equirect の原点をその座標に固定（複数パノを共通座標に並べる用）。
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
        world = (R @ cam).T + (c2w[:3, 3] - center)  # (hw,3) 中心相対（DA3はy下向き）
        world[:, 1] *= -1.0  # y下向き → y上向き(glTF)。これを忘れると上下逆さまになる
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
    """equirect の無効画素（穴）を補完。カラーは LaMa、半径は近傍からの拡散で埋める。

    返り値: (filled_color, radius, lama_used)。lama_used は実際に LaMa が走ったか。
    LaMa が失敗した場合は無言で握りつぶさずログを出し、cv2 のぼかし補完に降格する。
    """
    import cv2

    hole = ~valid
    if not hole.any():
        return color, radius, False
    mask = (hole.astype(np.uint8)) * 255
    # カラー: LaMa（失敗時は OpenCV inpaint にフォールバックし、必ずログを残す）
    filled_color = color.copy()
    lama_used = False
    if use_lama:
        try:
            from . import segment
            out = segment.inpaint_masked(color[None], hole[None])
            filled_color = out[0]
            lama_used = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("LaMa 穴埋め失敗 → cv2.inpaint に降格: %s", exc, exc_info=True)
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
    return filled_color, rf.reshape(radius.shape), lama_used


def _eq_layer(color, radius, valid, scale, discontinuity):
    """equirect の1レイヤー(カラー/半径/有効) → (verts, faces, rgba)。経度は wrap。"""
    eq_h, eq_w2 = radius.shape
    lon = (np.arange(eq_w2) / eq_w2 - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(eq_h) / eq_h) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    verts = (dirs * radius[..., None]).reshape(-1, 3) * scale

    idx = np.arange(eq_h * eq_w2).reshape(eq_h, eq_w2)
    j = np.arange(eq_w2); jr = (j + 1) % eq_w2
    tl = idx[:-1, j].ravel(); tr = idx[:-1, jr].ravel()
    bl = idx[1:, j].ravel(); br = idx[1:, jr].ravel()
    rr = (radius * scale).ravel()
    rq = np.stack([rr[tl], rr[tr], rr[bl], rr[br]], 1)
    rmean = rq.mean(1) + 1e-9
    cont = (rq.max(1) - rq.min(1)) / rmean < discontinuity
    vflat = valid.ravel()
    keep = cont & vflat[tl] & vflat[tr] & vflat[bl] & vflat[br]
    faces = np.concatenate([np.stack([tl, bl, tr], 1)[keep],
                            np.stack([tr, bl, br], 1)[keep]], 0)
    rgba = np.concatenate([color.reshape(-1, 3), np.full((eq_h * eq_w2, 1), 255, np.uint8)], 1)
    return verts.astype(np.float32), faces, rgba


def _equirect_dirs(eq_h, eq_w2):
    """equirect 各画素の単位方向ベクトル (eq_h,eq_w2,3)。_eq_layer と同じ規約(y上)。"""
    lon = (np.arange(eq_w2) / eq_w2 - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(eq_h) / eq_h) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack(
        [np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1
    )


def _fit_ground_plane_ransac(points, thresh, iters=400, min_inliers=80):
    """点群 (M,3) に RANSAC で平面を当て (法線n(上向き), 平面上の点p0, inlier_mask) を返す。

    測量の基準面(datum)を作るイメージ: 外れ値(物体・壁)に強く、道路面だけを拾う。
    当てられなければ None。
    """
    m = len(points)
    if m < min_inliers:
        return None
    best_inl = None
    best_cnt = 0
    rng = np.random.default_rng(0)  # 決定的（再現性のため固定シード）
    for _ in range(iters):
        ids = rng.choice(m, 3, replace=False)
        a, b, c = points[ids]
        n = np.cross(b - a, c - a)
        nl = np.linalg.norm(n)
        if nl < 1e-9:
            continue
        n = n / nl
        dist = np.abs((points - a) @ n)
        inl = dist < thresh
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
    if best_inl is None or best_cnt < min_inliers:
        return None
    # inlier で最小二乗リフィット（重心＋共分散最小固有ベクトル＝法線）。
    # 注意: np.linalg.svd(P-p0) は U が (M,M) になり巨大配列を確保するため、
    # 3x3 の共分散行列の固有分解（最小固有値の固有ベクトル＝法線）で求める。
    P = points[best_inl]
    p0 = P.mean(axis=0)
    Q = P - p0
    cov = Q.T @ Q  # (3,3)
    _, eigvecs = np.linalg.eigh(cov)  # 昇順。最小固有値=eigvecs[:,0]＝法線
    n = eigvecs[:, 0]
    if n[1] < 0:  # 上向き(+Y)に揃える
        n = -n
    return n / (np.linalg.norm(n) + 1e-12), p0, best_inl


def _ground_plane_cap(radius, valid, camera_height_m=2.5, min_down_deg=8.0):
    """RANSACで床平面を当て、下向きレイの半径を平面でキャップする。

    単眼深度は遠方を圧縮し、水平線に向かって床がせり上がる(お椀状)。床を解析平面で
    キャップすると、床が平面・水平になりこの歪みが消える。さらに既知のカメラ高さ
    (Street View ≈ 2.5m)を標尺に、平面までの距離からメートルスケールを推定する。

    radius: equirect(中心からの距離, y上). 返り値: (radius2, info)。
    """
    eq_h, eq_w2 = radius.shape
    dirs = _equirect_dirs(eq_h, eq_w2)  # (H,W,3)
    laty = dirs[..., 1]  # = sin(lat); <0 が下向き
    down_cand = valid & (laty < -np.sin(np.deg2rad(min_down_deg))) & (radius > 0) & np.isfinite(radius)
    if down_cand.sum() < 200:
        return radius, {"ground_cap_applied": False}
    pts = (dirs * radius[..., None])[down_cand]  # (M,3)
    thresh = 0.03 * float(np.median(radius[down_cand])) or 0.05
    fit = _fit_ground_plane_ransac(pts, thresh)
    if fit is None:
        return radius, {"ground_cap_applied": False}
    n, p0, inl = fit
    offset = -float(n @ p0)              # 平面: n·x + offset = 0
    height_da3 = abs(offset)             # カメラ(原点)から平面までの距離
    if height_da3 < 1e-6:
        return radius, {"ground_cap_applied": False}
    # 各画素の平面交点距離 r = -offset / (n·dir)（n·dir<0 のみ前方で交わる）
    nd = dirs @ n                        # (H,W)
    forward = nd < -1e-3
    r_plane = np.full_like(radius, np.inf)
    r_plane[forward] = -offset / nd[forward]
    # 床帯（水平線のやや下〜真下）は平面に「スナップ」して、せり上がり/曲がり/傾きを
    # 平らな水平床へ強制する（min では手前に圧縮された床=せり上がりが直らないため）。
    snap = forward & valid & np.isfinite(r_plane) & (r_plane > 0) & (nd < -np.sin(np.deg2rad(3.0)))
    radius2 = radius.copy()
    radius2[snap] = r_plane[snap]
    # ただし足元(急角度)で平面より十分手前にある点は実在の近接物体として残す。
    keep_obj = snap & (nd < -0.30) & (radius < r_plane - 0.5 * height_da3)
    radius2[keep_obj] = radius[keep_obj]
    capped = snap
    metric_scale = camera_height_m / height_da3
    info = {
        "ground_cap_applied": True,
        "ground_camera_height_da3": round(height_da3, 4),
        "ground_metric_scale": float(metric_scale),
        "ground_inlier_frac": round(float(inl.mean()), 3),
        "ground_capped_frac": round(float(capped.mean()), 3),
        "ground_tilt_deg": round(float(np.degrees(np.arccos(np.clip(n[1], -1, 1)))), 2),
    }
    return radius2, info


def _global_scale(prediction, view_index, viewpoints, radius_hint=None):
    """視点間距離からメートル/単位スケールを推定。単一視点等で不能なら半径フォールバック。"""
    n = prediction["depth"].shape[0]
    ext = prediction["extrinsics"]
    centers = np.array([np.linalg.inv(_as_homogeneous44(ext[i]))[:3, 3] for i in range(n)])
    scale = _metric_scale(centers, list(view_index), viewpoints)
    if not scale or not np.isfinite(scale) or scale <= 0:
        scale = (15.0 / radius_hint) if radius_hint else None
    return scale


def _pano_geometry(prediction, view_index, viewpoints, vp_idx, scale=None, eq_w=1536,
                   inpaint=True, layered=True, front_discontinuity=0.45,
                   back_discontinuity=0.5, ground_cap=True, camera_height_m=2.5,
                   progress=None):
    """1視点パノラマの (verts(視点中心相対メートル), faces, rgba, center(DA3単位), scale, info)。

    leveling/centering はしない（複数パノを並べる呼び出し側でまとめて行う）。
    verts は視点中心を原点とするメートル座標 → 配置時は (center - ref) * scale で平行移動。
    """
    import cv2

    progress = progress or _noop
    progress("mesh", 0, 1, "パノラマ統合（equirect RGBD）...")
    color, radius, valid, center = _build_equirect_rgbd(prediction, view_index, vp_idx, eq_w)

    filled_frac = float((~valid).mean())
    hole_lama = False
    if inpaint:
        progress("mesh", 0, 1, "穴を生成補完（LaMa）...")
        color, radius, hole_lama = _inpaint_equirect(color, radius, valid)
        valid = np.ones_like(valid)

    # 床平面キャップ（RANSAC）: 床のせり上がり/傾きを除去し、既知カメラ高さで実寸推定。
    ground_info = {"ground_cap_applied": False}
    if ground_cap:
        progress("mesh", 0, 1, "床平面をRANSACで推定（測量）...")
        radius, ground_info = _ground_plane_cap(radius, valid, camera_height_m=camera_height_m)

    if scale is None:
        # 床平面から得た実寸スケール（既知カメラ高さ基準）を最優先。なければ従来推定。
        gscale = ground_info.get("ground_metric_scale")
        if gscale and np.isfinite(gscale) and gscale > 0:
            scale = float(gscale)
        else:
            rad95 = float(np.percentile(radius[valid], 95)) or 1.0
            scale = _global_scale(prediction, view_index, viewpoints, radius_hint=rad95)

    # 前景レイヤー（不連続でしっかり切る＝柱が背景へ伸びない）
    progress("mesh", 0, 1, "球面メッシュ生成（前景）...")
    fv, ff, frgba = _eq_layer(color, radius, valid, scale, front_discontinuity)
    all_v = [fv]; all_f = [ff]; all_c = [frgba]; voff = len(fv)

    # 背景レイヤー（遮蔽補完）: 前景(近)を周囲の遠で置換し、その色を LaMa で描き直す。
    fg_frac = 0.0
    back_lama = False
    if layered:
        progress("mesh", 0, 1, "遮蔽の背後を生成補完（レイヤード）...")
        rad32 = radius.astype(np.float32)
        win = max(9, (eq_w // 64) | 1)  # 奇数の窓
        rmax = cv2.dilate(rad32, np.ones((win, win), np.uint8))  # 周囲の遠い半径
        margin = np.maximum(0.5, 0.25 * rmax)
        fg = (rmax - rad32) > margin  # 周囲より十分近い＝前景の縁/物体
        fg = cv2.dilate(fg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        fg_frac = float(fg.mean())
        if fg.any():
            try:
                from . import segment
                bg_color = segment.inpaint_masked(color[None], fg[None])[0]
                back_lama = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("LaMa 遮蔽補完失敗 → cv2.inpaint に降格: %s", exc, exc_info=True)
                bg_color = cv2.inpaint(color, (fg.astype(np.uint8) * 255), 5, cv2.INPAINT_TELEA)
            bv, bf, brgba = _eq_layer(bg_color, rmax, fg, scale, back_discontinuity)
            all_v.append(bv); all_f.append(bf + voff); all_c.append(brgba)

    verts = np.concatenate(all_v, 0).astype(np.float32)
    faces = np.concatenate(all_f, 0)
    rgba = np.concatenate(all_c, 0)
    # lama_used: この視点で LaMa が実際に1回でも走ったか（穴埋め or 遮蔽補完）。
    # False かつ inpaint/layered が True なら cv2 ぼかしに降格している＝要調査。
    lama_used = bool(hole_lama or back_lama)
    info = {"hole_filled_frac": round(filled_frac, 3), "occlusion_fill_frac": round(fg_frac, 3),
            "lama_used": lama_used, **ground_info}
    return verts, faces, rgba, np.asarray(center, np.float64), scale, info


def build_panorama_mesh(prediction, view_index, viewpoints, vp_idx=0, eq_w=1536,
                        inpaint=True, layered=True, front_discontinuity=0.45,
                        back_discontinuity=0.5, ground_cap=True, camera_height_m=2.5,
                        max_faces=1_200_000, progress=None):
    """単一視点 equirect RGBD → 穴埋め＋レイヤード遮蔽補完 → 球面メッシュ(trimesh.Scene)。"""
    progress = progress or _noop
    verts, faces, rgba, _c, _s, info = _pano_geometry(
        prediction, view_index, viewpoints, vp_idx, scale=None, eq_w=eq_w,
        inpaint=inpaint, layered=layered, front_discontinuity=front_discontinuity,
        back_discontinuity=back_discontinuity, ground_cap=ground_cap,
        camera_height_m=camera_height_m, progress=progress)

    verts, _ = _level_ground(verts)
    ref = np.zeros(len(verts), bool)
    if len(faces):
        ref[np.unique(faces)] = True
    verts = (verts - np.median(verts[ref], axis=0)).astype(np.float32)

    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
    m.remove_unreferenced_vertices()
    m = _drop_small_components(m)
    if max_faces and len(m.faces) > max_faces:
        m = _decimate_mesh(m, max_faces)
    scene = trimesh.Scene(); scene.add_geometry(m)
    info.update({"representation": "mesh", "method": "panorama",
                 "vertex_count": int(len(m.vertices)), "face_count": int(len(m.faces)),
                 "viewpoints": 1, "eq_w": eq_w, "inpaint": bool(inpaint),
                 "layered": bool(layered)})
    return scene, info


def build_multipano_scene(prediction, view_index, viewpoints, eq_w=1536, inpaint=True,
                          layered=True, front_discontinuity=0.45, back_discontinuity=0.5,
                          ground_cap=True, camera_height_m=2.5,
                          max_faces_per_pano=500_000, progress=None):
    """複数視点パノラマを共通座標に配置 → 連続的に歩ける「つなぎ目のない」シーン。

    各視点の深度パノラマメッシュを実位置(視点間の実距離スケール)に並べる。隣の
    パノラマが遮蔽の裏を実際に撮っているので、移動すると本物の視差で裏が見える。
    返り値の info["panos"] = [{"name","center":[x,y,z]}] はビューワーの距離フェード用。
    """
    progress = progress or _noop
    vps = sorted({int(v) for v in view_index})
    if len(vps) < 2:
        return build_panorama_mesh(prediction, view_index, viewpoints, vp_idx=vps[0],
                                   eq_w=eq_w, inpaint=inpaint, layered=layered,
                                   front_discontinuity=front_discontinuity,
                                   back_discontinuity=back_discontinuity,
                                   ground_cap=ground_cap, camera_height_m=camera_height_m,
                                   progress=progress)

    # スケールは従来どおり視点間ベースラインから推定し全パノで共有（安定・既知挙動）。
    # 床RANSAC高さ基準のスケールは外れると全体が激変するため、ここでは使わない。
    scale = _global_scale(prediction, view_index, viewpoints)

    # GPSアンカー: 視点の水平配置は実GPS(ENU)で固定し、DA3ポーズドリフトから切り離す。
    # Street Viewは全カメラがほぼ同じ高さ → 視点間の上下差はDA3誤差なので等高にする。
    p_arr = _viewpoint_enu([viewpoints[vp] for vp in vps])  # (V,3) ENU メートル
    p_vp = {vp: p_arr[k] for k, vp in enumerate(vps)}

    geoms = []          # [vp, verts(視点中心相対メートル), faces, rgba]
    centers = {}        # vp -> DA3 center（ヨー整列の推定に使う）
    infos = []
    for k, vp in enumerate(vps):
        progress("mesh", k, len(vps), f"パノラマ {k + 1}/{len(vps)} 構築 ...")
        v, f, c, ctr, sc, info = _pano_geometry(
            prediction, view_index, viewpoints, vp, scale=scale, eq_w=eq_w,
            inpaint=inpaint, layered=layered, front_discontinuity=front_discontinuity,
            back_discontinuity=back_discontinuity, ground_cap=ground_cap,
            camera_height_m=camera_height_m, progress=progress)
        scale = sc  # 最初のパノで確定したスケールを以降で共有
        centers[vp] = ctr
        geoms.append([vp, v, f, c])
        infos.append(info)

    # DA3水平面(x,z)→ENU(east,north)のヨー回転を1つ推定（全パノ共通）。
    qa = np.array([centers[vp] for vp in vps])
    _, R2, _ = _umeyama_2d(qa[:, [0, 2]], p_arr[:, [0, 2]])

    def _place(verts, vp):
        # verts: 視点中心相対メートル(y-up)。水平をENUへ回し、ENU実位置へ。高さは等高(0)。
        en = (R2 @ verts[:, [0, 2]].T).T  # (M,2) east,north
        x = en[:, 0] + p_vp[vp][0]
        z = -(en[:, 1] + p_vp[vp][1])     # glTF z = -north
        return np.stack([x, verts[:, 1], z], -1).astype(np.float32)

    for g in geoms:
        g[1] = _place(g[1], g[0])

    # leveling/centering は全体で1回（パノ間の整合を保つため）。視点原点も一緒に変換。
    counts = [len(g[1]) for g in geoms]
    total = sum(counts)
    vp_origins = np.array(
        [[p_vp[vp][0], 0.0, -p_vp[vp][1]] for vp in vps], np.float32)
    allv = np.concatenate([g[1] for g in geoms] + [vp_origins], 0)
    allv, _ = _level_ground(allv)
    med = np.median(allv[:total], axis=0)
    allv = (allv - med).astype(np.float32)
    origins_final = allv[total:]

    scene = trimesh.Scene()
    pano_meta = []
    idx = 0
    for (vp, _v, f, c), cnt, k in zip(geoms, counts, range(len(geoms))):
        vv = allv[idx:idx + cnt]; idx += cnt
        m = trimesh.Trimesh(vertices=vv, faces=f, vertex_colors=c, process=False)
        m.remove_unreferenced_vertices()
        m = _drop_small_components(m)
        if max_faces_per_pano and len(m.faces) > max_faces_per_pano:
            m = _decimate_mesh(m, max_faces_per_pano)
        name = f"pano_{vp}"
        scene.add_geometry(m, geom_name=name)
        pano_meta.append({"name": name,
                          "center": [float(x) for x in origins_final[k]]})

    info = {"representation": "mesh", "method": "multipano",
            "vertex_count": int(sum(len(g.vertices) for g in scene.geometry.values())),
            "face_count": int(sum(len(g.faces) for g in scene.geometry.values())),
            "viewpoints": len(vps), "eq_w": eq_w, "inpaint": bool(inpaint),
            "layered": bool(layered), "panos": pano_meta,
            "hole_filled_frac": round(float(np.mean([i["hole_filled_frac"] for i in infos])), 3),
            "occlusion_fill_frac": round(float(np.mean([i["occlusion_fill_frac"] for i in infos])), 3),
            "lama_used": bool(any(i.get("lama_used") for i in infos)),
            "ground_cap_applied": bool(any(i.get("ground_cap_applied") for i in infos)),
            "ground_metric_scale": float(scale),
            "metric_scale": float(scale)}
    return scene, info
