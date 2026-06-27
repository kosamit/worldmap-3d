"""意味コライダー層（戦略1・深度モデル不要）。

セマンティックセグメンテーション＋既知カメラ高さ（測量）から、床/壁/物体の
3Dコライダー（メートル単位）を生成する。深度推定(DA3)は使わず、各視点の「向き」と
床平面仮定だけで幾何を決める。docs/semantic_scene_colliders.md の戦略1。

出力:
  - colliders: [{class, type:"box"|"cylinder"|"plane", center, size, yaw}] (メートル, y上)
  - debug Scene: クラス色のメッシュ（目視確認用）

座標系: equirect と同じ y 上・カメラ原点。床は y=-H の水平面。
"""
from __future__ import annotations

import numpy as np
import trimesh

from . import semseg
from .reconstruct_da3 import _as_homogeneous44, _noop

# カテゴリ → 表示色 (RGB)。debug メッシュ用。
CATEGORY_COLOR = {
    "floor": (90, 90, 95),
    "wall": (170, 150, 130),
    "sky": (135, 180, 235),
    "pole": (210, 200, 60),
    "vegetation": (70, 150, 70),
    "person": (220, 60, 60),
    "vehicle": (60, 120, 220),
    "other": (150, 150, 150),
}
WALL_CATS = {"wall"}
OBJECT_BOX_CATS = {"vehicle", "person"}
OBJECT_CYL_CATS = {"pole"}


def _equirect_dirs(eq_h, eq_w):
    lon = (np.arange(eq_w) / eq_w - 0.5) * 2 * np.pi
    lat = (0.5 - np.arange(eq_h) / eq_h) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1)
    return dirs, LAT


def _pil(arr):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(arr[:, :, :3]))


def _pose_from_angles(heading_deg, pitch_deg):
    """既知の取得角(heading 北から時計回り, pitch 上+)から DA3 規約の外部パラメータ(w2c, 3x4)。

    DA3 はカメラ y 下向き・前方 +Z。build_semantic_equirect は d=R·K⁻¹p の後に y を反転して
    y上 equirect にする。ここではその規約に合う回転 R（c2w）を解析的に作る（深度・DA3不要）。
    方位 h は +Z(=正面)から +X 方向へ。translation は equirect の向きには無関係なので 0。
    """
    h = np.deg2rad(float(heading_deg))
    p = np.deg2rad(float(pitch_deg))
    # world_da3(y下) での前方ベクトル（y上では elevation=p, azimuth=h）。
    f = np.array([np.sin(h) * np.cos(p), -np.sin(p), np.cos(h) * np.cos(p)])
    up = np.array([0.0, -1.0, 0.0])              # da3 の上（y下なので -Y）
    right = np.cross(up, f)
    right /= np.linalg.norm(right) + 1e-9
    down = np.cross(f, right)                     # カメラ下（右手系: x×y=z → right×down=f）
    R = np.stack([right, down, f], axis=1)        # 列 = カメラ軸の world 表現（c2w 回転）
    c2w = np.eye(4)
    c2w[:3, :3] = R
    return np.linalg.inv(c2w)[:3]


def build_analytic_prediction(images, view_angles, size=384):
    """既知取得角から DA3-free な prediction を作る（depth/conf/sky は無し）。

    images: PIL 画像のリスト。view_angles: [(heading, pitch, fov)]（images と同順・同数）。
    返り値: build_semantic_equirect が使う {processed_images, intrinsics, extrinsics}。
    """
    imgs = np.stack([
        np.asarray(im.convert("RGB").resize((size, size))) for im in images
    ])
    K, ext = [], []
    for (h, p, fov) in view_angles:
        fx = (size / 2) / np.tan(np.deg2rad(float(fov)) / 2)
        K.append([[fx, 0, size / 2], [0, fx, size / 2], [0, 0, 1]])
        ext.append(_pose_from_angles(h, p))
    return {
        "processed_images": imgs,
        "intrinsics": np.array(K, np.float64),
        "extrinsics": np.array(ext, np.float64),
    }


def build_semantic_equirect(prediction, view_index, vp_idx, eq_w=1024, progress=None):
    """各視点のカテゴリmaps を equirect に多数決投票（深度不要・向きのみ）。

    返り値: cat_eq (eq_h,eq_w) int8（CATEGORIES添字）。
    """
    progress = progress or _noop
    imgs = prediction["processed_images"]
    K = prediction["intrinsics"].astype(np.float64)
    ext = prediction["extrinsics"]
    n, h, w = imgs.shape[:3]
    eq_h = eq_w // 2
    ncat = len(semseg.CATEGORIES)

    views = [i for i in range(n) if view_index[i] == vp_idx]
    progress("mesh", 0, 1, f"セグメンテーション（{len(views)}枚）...")
    cats = semseg.segment_categories([_pil(imgs[i]) for i in views])  # (V,h,w) int8

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    pix = np.stack([us.ravel(), vs.ravel(), np.ones(h * w)], 0)  # (3,hw)
    votes = np.zeros((eq_h, eq_w, ncat), np.int32)

    for k, i in enumerate(views):
        c2w = np.linalg.inv(_as_homogeneous44(ext[i]))
        R = c2w[:3, :3]
        rays = np.linalg.inv(K[i]) @ pix                 # (3,hw) 方向（深度不要）
        d = R @ rays
        d = d / (np.linalg.norm(d, axis=0) + 1e-9)
        dx, dy, dz = d[0], -d[1], d[2]                   # y下→y上
        lon = np.arctan2(dx, dz)
        lat = np.arcsin(np.clip(dy, -1, 1))
        px = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.int64) % eq_w
        py = ((0.5 - lat / np.pi) * eq_h).astype(np.int64).clip(0, eq_h - 1)
        np.add.at(votes, (py, px, cats[k].ravel()), 1)

    cat_eq = votes.argmax(2).astype(np.int8)
    cat_eq[votes.max(2) == 0] = semseg.CATEGORIES.index("other")
    return cat_eq


def _box(center, size, yaw, rgba):
    m = trimesh.creation.box(extents=np.maximum(size, 0.05))
    T = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    T[:3, 3] = center
    m.apply_transform(T)
    m.visual.vertex_colors = np.tile(rgba, (len(m.vertices), 1))
    return m


def _cyl_y(center, radius, height, rgba):
    m = trimesh.creation.cylinder(radius=max(radius, 0.05), height=max(height, 0.1), sections=16)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    m.visual.vertex_colors = np.tile(rgba, (len(m.vertices), 1))
    return m


def build_collider_scene(prediction, view_index, viewpoints, camera_height_m=2.5,
                         eq_w=1024, floor_radius_m=30.0, wall_sectors=64,
                         min_object_px=80, progress=None):
    """意味equirect → 床/壁/物体コライダー（メートル）。(scene, info)。"""
    import cv2

    progress = progress or _noop
    H = float(camera_height_m)
    vp0 = sorted({int(v) for v in view_index})[0]
    cat_eq = build_semantic_equirect(prediction, view_index, vp0, eq_w=eq_w, progress=progress)
    eq_h, eq_w2 = cat_eq.shape
    _dirs, LAT = _equirect_dirs(eq_h, eq_w2)
    idx = {c: i for i, c in enumerate(semseg.CATEGORIES)}

    scene = trimesh.Scene()
    colliders = []

    # --- 1. 床（水平面 y=-H） ---
    progress("mesh", 0, 1, "床平面コライダー...")
    R = floor_radius_m
    scene.add_geometry(_box([0, -H, 0], [2 * R, 0.1, 2 * R], 0.0, [*CATEGORY_COLOR["floor"], 255]))
    colliders.append({"class": "floor", "type": "plane",
                      "center": [0.0, -H, 0.0], "size": [2 * R, 0.1, 2 * R], "yaw": 0.0})

    # --- 2. 壁（方位セクタ毎に床-壁境界の接地距離へ） ---
    progress("mesh", 0, 1, "壁コライダー...")
    is_wall = np.isin(cat_eq, [idx[c] for c in WALL_CATS])
    n_wall = 0
    sector = (((np.arange(eq_w2) / eq_w2) * wall_sectors).astype(int)) % wall_sectors
    for s in range(wall_sectors):
        cols = np.flatnonzero(sector == s)
        wmask = is_wall[:, cols]
        if wmask.sum() < 20:
            continue
        theta = (s + 0.5) / wall_sectors * 2 * np.pi - np.pi
        rows = np.flatnonzero(wmask.any(1))
        lat_base = float(LAT[rows.max(), cols[0]])
        sin_d = -np.sin(lat_base)
        dist = H / sin_d if sin_d > 0.03 else floor_radius_m
        dist = min(dist, floor_radius_m)
        lat_top = float(LAT[rows.min(), cols[0]])
        height = max(2.0, min(20.0, dist * (np.tan(max(lat_top, 0.01)) - np.tan(min(lat_base, -0.0)))))
        cx, cz = dist * np.sin(theta), dist * np.cos(theta)
        width = 2 * np.pi * dist / wall_sectors * 1.1
        scene.add_geometry(_box([cx, -H + height / 2, cz], [width, height, 0.3], theta,
                                [*CATEGORY_COLOR["wall"], 255]))
        colliders.append({"class": "wall", "type": "box",
                          "center": [cx, -H + height / 2, cz],
                          "size": [width, height, 0.3], "yaw": float(theta)})
        n_wall += 1

    # --- 3. 物体（車/人=箱, 柱=円柱）: 連結成分の接地点を床へ投影 ---
    progress("mesh", 0, 1, "物体コライダー...")
    n_obj = 0
    for cat in (OBJECT_BOX_CATS | OBJECT_CYL_CATS):
        mask = (cat_eq == idx[cat]).astype(np.uint8)
        if int(mask.sum()) < min_object_px:
            continue
        nlab, lab = cv2.connectedComponents(mask)
        for li in range(1, nlab):
            ys, xs = np.where(lab == li)
            if len(ys) < min_object_px:
                continue
            yb = int(ys.max())
            lat_b = float(LAT[yb, int(np.median(xs[ys == yb]))])
            theta = (float(np.median(xs)) / eq_w2) * 2 * np.pi - np.pi
            sin_d = -np.sin(lat_b)
            if sin_d <= 0.03:
                continue
            dist = min(H / sin_d, floor_radius_m)
            cx, cz = dist * np.sin(theta), dist * np.cos(theta)
            ang_w = (xs.max() - xs.min()) / eq_w2 * 2 * np.pi
            ang_h = (ys.max() - ys.min()) / eq_h * np.pi
            wsz = float(max(0.3, min(12.0, ang_w * dist)))
            hsz = float(max(0.5, min(8.0, np.tan(min(ang_h, 1.2)) * dist)))
            rgba = [*CATEGORY_COLOR[cat], 255]
            if cat in OBJECT_CYL_CATS:
                scene.add_geometry(_cyl_y([cx, -H + hsz / 2, cz], wsz / 2, hsz, rgba))
                ctype = "cylinder"
            else:
                scene.add_geometry(_box([cx, -H + hsz / 2, cz], [wsz, hsz, wsz], theta, rgba))
                ctype = "box"
            colliders.append({"class": cat, "type": ctype,
                              "center": [cx, -H + hsz / 2, cz],
                              "size": [wsz, hsz, wsz], "yaw": float(theta)})
            n_obj += 1

    info = {
        "representation": "colliders", "method": "colliders",
        "viewpoints": len(sorted({int(v) for v in view_index})),
        "camera_height_m": H, "eq_w": eq_w,
        "collider_counts": {"floor": 1, "wall": n_wall, "object": n_obj},
        "colliders": colliders,
        "vertex_count": int(sum(len(m.vertices) for m in scene.geometry.values())),
        "face_count": int(sum(len(m.faces) for m in scene.geometry.values())),
    }
    return scene, info
