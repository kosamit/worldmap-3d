"""② 動画拡散で遮蔽補完（ViewCrafter 系）の足回り。

パイプライン:
  1. DA3 点群を構築（build_multiview_pointcloud を再利用）
  2. カメラ軌道に沿って点群をレンダ → 「穴あき」フレーム＋穴マスク＋各フレームのカメラ
  3. 動画拡散モデルで穴を時間的に一貫して生成補完（complete_video, 別途）
  4. 補完フレームに DA3 を再適用して 3D へ融合（refuse, 別途）

このモジュールは Phase1（軌道レンダラ）。点群レンダは torch で GPU z-buffer 実装。
穴（どのカメラからも見えない遮蔽部）は valid=False の画素として出力され、これが
生成補完すべき領域になる。
"""
from __future__ import annotations

import numpy as np

from .reconstruct_da3 import _noop, build_multiview_pointcloud


def build_cloud(prediction, view_index, viewpoints, max_width=400,
                conf_percentile=10.0, drop_sky=True, anchor_gps=True,
                far_clip_m=0.0, height_clip_m=40.0, max_points=600000,
                progress=None):
    """DA3 予測 → (points(M,3) float32, colors(M,3) uint8)。world は y-up・整地済み。"""
    progress = progress or _noop
    progress("mesh", 0, 1, "点群生成（動画拡散補完用）...")
    cloud_scene, _ = build_multiview_pointcloud(
        prediction, view_index, viewpoints, max_width=max_width,
        conf_percentile=conf_percentile, drop_sky=drop_sky, anchor_gps=anchor_gps,
        mesh=False, far_clip_m=far_clip_m, height_clip_m=height_clip_m,
        level_ground=True, max_points=max_points)
    g = list(cloud_scene.geometry.values())[0]
    pts = np.asarray(g.vertices, np.float32)
    cols = np.asarray(g.colors)[:, :3].astype(np.uint8)
    return pts, cols


def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """eye から target を見る c2w(4x4)。カメラ規約 OpenCV: x=右, y=下, z=前。world up=+Y。"""
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f) + 1e-9
    up = np.asarray(up, np.float64)
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-9
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0] = r
    c2w[:3, 1] = -u  # world up=+Y に対しカメラ y は下向き
    c2w[:3, 2] = f
    c2w[:3, 3] = eye
    return c2w


def make_trajectory(points, n_frames=25, H=512, W=512, fov_deg=60.0,
                    kind="dolly_sweep", lateral=0.25, forward_span=(0.15, 0.65)):
    """点群分布から歩行風カメラ軌道を生成 → [(K(3x3), w2c(4x4))]。

    kind: dolly_sweep=前進しつつ左右に振って視差で遮蔽を露出 / orbit=中心周回。
    """
    pts = np.asarray(points, np.float64)
    c = np.median(pts, axis=0)
    # 床と目線高さ（y-up）
    y = pts[:, 1]
    floor = np.percentile(y, 5)
    top = np.percentile(y, 95)
    eye_y = floor + 0.28 * (top - floor)
    # 水平の主軸 = 廊下方向（x,z の PCA）
    xz = pts[:, [0, 2]] - c[[0, 2]]
    cov = xz.T @ xz / max(len(xz), 1)
    w_, v_ = np.linalg.eigh(cov)
    fwd2 = v_[:, np.argmax(w_)]
    forward = np.array([fwd2[0], 0.0, fwd2[1]])
    forward /= np.linalg.norm(forward) + 1e-9
    side = np.array([-forward[2], 0.0, forward[0]])  # 水平直交
    # 前進範囲（点群の前後広がりに収める）
    s = (pts - c) @ forward
    s0 = np.percentile(s, forward_span[0] * 100)
    s1 = np.percentile(s, forward_span[1] * 100)
    extent = max(np.percentile(s, 90) - np.percentile(s, 10), 1.0)
    amp = lateral * extent

    fx = fy = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float64)

    cams = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if kind == "orbit":
            ang = 2 * np.pi * t
            eye = c.copy(); eye[1] = eye_y
            eye = eye + amp * (np.cos(ang) * side + np.sin(ang) * forward)
            target = c.copy(); target[1] = eye_y
        else:  # dolly_sweep
            along = s0 + (s1 - s0) * t
            lat = amp * np.sin(2 * np.pi * t)
            eye = c + forward * along + side * lat
            eye[1] = eye_y
            target = c + forward * (along + extent * 0.5)  # 少し先を見る
            target[1] = eye_y
        c2w = _look_at(eye, target)
        w2c = np.linalg.inv(c2w)
        cams.append((K.copy(), w2c))
    return cams


def render_pointcloud(points, colors, K, w2c, H, W, splat=1, device=None):
    """点群を1カメラへ z-buffer スプラットレンダ → (rgb uint8(H,W,3), valid bool(H,W), depth(H,W))。

    valid=False の画素＝どの点も投影されない穴（遮蔽/未撮影）＝生成補完すべき領域。
    depth は穴で inf（カメラ前方距離 z）。
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=dev)
    C = torch.as_tensor(np.asarray(colors), dtype=torch.uint8, device=dev)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    Rt = torch.as_tensor(w2c[:3, :3], dtype=torch.float32, device=dev)
    tt = torch.as_tensor(w2c[:3, 3], dtype=torch.float32, device=dev)

    cam = P @ Rt.T + tt  # (M,3)
    z = cam[:, 2]
    front = z > 0.05
    cam = cam[front]; col = C[front]; z = z[front]
    uvw = cam @ Kt.T
    u = (uvw[:, 0] / uvw[:, 2]).round().long()
    v = (uvw[:, 1] / uvw[:, 2]).round().long()

    colbuf = torch.zeros((H * W, 3), dtype=torch.uint8, device=dev)
    depth = torch.full((H * W,), float("inf"), device=dev)
    order = torch.argsort(z, descending=True)  # 遠い順→最後に手前が残る
    uo, vo, co, zo = u[order], v[order], col[order], z[order]
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            uu = uo + du; vv = vo + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            flat = (vv[ok] * W + uu[ok])
            colbuf[flat] = co[ok]
            depth[flat] = torch.minimum(depth[flat], zo[ok])
    rgb = colbuf.reshape(H, W, 3).cpu().numpy()
    dep = depth.reshape(H, W).cpu().numpy()
    valid = np.isfinite(dep)
    return rgb, valid, dep


def render_trajectory(points, colors, cams, H=512, W=512, splat=1, progress=None):
    """軌道全フレームをレンダ → (frames(N,H,W,3 uint8), holes(N,H,W bool), depths(N,H,W), cams)。

    holes は「穴(True=補完すべき)」。フレームは穴を黒で表現。depths は穴で inf。
    """
    progress = progress or _noop
    frames = np.zeros((len(cams), H, W, 3), np.uint8)
    holes = np.zeros((len(cams), H, W), bool)
    depths = np.full((len(cams), H, W), np.inf, np.float32)
    for i, (K, w2c) in enumerate(cams):
        progress("mesh", i, len(cams), f"軌道レンダ {i + 1}/{len(cams)} ...")
        rgb, valid, dep = render_pointcloud(points, colors, K, w2c, H, W, splat)
        frames[i] = rgb
        holes[i] = ~valid
        depths[i] = dep
    return frames, holes, depths, cams


def _fill_hole_depth(depth, hole, win=21):
    """穴の深度を「周囲の遠い側」で補完（遮蔽の背後＝背景深度になるように）。"""
    import cv2

    d = depth.copy().astype(np.float32)
    d[hole] = 0.0
    # 既知画素の最大(遠)を窓で広げて穴へ。背景(遠)で埋める＝柱裏が背景深度に。
    far = cv2.dilate(d, np.ones((win, win), np.uint8))
    out = d.copy()
    out[hole] = far[hole]
    # まだ0(周囲も穴)の所は最近傍既知で埋める
    still = hole & (out <= 0)
    if still.any():
        known = (~hole & (depth > 0)).astype(np.uint8)
        if known.any():
            _, lab = cv2.distanceTransformWithLabels(
                1 - known, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            ki = np.flatnonzero(known.ravel())
            lk = lab.ravel()[ki]
            order = np.argsort(lk)
            lut = np.zeros(lk.max() + 1, np.float32)
            lut[lk[order]] = depth.ravel()[ki][order]
            out.ravel()[still.ravel()] = lut[lab.ravel()[still.ravel()]]
    return out


def inpaint_and_refuse(frames, holes, depths, cams, max_new=400000, progress=None):
    """各フレームの穴を LaMa で色生成＋深度を遠側補完 → 逆投影して新規点群を得る。

    返り値: (new_points(M,3) float32, new_colors(M,3) uint8)。元点群に足して再メッシュする。
    """
    from . import segment

    progress = progress or _noop
    new_pts = []
    new_cols = []
    N = len(cams)
    for i, (K, w2c) in enumerate(cams):
        hole = holes[i]
        if not hole.any():
            continue
        progress("mesh", i, N, f"穴を生成補完＋逆投影 {i + 1}/{N} ...")
        # 色: LaMa（失敗時 OpenCV）
        try:
            filled = segment.inpaint_masked(frames[i][None], hole[None])[0]
        except Exception:
            import cv2
            filled = cv2.inpaint(frames[i], (hole.astype(np.uint8) * 255), 5,
                                 cv2.INPAINT_TELEA)
        # 深度: 遠側で補完
        dep = _fill_hole_depth(depths[i], hole)
        H, W = hole.shape
        vy, vx = np.nonzero(hole)
        z = dep[vy, vx]
        ok = np.isfinite(z) & (z > 0)
        vy, vx, z = vy[ok], vx[ok], z[ok]
        if not len(vy):
            continue
        Kinv = np.linalg.inv(K)
        pix = np.stack([vx, vy, np.ones_like(vx)], 0).astype(np.float64)
        camp = (Kinv @ pix) * z[None, :]  # (3,M) カメラ座標
        c2w = np.linalg.inv(w2c)
        world = (c2w[:3, :3] @ camp).T + c2w[:3, 3]
        new_pts.append(world.astype(np.float32))
        new_cols.append(filled[vy, vx])
    if not new_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    P = np.concatenate(new_pts, 0)
    C = np.concatenate(new_cols, 0)
    if len(P) > max_new:  # 間引き
        idx = np.random.default_rng(0).choice(len(P), max_new, replace=False)
        P, C = P[idx], C[idx]
    return P, C
