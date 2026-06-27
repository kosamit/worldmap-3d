"""DA3 点群 → 3D Gaussian Splatting(.ply) 初期化（学習なし）。

DA3 マルチビュー予測が出す「深度×レイのメトリック点群」と色を、そのまま 3D Gaussian
(INRIA 形式 .ply) に変換する。各点を 1 つの等方ガウスにし、色は SH(0次)、スケールは
近傍点間距離、不透明度は固定、回転は単位四元数で初期化する。CUDA ラスタライザの
ビルドや反復最適化は不要で、既存の点群資産をそのまま写実 splat 表現にする。

docs/new_paper_concept.md の「見た目=3DGS」の第一段（初期化型）。最適化(gsplat)を
将来上乗せする場合も、この .ply を初期点群として使える。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .reconstruct_da3 import _noop, build_multiview_pointcloud

# SH 0次の基底係数（3DGS 標準）。色 c∈[0,1] → f_dc = (c-0.5)/C0。
SH_C0 = 0.28209479177387814

# .ply に書く 3DGS プロパティ（INRIA gaussian-splatting 互換の並び）。
_PLY_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)


def _inverse_sigmoid(x: float) -> float:
    return float(np.log(x / (1.0 - x)))


def _per_point_scale(pts: np.ndarray, fallback: float) -> np.ndarray:
    """各点の近傍点間距離(=ガウスの広がり)を返す。隙間をちょうど埋める大きさ。"""
    if len(pts) < 2:
        return np.full(len(pts), max(fallback, 1e-3), np.float32)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        d = np.asarray(pcd.compute_nearest_neighbor_distance())
    except Exception:
        d = np.full(len(pts), fallback)
    d = np.nan_to_num(d, nan=fallback, posinf=fallback, neginf=fallback)
    # 外れ値（孤立点で巨大化）を抑える。中央値の数倍でクランプ。
    med = float(np.median(d[d > 0])) if np.any(d > 0) else max(fallback, 1e-3)
    return np.clip(d, med * 0.3, med * 4.0).astype(np.float32)


def _write_gaussian_ply(path, pts, colors, scales, opacity=0.85) -> int:
    """点群＋色 → 3DGS .ply を書き出し、ガウス数を返す。"""
    from plyfile import PlyData, PlyElement

    n = len(pts)
    f_dc = (colors.astype(np.float32) / 255.0 - 0.5) / SH_C0     # (n,3) SH DC
    log_scale = np.log(np.maximum(scales, 1e-6))[:, None].repeat(3, axis=1)  # 等方
    op = np.full((n, 1), _inverse_sigmoid(float(opacity)), np.float32)
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))  # 単位四元数(w,x,y,z)
    normals = np.zeros((n, 3), np.float32)

    arr = np.concatenate(
        [pts.astype(np.float32), normals, f_dc, op, log_scale, rot], axis=1
    )
    dtype = [(name, "f4") for name in _PLY_FIELDS]
    rec = np.empty(n, dtype=dtype)
    for i, name in enumerate(_PLY_FIELDS):
        rec[name] = arr[:, i]
    el = PlyElement.describe(rec, "vertex")
    PlyData([el]).write(str(path))
    return n


def build_gaussian_scene(prediction, view_index, viewpoints, params, progress=None,
                         opacity=0.85, splat_tmp_path=None):
    """DA3 予測 → 3D Gaussian .ply（初期化型）。(scene, info) を返す。

    scene は点群(glTF Y-up)で GLB フォールバック兼用。.ply は splat_tmp_path に書き、
    info["_splat_tmp"] にそのパスを入れて返す（呼び出し側が scene ディレクトリへ移す）。
    """
    progress = progress or _noop

    # DA3 のメトリック点群（GPSアンカー・空/遠景クリップ済み）をそのまま使う。
    progress("mesh", 0, 1, "点群を構築中（3DGS初期化）...")
    scene, info = build_multiview_pointcloud(
        prediction, view_index, viewpoints,
        max_width=params["max_width"],
        conf_percentile=params["conf_percentile"],
        ensure_percentile=params["ensure_percentile"],
        drop_sky=params["drop_sky"],
        filter_black_bg=params["filter_black_bg"],
        filter_white_bg=params["filter_white_bg"],
        anchor_gps=params["anchor_gps"],
        far_clip_m=params["far_clip_m"],
        height_clip_m=params["height_clip_m"],
        mesh=False,
        progress=progress,
    )

    clouds = [g for g in scene.geometry.values() if isinstance(g, trimesh.PointCloud)]
    if not clouds:
        raise ValueError("点群が空のため 3DGS を初期化できませんでした")
    pc = clouds[0]
    pts = np.asarray(pc.vertices, np.float32)
    colors = np.asarray(pc.colors, np.uint8)[:, :3]

    progress("mesh", 0, 1, f"ガウスのスケールを推定中（{len(pts)}点）...")
    fallback = float(info.get("point_size") or 0.05)
    scales = _per_point_scale(pts, fallback)

    if splat_tmp_path is None:
        import tempfile
        splat_tmp_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False).name
    progress("mesh", 0, 1, "3D Gaussian (.ply) を書き出し中 ...")
    ngauss = _write_gaussian_ply(splat_tmp_path, pts, colors, scales, opacity=opacity)

    info = dict(info)
    info["representation"] = "gaussian"
    info["method"] = "gaussian"
    info["gaussian_count"] = int(ngauss)
    info["splat_format"] = "ply_inria_sh0"
    info["_splat_tmp"] = str(splat_tmp_path)
    return scene, info
