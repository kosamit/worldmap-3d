"""DA3 点群 → 3D Gaussian Splatting(.ply) 初期化（学習なし）。

DA3 マルチビュー予測が出す「深度×レイのメトリック点群」と色を、そのまま 3D Gaussian
(INRIA 形式 .ply) に変換する。各点を 1 つの等方ガウスにし、色は SH(0次)、スケールは
近傍点間距離、不透明度は固定、回転は単位四元数で初期化する。CUDA ラスタライザの
ビルドや反復最適化は不要で、既存の点群資産をそのまま写実 splat 表現にする。

docs/new_paper_concept.md の「見た目=3DGS」の第一段（初期化型）。最適化(gsplat)を
将来上乗せする場合も、この .ply を初期点群として使える。
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh

from .reconstruct_da3 import _noop, build_multiview_pointcloud

logger = logging.getLogger(__name__)

# SH 0次の基底係数（3DGS 標準）。色 c∈[0,1] → f_dc = (c-0.5)/C0。
SH_C0 = 0.28209479177387814

# 面向きガウス(surfel)化: 法線方向の厚みを接平面方向に対し何倍にするか。
# 小さいほど平たい円盤になり「点の塊」→「面」に見える。
_SPLAT_THIN_RATIO = 0.18

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


def _estimate_normals(pts: np.ndarray) -> np.ndarray:
    """open3d で各点の法線(N,3)を推定。失敗時は例外を上げる（握りつぶさない）。"""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    return np.asarray(pcd.normals, np.float32)


def _quat_from_matrices(R: np.ndarray) -> np.ndarray:
    """回転行列バッチ(N,3,3) → 単位四元数(N,4) xyzw（three.js 規約）。

    4 つの定式化を全計算し、対角/トレースで数値安定なものを点ごとに選ぶ。
    """
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    def _pack(x, y, z, w):
        return np.stack([x, y, z, w], axis=1)

    sA = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    A = _pack((m21 - m12) / sA, (m02 - m20) / sA, (m10 - m01) / sA, 0.25 * sA)
    sB = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    B = _pack(0.25 * sB, (m01 + m10) / sB, (m02 + m20) / sB, (m21 - m12) / sB)
    sC = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    C = _pack((m01 + m10) / sC, 0.25 * sC, (m12 + m21) / sC, (m02 - m20) / sC)
    sD = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    D = _pack((m02 + m20) / sD, (m12 + m21) / sD, 0.25 * sD, (m10 - m01) / sD)

    condA = tr > 0
    condB = (~condA) & (m00 >= m11) & (m00 >= m22)
    condC = (~condA) & (~condB) & (m11 >= m22)
    q = np.where(condA[:, None], A,
                 np.where(condB[:, None], B, np.where(condC[:, None], C, D)))
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def _oriented_splat_arrays(scales, normals):
    """法線→各点ローカル枠(接1,接2,法線)を作り、面向き surfel 用の
    scale(N,3) と drei 回転バイト(N,4) を返す。法線が不正な点は等方フォールバック。

    drei は decode 時に quat を invert するので、ここでは目標回転 R の逆四元数を
    バイト化する。scale 第3軸を法線方向(薄)に割り当てると、復元される世界座標の
    共分散 R^T S^2 R の最小分散軸が法線に一致する（verify_orient.py で検証済）。
    """
    n_pts = len(scales)
    nm = np.asarray(normals, np.float32)
    norm = np.linalg.norm(nm, axis=1)
    good = np.isfinite(norm) & (norm > 0.5)
    n_unit = nm / np.where(norm[:, None] > 1e-6, norm[:, None], 1.0)
    n_unit = np.where(good[:, None], n_unit, np.array([0, 1, 0], np.float32))

    # 法線が ±Y に近い点だけ基準upを X にして接ベクトルの退化を避ける。
    up = np.where((np.abs(n_unit[:, 1]) < 0.95)[:, None],
                  np.array([0, 1, 0], np.float32), np.array([1, 0, 0], np.float32))
    t1 = np.cross(up, n_unit)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(n_unit, t1)
    R = np.stack([t1, t2, n_unit], axis=1)          # (N,3,3) 行=ローカル軸
    q = _quat_from_matrices(R)                       # mat(q)=R
    q_dec = q.copy()
    q_dec[:, 0:3] *= -1.0                             # 逆四元数(=共役, 単位)
    xd, yd, zd, wd = q_dec[:, 0], q_dec[:, 1], q_dec[:, 2], q_dec[:, 3]
    b = np.stack([128 - 128 * wd, 128 - 128 * xd,
                  128 + 128 * yd, 128 + 128 * zd], axis=1)
    rot = np.clip(np.round(b), 0, 255).astype(np.uint8)

    s = np.maximum(scales, 1e-6).astype(np.float32)
    scale3 = np.stack([s, s, s * _SPLAT_THIN_RATIO], axis=1)  # 第3軸=法線方向を薄く

    # 不正法線の点は等方ガウス＋単位回転に戻す（surfel化しない）。
    bad = ~good
    if np.any(bad):
        scale3[bad] = s[bad, None]
        rot[bad] = np.array([255, 128, 128, 128], np.uint8)
    return scale3, rot, int(good.sum())


def _write_drei_splat(path, pts, colors, scales, normals=None, opacity=0.85):
    """点群＋色 → drei `<Splat>` 互換 .splat（32B/頂点）。(頂点数, oriented) を返す。

    レイアウト(各32バイト): pos f32×3 / scale f32×3 / RGBA u8×4 / rot u8×4。
    drei は描画時に world=(x,-y,-z) と解釈する＝INRIA(y下/z前)前提。本点群は
    three.js(y上/z後)なので位置を [1,-1,-1] で INRIA 系へ変換して書く。法線が
    与えられれば面向き surfel(薄い円盤)に、無ければ等方ガウス(単位回転)にする。
    """
    n = len(pts)
    pos = (pts.astype(np.float32) * np.array([1.0, -1.0, -1.0], np.float32))
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, :3] = colors.astype(np.uint8)
    rgba[:, 3] = int(round(float(opacity) * 255.0))

    oriented = normals is not None and len(normals) == n
    if oriented:
        scl, rot, _ = _oriented_splat_arrays(scales, normals)
    else:
        scl = np.maximum(scales, 1e-6).astype(np.float32)[:, None].repeat(3, axis=1)
        rot = np.tile(np.array([255, 128, 128, 128], np.uint8), (n, 1))

    row = np.empty((n, 32), np.uint8)
    row[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
    row[:, 12:24] = np.ascontiguousarray(scl, np.float32).view(np.uint8).reshape(n, 12)
    row[:, 24:28] = rgba
    row[:, 28:32] = rot
    with open(path, "wb") as f:
        f.write(row.tobytes())
    return n, oriented


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
    # 面向き surfel 化のための法線推定。失敗は握りつぶさずログ＋metaフラグで可視化し、
    # 等方ガウスにフォールバックする（[[feedback-no-silent-fallback]]）。
    progress("mesh", 0, 1, f"法線を推定中（surfel化, {len(pts)}点）...")
    normals = None
    orient_error = None
    try:
        normals = _estimate_normals(pts)
    except Exception as e:  # open3d 失敗 → 等方ガウスで継続
        orient_error = f"{type(e).__name__}: {e}"
        logger.warning("法線推定に失敗、等方ガウスにフォールバック: %s", orient_error)

    progress("mesh", 0, 1, "3D Gaussian (.ply / .splat) を書き出し中 ...")
    ngauss = _write_gaussian_ply(splat_tmp_path, pts, colors, scales, opacity=opacity)
    # フロント(drei <Splat>)が読めるのは 32B/頂点の .splat のみ。INRIA .ply は
    # 将来の gsplat 最適化の初期点群として併存させ、配信は .splat を使う。
    import tempfile
    drei_tmp = tempfile.NamedTemporaryFile(suffix=".splat", delete=False).name
    _, oriented = _write_drei_splat(
        drei_tmp, pts, colors, scales, normals=normals, opacity=opacity
    )

    info = dict(info)
    info["representation"] = "gaussian"
    info["method"] = "gaussian"
    info["gaussian_count"] = int(ngauss)
    info["splat_format"] = "splat_v1_32b"
    info["splat_oriented"] = bool(oriented)        # surfel化できたか（法線推定成否）
    if orient_error:
        info["splat_orient_error"] = orient_error
    info["_splat_tmp"] = str(splat_tmp_path)      # INRIA .ply（保管用）
    info["_splat_tmp_drei"] = str(drei_tmp)       # drei .splat（配信用）
    return scene, info
