"""視差マップ + 元画像から、歩ける 3D メッシュ(glb)を生成する。

手順:
1. 視差を相対的な距離(メートル近似)へ変換 (大きい視差=近い)。
2. 仮定した水平 FOV からピンホールカメラの内部パラメータを作り、各ピクセルを
   3D 空間へ逆投影 → 頂点(点群)。
3. 深度グリッドの隣接ピクセルを三角形で接続してメッシュ化。深度が急に変わる
   境界(オクルージョン境界)は分断して、引き伸ばされた面を作らない。
4. 頂点色を元画像から付与。

座標系は glTF/Three.js と同じ Y-up・右手系。カメラは原点で -Z を向く。
"""

import numpy as np
from PIL import Image
import trimesh

MAX_WIDTH = 480  # 頂点数を抑えるための最大横幅
DEFAULT_FOV_DEG = 75.0
NEAR_M = 1.0  # 最も近い点の距離 (m)
FAR_M = 25.0  # 最も遠い点の距離 (m)
DISCONTINUITY_RATIO = 0.08  # この比率を超える深度差の面は分断


def _resize(image: Image.Image, disparity: np.ndarray, max_width: int):
    width, height = image.size
    if width <= max_width:
        return image, disparity.astype(np.float32), width, height

    new_w = max_width
    new_h = int(round(height * new_w / width))
    image2 = image.resize((new_w, new_h), Image.BILINEAR)
    disp_img = Image.fromarray(disparity.astype(np.float32))
    disp2 = np.asarray(disp_img.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    return image2, disp2, new_w, new_h


def disparity_to_depth(
    disparity: np.ndarray, disp_range: tuple[float, float] | None = None
) -> np.ndarray:
    """視差(大=近)を距離(小=近)へ変換し、[NEAR_M, FAR_M] に正規化する。

    disp_range を渡すと、その共通の (min, max) で正規化する。パノラマ合成で
    複数視点のスケールを揃えるために使う。
    """
    d = disparity.astype(np.float32)
    if disp_range is None:
        dmin, dmax = float(d.min()), float(d.max())
    else:
        dmin, dmax = disp_range
    if dmax - dmin < 1e-6:
        norm = np.zeros_like(d)
    else:
        norm = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0)  # 0..1 (1=最も近い)
    far_to_near = 1.0 - norm  # 0=近い, 1=遠い
    return NEAR_M + far_to_near * (FAR_M - NEAR_M)


def _rotate_y(verts: np.ndarray, yaw_deg: float) -> np.ndarray:
    """頂点を Y 軸まわりに yaw_deg 度回転する (Street View の heading 反映用)。"""
    if not yaw_deg:
        return verts
    theta = np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    return verts @ rot.T


def reconstruct_mesh(
    image: Image.Image,
    disparity: np.ndarray,
    fov_deg: float = DEFAULT_FOV_DEG,
    max_width: int = MAX_WIDTH,
    disp_range: tuple[float, float] | None = None,
    yaw_deg: float = 0.0,
):
    """(trimesh.Trimesh, info dict) を返す。

    disp_range: 複数視点で共通の視差スケールを使う場合に指定。
    yaw_deg: 生成メッシュを Y 軸まわりに回転（撮影方位の反映）。
    """
    image = image.convert("RGB")
    image, disparity, width, height = _resize(image, disparity, max_width)
    depth = disparity_to_depth(disparity, disp_range=disp_range)
    rgb = np.asarray(image, dtype=np.uint8)  # (H, W, 3)

    # ピンホール内部パラメータ
    fov = np.radians(fov_deg)
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0

    us = np.arange(width)
    vs = np.arange(height)
    uu, vv = np.meshgrid(us, vs)  # (H, W)
    z = depth
    x = (uu - cx) / fx * z
    y = -(vv - cy) / fy * z  # 画像の下方向 → ワールドの -Y
    verts = np.stack([x, y, -z], axis=-1).reshape(-1, 3).astype(np.float32)
    verts = _rotate_y(verts, yaw_deg)

    colors = rgb.reshape(-1, 3)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    colors = np.concatenate([colors, alpha], axis=1)

    # グリッド接続 (各クアッドを2三角形に)
    idx = np.arange(height * width).reshape(height, width)
    tl = idx[:-1, :-1].ravel()
    tr = idx[:-1, 1:].ravel()
    bl = idx[1:, :-1].ravel()
    br = idx[1:, 1:].ravel()

    d_tl = depth[:-1, :-1].ravel()
    d_tr = depth[:-1, 1:].ravel()
    d_bl = depth[1:, :-1].ravel()
    d_br = depth[1:, 1:].ravel()
    d_max = np.maximum.reduce([d_tl, d_tr, d_bl, d_br])
    d_min = np.minimum.reduce([d_tl, d_tr, d_bl, d_br])
    valid = (d_max - d_min) < DISCONTINUITY_RATIO * (FAR_M - NEAR_M)

    tri1 = np.stack([tl, bl, tr], axis=1)[valid]
    tri2 = np.stack([tr, bl, br], axis=1)[valid]
    faces = np.concatenate([tri1, tri2], axis=0)

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=colors, process=False
    )

    info = {
        "width": int(width),
        "height": int(height),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces.shape[0]),
        "fov_deg": float(fov_deg),
        "yaw_deg": float(yaw_deg),
    }
    return mesh, info
