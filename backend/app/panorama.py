"""複数方位の Street View 画像を 1 つの 360° メッシュへ合成する。

各画像を個別に正規化すると継ぎ目で深度スケールがずれるため、全方位の視差を
集めて共通の (min, max) で正規化し、撮影方位(heading)ぶん回転させて連結する。
"""

import trimesh

from .depth import estimate_depth
from .reconstruct import (
    DISCONTINUITY_RATIO,
    FAR_M,
    NEAR_M,
    reconstruct_mesh,
)


def _noop(*_args, **_kwargs):
    pass


def build_panorama(
    images_with_headings,
    fov_deg,
    max_width=None,
    near_m=NEAR_M,
    far_m=FAR_M,
    discontinuity_ratio=DISCONTINUITY_RATIO,
    depth_model=None,
    progress=None,
):
    """images_with_headings: list[(PIL.Image, heading_deg)]。

    near_m/far_m/discontinuity_ratio/depth_model で再構成パラメータを可変化。
    progress(phase, step, total, message) を渡すと、深度推定とメッシュ生成の
    各ステップで進捗を通知する。

    (trimesh.Trimesh, info dict) を返す。
    """
    if not images_with_headings:
        raise ValueError("画像が 1 枚もありません")
    progress = progress or _noop
    n = len(images_with_headings)

    # 全方位の視差を先に推定 → 共通スケールを決める
    disparities = []
    for i, (img, _) in enumerate(images_with_headings):
        progress("depth", i, n, f"深度推定 {i + 1}/{n}（モデルロード含む）")
        disparities.append(estimate_depth(img, model=depth_model))
    progress("depth", n, n, "深度推定 完了")

    global_min = min(float(d.min()) for d in disparities)
    global_max = max(float(d.max()) for d in disparities)
    disp_range = (global_min, global_max)

    meshes = []
    total_vertices = 0
    total_faces = 0
    for i, ((image, heading), disparity) in enumerate(
        zip(images_with_headings, disparities)
    ):
        progress("mesh", i, n, f"メッシュ生成 {i + 1}/{n}")
        kwargs = {
            "fov_deg": fov_deg,
            "disp_range": disp_range,
            "yaw_deg": heading,
            "near_m": near_m,
            "far_m": far_m,
            "discontinuity_ratio": discontinuity_ratio,
        }
        if max_width is not None:
            kwargs["max_width"] = max_width
        mesh, info = reconstruct_mesh(image, disparity, **kwargs)
        meshes.append(mesh)
        total_vertices += info["vertex_count"]
        total_faces += info["face_count"]
    progress("mesh", n, n, "メッシュ生成 完了")

    scene = trimesh.util.concatenate(meshes)

    info = {
        "views": len(meshes),
        "vertex_count": total_vertices,
        "face_count": total_faces,
        "fov_deg": float(fov_deg),
        "near_m": float(near_m),
        "far_m": float(far_m),
        "discontinuity_ratio": float(discontinuity_ratio),
        "max_width": int(max_width) if max_width is not None else None,
        "depth_model": depth_model,
    }
    return scene, info
