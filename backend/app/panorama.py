"""複数方位の Street View 画像を 1 つの 360° メッシュへ合成する。

各画像を個別に正規化すると継ぎ目で深度スケールがずれるため、全方位の視差を
集めて共通の (min, max) で正規化し、撮影方位(heading)ぶん回転させて連結する。
"""

import trimesh

from .depth import estimate_depth
from .reconstruct import reconstruct_mesh


def build_panorama(images_with_headings, fov_deg, max_width=None):
    """images_with_headings: list[(PIL.Image, heading_deg)]。

    (trimesh.Trimesh, info dict) を返す。
    """
    if not images_with_headings:
        raise ValueError("画像が 1 枚もありません")

    # 全方位の視差を先に推定 → 共通スケールを決める
    disparities = [estimate_depth(img) for img, _ in images_with_headings]
    global_min = min(float(d.min()) for d in disparities)
    global_max = max(float(d.max()) for d in disparities)
    disp_range = (global_min, global_max)

    meshes = []
    total_vertices = 0
    total_faces = 0
    for (image, heading), disparity in zip(images_with_headings, disparities):
        kwargs = {"fov_deg": fov_deg, "disp_range": disp_range, "yaw_deg": heading}
        if max_width is not None:
            kwargs["max_width"] = max_width
        mesh, info = reconstruct_mesh(image, disparity, **kwargs)
        meshes.append(mesh)
        total_vertices += info["vertex_count"]
        total_faces += info["face_count"]

    scene = trimesh.util.concatenate(meshes)

    info = {
        "views": len(meshes),
        "vertex_count": total_vertices,
        "face_count": total_faces,
        "fov_deg": float(fov_deg),
    }
    return scene, info
