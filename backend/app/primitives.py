"""D1: 構造プリミティブ化（ゲームのブロックアウト風）。

点群から平面(壁/床/天井)を RANSAC で抽出して薄い箱に、残りを DBSCAN で
クラスタ化して箱/円柱にフィットする。写真深度のぐにゃぐにゃ面ではなく、
独立した「ちゃんとしたオブジェクト」(平面・柱・什器の塊)として表現する。
open3d で完結（重い生成モデル不要）。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .reconstruct_da3 import _noop, build_multiview_pointcloud


def _box_from_obb(center, R, extent, rgba):
    """有向境界箱(OBB)から色付き箱メッシュを作る。"""
    extent = np.maximum(np.asarray(extent, float), 0.05)  # 退化防止
    box = trimesh.creation.box(extents=extent)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    box.apply_transform(T)
    box.visual.vertex_colors = np.tile(rgba, (len(box.vertices), 1))
    return box


def _cylinder(center, radius, height, rgba):
    cyl = trimesh.creation.cylinder(radius=max(radius, 0.05), height=max(height, 0.1), sections=20)
    cyl.apply_translation(center)  # 既定で軸=Z上向き... Y上へ回す
    return cyl


def _cylinder_y(center, radius, height, rgba):
    """Y軸(上)に立つ色付き円柱。"""
    cyl = trimesh.creation.cylinder(radius=max(radius, 0.05), height=max(height, 0.1), sections=24)
    # trimesh cylinder は Z 軸方向 → X軸まわり90°で Y 軸へ
    Rx = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    cyl.apply_transform(Rx)
    cyl.apply_translation(center)
    cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
    return cyl


def build_primitive_mesh(prediction, view_index, viewpoints, max_planes=10,
                         plane_dist=0.12, min_plane_pts=400, cluster_eps=0.6,
                         cluster_min=60, progress=None):
    """点群 → 平面(薄い箱)＋クラスタ(箱/円柱) の集合（trimesh.Scene）。"""
    import open3d as o3d

    progress = progress or _noop
    progress("mesh", 0, 1, "点群生成（プリミティブ化用）...")
    cloud_scene, _ = build_multiview_pointcloud(
        prediction, view_index, viewpoints, max_width=300, conf_percentile=10.0,
        drop_sky=True, anchor_gps=True, mesh=False, far_clip_m=0.0, height_clip_m=40.0,
        level_ground=True, max_points=300000)
    g = list(cloud_scene.geometry.values())[0]
    pts = np.asarray(g.vertices, np.float64)
    cols = np.asarray(g.colors)[:, :3].astype(np.float64) / 255.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    pc = pc.voxel_down_sample(voxel_size=0.05)

    scene = trimesh.Scene()
    n_planes = 0
    n_objs = 0
    rest = pc

    # 1. 平面抽出（床・天井・壁）。薄い箱として配置。
    for _ in range(max_planes):
        if len(rest.points) < min_plane_pts:
            break
        progress("mesh", 0, 1, f"平面抽出 {n_planes + 1} ...")
        try:
            _, inliers = rest.segment_plane(plane_dist, ransac_n=3, num_iterations=400)
        except Exception:
            break
        if len(inliers) < min_plane_pts:
            break
        inl = rest.select_by_index(inliers)
        obb = inl.get_oriented_bounding_box()
        rgba = np.concatenate([(np.asarray(inl.colors).mean(0) * 255).astype(np.uint8), [255]])
        ext = np.array(obb.extent)
        ext[np.argmin(ext)] = 0.06  # 板状に薄く
        scene.add_geometry(_box_from_obb(np.array(obb.center), np.array(obb.R), ext, rgba))
        n_planes += 1
        rest = rest.select_by_index(inliers, invert=True)

    # 2. 残りをクラスタ化 → 箱/円柱
    if len(rest.points) >= cluster_min:
        labels = np.array(rest.cluster_dbscan(eps=cluster_eps, min_points=cluster_min))
        rp = np.asarray(rest.points)
        rc = np.asarray(rest.colors)
        for lab in range(labels.max() + 1):
            m = labels == lab
            if m.sum() < cluster_min:
                continue
            cl = rp[m]
            ext_aabb = cl.max(0) - cl.min(0)
            rgba = np.concatenate([(rc[m].mean(0) * 255).astype(np.uint8), [255]])
            horiz = float(np.hypot(ext_aabb[0], ext_aabb[2]))
            # 縦長で footprint が丸っぽい → 円柱（柱）。それ以外は有向境界箱（什器等）。
            if ext_aabb[1] > 1.5 * (horiz + 1e-6) and ext_aabb[1] > 1.2:
                scene.add_geometry(_cylinder_y(cl.mean(0), horiz / 2, ext_aabb[1], rgba))
            else:
                cpc = o3d.geometry.PointCloud()
                cpc.points = o3d.utility.Vector3dVector(cl)
                try:
                    obb = cpc.get_oriented_bounding_box()
                    scene.add_geometry(_box_from_obb(np.array(obb.center), np.array(obb.R), np.array(obb.extent), rgba))
                except Exception:
                    scene.add_geometry(_box_from_obb(cl.mean(0), np.eye(3), ext_aabb, rgba))
            n_objs += 1

    if not scene.geometry:
        raise ValueError("プリミティブが生成されませんでした（点群が薄すぎます）")
    # 全体を中央寄せ
    try:
        c = scene.bounds.mean(0)
        scene.apply_translation(-c)
    except Exception:
        pass

    info = {"representation": "primitives", "method": "primitive",
            "planes": n_planes, "objects": n_objs,
            "vertex_count": int(sum(len(m.vertices) for m in scene.geometry.values())),
            "face_count": int(sum(len(m.faces) for m in scene.geometry.values())),
            "viewpoints": len(viewpoints)}
    return scene, info
