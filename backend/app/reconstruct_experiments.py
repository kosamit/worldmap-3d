"""3D再構成の品質改善・実験用バリアント。

主経路 (reconstruct_da3.build_multiview_pointcloud) を壊さずに、別手法を並べて
目視比較するための関数群。1回の DA3 推論 (prediction) を共有して各バリアントの
GLB を作る。

提供する手法:
  - smooth_depth_prediction   : 深度マップをエッジ保存平滑化（面のザラつき低減）
  - taubin_smooth_scene       : 生成メッシュを Taubin 平滑化
  - cloud_remove_outliers     : 点群の統計的外れ値除去
  - build_tsdf_mesh           : TSDF 融合（重なり層を1枚の面へ）
  - build_poisson_mesh        : Poisson 面再構成（滑らかな水密面）
"""

from __future__ import annotations

import numpy as np
import trimesh

from .reconstruct_da3 import (
    _alignment_transform,
    _as_homogeneous44,
    _level_ground,
    _metric_scale,
    build_multiview_pointcloud,
)


def smooth_depth_prediction(prediction: dict, diameter: int = 7, sigma_color: float = 0.04,
                            sigma_space: float = 5.0) -> dict:
    """各ビューの深度にエッジ保存（バイラテラル）平滑化をかけた新しい prediction を返す。

    深度ノイズ由来の面のザラつき・小スパイクを、オクルージョン境界を保ったまま低減。
    """
    import cv2

    depth = prediction["depth"].astype(np.float32)
    out = np.empty_like(depth)
    for i in range(depth.shape[0]):
        d = depth[i]
        # 相対深度なので中央値で正規化してからフィルタ（sigma_color をスケール非依存に）
        med = float(np.median(d[np.isfinite(d)])) or 1.0
        out[i] = cv2.bilateralFilter((d / med).astype(np.float32), diameter, sigma_color, sigma_space) * med
    new = dict(prediction)
    new["depth"] = out
    return new


def taubin_smooth_scene(scene: trimesh.Scene, iterations: int = 8, lamb: float = 0.5,
                        nu: float = -0.53) -> trimesh.Scene:
    """シーン内メッシュを Taubin 平滑化（収縮しない平滑化）。頂点色は保持。"""
    out = trimesh.Scene()
    for g in scene.geometry.values():
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            m = g.copy()
            try:
                trimesh.smoothing.filter_taubin(m, lamb=lamb, nu=nu, iterations=iterations)
            except Exception:
                pass
            out.add_geometry(m)
        else:
            out.add_geometry(g.copy())
    return out


def cloud_remove_outliers(scene: trimesh.Scene, nb_neighbors: int = 16, std_ratio: float = 2.0):
    """点群シーンに統計的外れ値除去（open3d）。浮遊した孤立点を落とす。"""
    import open3d as o3d

    out = trimesh.Scene()
    for g in scene.geometry.values():
        if isinstance(g, trimesh.PointCloud):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.asarray(g.vertices, np.float64))
            cols = np.asarray(g.colors)[:, :3].astype(np.float64) / 255.0
            pc.colors = o3d.utility.Vector3dVector(cols)
            pc2, _ = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            v = np.asarray(pc2.points)
            c = (np.asarray(pc2.colors) * 255).astype(np.uint8)
            rgba = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], axis=1)
            out.add_geometry(trimesh.PointCloud(vertices=v, colors=rgba))
        else:
            out.add_geometry(g.copy())
    return out


def _metric_depth_and_poses(prediction, view_index, viewpoints):
    """DA3 の相対深度・ポーズを GPS ベースのスケールでメートル化して返す。"""
    depth = prediction["depth"].astype(np.float32)
    ext = prediction["extrinsics"]
    n = depth.shape[0]
    c2ws = [np.linalg.inv(_as_homogeneous44(ext[i])) for i in range(n)]
    centers = np.array([c[:3, 3] for c in c2ws], dtype=np.float64)
    scale = _metric_scale(centers, list(view_index), viewpoints)
    if not scale or not np.isfinite(scale) or scale <= 0:
        # 相対のみ: 深度中央値が ~6m になるよう正規化
        med = float(np.median(depth[np.isfinite(depth)])) or 1.0
        scale = 6.0 / med
    return scale


def build_tsdf_mesh(prediction, view_index, viewpoints, voxel: float = 0.06,
                    sdf_trunc_vox: float = 5.0, depth_trunc: float = 45.0,
                    level_ground: bool = True, max_faces: int = 1_200_000, progress=None):
    """TSDF 融合で 72 層の重なりを1枚の面へ統合する。"""
    import open3d as o3d
    from .reconstruct_da3 import _decimate_mesh, _noop

    progress = progress or _noop
    depth = prediction["depth"].astype(np.float32)
    conf = prediction.get("conf")
    K = prediction["intrinsics"].astype(np.float64)
    ext = prediction["extrinsics"]
    imgs = prediction["processed_images"]
    n, h, w = depth.shape
    scale = _metric_depth_and_poses(prediction, view_index, viewpoints)

    # 信頼度の低い画素は深度を無効(0)化して融合から除外
    conf_thr = float(np.percentile(conf, 40)) if conf is not None else None

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * sdf_trunc_vox,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(n):
        progress("mesh", i, n, f"TSDF統合 {i+1}/{n}")
        d = depth[i].astype(np.float32) * scale  # メートル
        if conf_thr is not None:
            d = np.where(conf[i] >= conf_thr, d, 0.0).astype(np.float32)
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        color = o3d.geometry.Image(np.ascontiguousarray(imgs[i][:, :, :3]))
        depth_o = o3d.geometry.Image(d)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_o, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2])
        w2c = _as_homogeneous44(ext[i]).astype(np.float64).copy()
        w2c[:3, 3] *= scale  # 並進もメートル化
        vol.integrate(rgbd, intr, w2c)

    o3m = vol.extract_triangle_mesh()
    o3m.compute_vertex_normals()
    verts = np.asarray(o3m.vertices)
    faces = np.asarray(o3m.triangles)
    cols = (np.asarray(o3m.vertex_colors) * 255).astype(np.uint8) if len(o3m.vertex_colors) else None
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("TSDF: 面が生成されませんでした（voxel/scale を調整）")

    # 整列(先頭カメラ基準) + 中央寄せ + 地面水平化
    a = _alignment_transform(ext[0], verts)
    verts = trimesh.transform_points(verts, a).astype(np.float32)
    if level_ground:
        verts, _ = _level_ground(verts)
    verts -= np.median(verts, axis=0)

    rgba = None
    if cols is not None:
        rgba = np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], axis=1)
    m = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, vertex_colors=rgba, process=False)
    if max_faces and len(m.faces) > max_faces:
        m = _decimate_mesh(m, max_faces)
    scene = trimesh.Scene(); scene.add_geometry(m)
    info = {"representation": "mesh", "method": "tsdf", "vertex_count": int(len(m.vertices)),
            "face_count": int(len(m.faces)), "viewpoints": len(viewpoints), "voxel": voxel, "scale": float(scale)}
    return scene, info


def build_poisson_mesh(prediction, view_index, viewpoints, depth_octree: int = 9,
                       density_quantile: float = 0.08, max_faces: int = 1_200_000, progress=None):
    """クリーンな整合点群＋法線から Poisson 面再構成（滑らかな水密寄りの面）。"""
    import open3d as o3d
    from .reconstruct_da3 import _decimate_mesh, _noop

    progress = progress or _noop
    progress("mesh", 0, 1, "Poisson: 点群生成中 ...")
    # 主経路の点群（GPSアンカー・空除去・水平化済み）を素材にする
    cloud_scene, _ = build_multiview_pointcloud(
        prediction, view_index, viewpoints, max_width=400, conf_percentile=45.0,
        drop_sky=True, anchor_gps=True, mesh=False, far_clip_m=0.0, height_clip_m=40.0,
        level_ground=True, max_points=800000)
    g = list(cloud_scene.geometry.values())[0]
    pts = np.asarray(g.vertices, np.float64)
    cols = np.asarray(g.colors)[:, :3].astype(np.float64) / 255.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    progress("mesh", 0, 1, "Poisson: 法線推定中 ...")
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pc.orient_normals_towards_camera_location(camera_location=np.array([0.0, 50.0, 0.0]))

    progress("mesh", 0, 1, "Poisson: 面再構成中 ...")
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth_octree)
    density = np.asarray(density)
    keep = density > np.quantile(density, density_quantile)
    mesh.remove_vertices_by_mask(~keep)  # 低密度(膨らみ)を除去
    verts = np.asarray(mesh.vertices); faces = np.asarray(mesh.triangles)
    cols2 = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if len(mesh.vertex_colors) else None
    if len(faces) == 0:
        raise ValueError("Poisson: 面が生成されませんでした")
    rgba = None
    if cols2 is not None:
        rgba = np.concatenate([cols2, np.full((len(cols2), 1), 255, np.uint8)], axis=1)
    m = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces, vertex_colors=rgba, process=False)
    if max_faces and len(m.faces) > max_faces:
        m = _decimate_mesh(m, max_faces)
    scene = trimesh.Scene(); scene.add_geometry(m)
    info = {"representation": "mesh", "method": "poisson", "vertex_count": int(len(m.vertices)),
            "face_count": int(len(m.faces)), "viewpoints": len(viewpoints), "octree": depth_octree}
    return scene, info
