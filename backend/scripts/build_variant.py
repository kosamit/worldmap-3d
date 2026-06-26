"""1バリアントを1プロセスで生成して GLB を保存する（メモリ隔離のため変種ごとに別プロセス）。

使い方: build_variant.py <pred_npz> <variant> <scene_id> [<label>]
GLB は backend/data/scenes/<scene_id>/scene.glb に保存（/scenes で配信される）。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from app import storage  # noqa: E402
from app.reconstruct_da3 import build_multiview_pointcloud  # noqa: E402
import app.reconstruct_experiments as ex  # noqa: E402

COMMON = dict(far_clip_m=0.0, height_clip_m=40.0)


def baseline(p, vi, vps):
    return build_multiview_pointcloud(p, vi, vps, mesh=True, **COMMON)


def aggressive(p, vi, vps):
    return build_multiview_pointcloud(p, vi, vps, mesh=True, edge_factor=4.0,
                                      discontinuity_ratio=0.05, **COMMON)


def depth_smooth(p, vi, vps):
    return build_multiview_pointcloud(ex.smooth_depth_prediction(p), vi, vps, mesh=True, **COMMON)


def taubin(p, vi, vps):
    sc, info = build_multiview_pointcloud(p, vi, vps, mesh=True, **COMMON)
    info = dict(info, method="taubin")
    return ex.taubin_smooth_scene(sc, iterations=10), info


def cloud_clean(p, vi, vps):
    sc, info = build_multiview_pointcloud(p, vi, vps, mesh=False, **COMMON)
    info = dict(info, method="pointcloud_clean")
    return ex.cloud_remove_outliers(sc), info


def tsdf(p, vi, vps):
    return ex.build_tsdf_mesh(p, vi, vps, voxel=0.12, depth_trunc=40)


def poisson(p, vi, vps):
    return ex.build_poisson_mesh(p, vi, vps, depth_octree=9)


VARIANTS = {
    "baseline": baseline, "aggressive": aggressive, "depth_smooth": depth_smooth,
    "taubin": taubin, "cloud_clean": cloud_clean, "tsdf": tsdf, "poisson": poisson,
}


def main():
    pred_npz, variant, scene_id = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else variant
    z = np.load(pred_npz)
    sky = z["sky"]
    pred = dict(depth=z["depth"], conf=z["conf"], sky=(None if sky.size == 0 else sky),
                is_metric=int(z["is_metric"]), intrinsics=z["intrinsics"],
                extrinsics=z["extrinsics"], processed_images=z["processed_images"])
    vi = list(z["view_index"])
    vps = [{"lat": float(a), "lng": float(b)} for a, b in zip(z["vp_lat"], z["vp_lng"])]

    t0 = time.time()
    scene, info = VARIANTS[variant](pred, vi, vps)
    dt = time.time() - t0

    out_dir = storage.DATA_DIR / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_dir / "scene.glb"))
    meta = {"id": scene_id, "variant": variant, "label": label,
            "build_seconds": round(dt, 1), "glb_url": f"/scenes/{scene_id}/scene.glb", **info}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print(json.dumps({"ok": True, "variant": variant, "scene_id": scene_id,
                      "seconds": round(dt, 1),
                      "faces": info.get("face_count"), "verts": info.get("vertex_count")}))


if __name__ == "__main__":
    main()
