"""reconstruct_da3.build_multiview_mesh の実機検証。

視点収集 → 透視画像サンプル → DA3 マルチビュー推論 → 整合メッシュ → glb 出力。
頂点/面数・メートルスケール・バウンディングボックスを確認する。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np  # noqa: E402

from app import depth_da3  # noqa: E402
from app.reconstruct_da3 import build_multiview_mesh  # noqa: E402
from app.streetview import gather_nearby_viewpoints, sample_viewpoint_images  # noqa: E402


def main():
    lat, lng = 35.6595, 139.7005  # 渋谷
    vps = gather_nearby_viewpoints(lat, lng, radius_m=12.0, max_views=4)
    print(f"viewpoints: {len(vps)}")
    for vp in vps:
        print("  ", vp["pano_id"][:14], round(vp["lat"], 6), round(vp["lng"], 6))

    headings = [0.0, 90.0, 180.0, 270.0]
    images, view_index = sample_viewpoint_images(vps, headings, fov=90)
    print(f"images: {len(images)}  view_index: {view_index}")

    print("DA3 multiview inference ...")
    pred = depth_da3.infer_multiview(images)
    print("  depth", pred["depth"].shape, "conf",
          None if pred["conf"] is None else pred["conf"].shape,
          "ext", pred["extrinsics"].shape, "K", pred["intrinsics"].shape)

    print("build mesh ...")
    mesh, info = build_multiview_mesh(pred, view_index, vps, max_width=200)
    print("INFO:", info)
    if len(mesh.vertices):
        lo = mesh.vertices.min(axis=0)
        hi = mesh.vertices.max(axis=0)
        print("bbox min:", np.round(lo, 2), "max:", np.round(hi, 2),
              "extent(m):", np.round(hi - lo, 2))

    out = tempfile.mkdtemp(prefix="mvmesh_")
    path = os.path.join(out, "scene.glb")
    mesh.export(path)
    print(f"exported {path}  size={os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
