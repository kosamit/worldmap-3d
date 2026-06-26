"""指定地点で視点収集＋DA3推論を1回行い、prediction を npz にキャッシュする。

各バリアントは同じ推論を共有して後処理だけ変えるので、推論(遅い)は1回で済む。
使い方: cache_pred.py <lat> <lng> <out_npz> [max_views] [heading_count] [pitch_count]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app import depth_da3  # noqa: E402
from app.main import _pitch_rows  # noqa: E402
from app.streetview import fetch_streetview, gather_nearby_viewpoints  # noqa: E402


def main():
    lat, lng, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    max_views = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    hc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    pc = int(sys.argv[6]) if len(sys.argv) > 6 else 3

    vps = gather_nearby_viewpoints(lat, lng, radius_m=12.0, max_views=max_views)
    headings = [i * 360.0 / hc for i in range(hc)]
    pitches = _pitch_rows(pc)
    images, vi = [], []
    for k, vp in enumerate(vps):
        for h in headings:
            for p in pitches:
                img, _ = fetch_streetview(vp["lat"], vp["lng"], heading=h, pitch=p,
                                          fov=90, pano=vp.get("pano_id"))
                images.append(img); vi.append(k)
    pred = depth_da3.infer_multiview(images, use_ray_pose=True)
    sky = pred.get("sky")
    np.savez_compressed(
        out, depth=pred["depth"], conf=pred["conf"],
        sky=(np.zeros(0) if sky is None else sky),
        is_metric=pred["is_metric"], intrinsics=pred["intrinsics"],
        extrinsics=pred["extrinsics"], processed_images=pred["processed_images"],
        view_index=np.array(vi),
        vp_lat=np.array([v["lat"] for v in vps]), vp_lng=np.array([v["lng"] for v in vps]))
    print(f"cached {out}: images={len(images)} viewpoints={len(vps)}")


if __name__ == "__main__":
    main()
