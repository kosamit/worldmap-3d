"""DA3 公式フロー（点群 glb）でシーンを生成して、自前メッシュと比較するための出力。

model.inference(images, export_dir=..., export_format="glb") をそのまま使い、
著者が想定する点群表現を出力する。出力先は backend/data/scenes/<id>/scene.glb 形式に
そろえて静的配信できるようにする。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np  # noqa: E402

from app import depth_da3  # noqa: E402
from app import storage  # noqa: E402
from app.streetview import gather_nearby_viewpoints, sample_viewpoint_images  # noqa: E402


def main():
    lat, lng = 35.6595, 139.7005
    vps = gather_nearby_viewpoints(lat, lng, radius_m=12.0, max_views=4)
    headings = [i * 360.0 / 6 for i in range(6)]  # 60°刻み（重なり確保）
    images, view_index = sample_viewpoint_images(vps, headings, fov=90)
    print(f"viewpoints={len(vps)} images={len(images)}")

    model = depth_da3._load(depth_da3.DA3_MULTIVIEW_MODEL_ID)
    arrays = [np.asarray(im) for im in images]

    sid = "official_ptcloud"
    out_dir = storage.DATA_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    print("inference + export_to_glb (official point cloud) ...")
    model.inference(
        arrays,
        export_dir=str(out_dir),
        export_format="glb",
        show_cameras=False,
        conf_thresh_percentile=50.0,
        num_max_points=800_000,
    )
    glb = out_dir / "scene.glb"
    print("exported:", glb, os.path.getsize(glb), "bytes")


if __name__ == "__main__":
    main()
