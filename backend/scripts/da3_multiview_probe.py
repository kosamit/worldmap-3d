"""DA3 マルチビュー整合の実機検証スクリプト（本実装前の確認用）。

中心地点の周囲を少しずらして実在パノラマを集め（=視差を作る）、各ビューから
透視画像を数枚サンプルし、DA3 inference に一括投入。推定ポーズ(extrinsics)に
ベースライン（並進）が出るか、glb 点群が妥当かを確認する。
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.streetview import fetch_streetview, fetch_streetview_metadata  # noqa: E402


def offset_latlng(lat, lng, north_m, east_m):
    dlat = north_m / 111320.0
    dlng = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng


def gather_viewpoints(lat, lng, radius_m=10.0):
    """中心 + 周囲4方向のオフセットを実パノラマにスナップし、重複除去して返す。"""
    seeds = [(0.0, 0.0)]
    for n, e in [(radius_m, 0), (-radius_m, 0), (0, radius_m), (0, -radius_m)]:
        seeds.append((n, e))
    seen = {}
    for n, e in seeds:
        plat, plng = offset_latlng(lat, lng, n, e)
        try:
            meta = fetch_streetview_metadata(plat, plng)
        except Exception as exc:  # noqa: BLE001
            print("  metadata err:", exc)
            continue
        if not meta or not meta.get("pano_id"):
            continue
        pid = meta["pano_id"]
        if pid not in seen:
            seen[pid] = (meta["lat"], meta["lng"])
    return seen  # pano_id -> (lat, lng)


def main():
    lat, lng = 35.6595, 139.7005  # 渋谷
    print(f"center=({lat},{lng})")
    vps = gather_viewpoints(lat, lng, radius_m=12.0)
    print(f"distinct viewpoints (panos): {len(vps)}")
    for pid, (a, b) in vps.items():
        print(f"  {pid[:16]}... ({a:.6f},{b:.6f})")

    headings = [40.0, 160.0]  # 前方アーク寄りで重なりを確保
    images = []
    for pid in list(vps.keys())[:4]:  # 最大4視点
        for h in headings:
            img, _ = fetch_streetview(0, 0, heading=h, pitch=0, fov=90, pano=pid)
            images.append(np.asarray(img))
    print(f"total images fed to DA3: {len(images)}")

    from depth_anything_3.api import DepthAnything3

    model_id = os.environ.get("DA3_MODEL", "depth-anything/DA3METRIC-LARGE")
    print("loading", model_id, "...")
    model = DepthAnything3.from_pretrained(model_id).to("cuda")

    out_dir = tempfile.mkdtemp(prefix="da3probe_")
    print("inference (no export first) ...")
    pred = model.inference(images)

    def shp(x):
        return None if x is None else getattr(x, "shape", type(x).__name__)

    print("FIELDS:",
          "depth=", shp(pred.depth),
          "conf=", shp(pred.conf),
          "extrinsics=", shp(pred.extrinsics),
          "intrinsics=", shp(pred.intrinsics),
          "processed_images=", shp(pred.processed_images))
    print("aux keys:", None if pred.aux is None else list(pred.aux.keys()))

    ext = pred.extrinsics  # (N,4,4) w2c
    print("is_metric:", getattr(pred, "is_metric", None),
          "scale_factor:", getattr(pred, "scale_factor", None))
    print("depth shape:", None if pred.depth is None else pred.depth.shape)
    if ext is not None:
        cams = []
        for e in ext:
            Rt = np.asarray(e)
            R = Rt[:3, :3]
            t = Rt[:3, 3]
            cam_center = -R.T @ t  # カメラ世界座標
            cams.append(cam_center)
        cams = np.array(cams)
        spread = cams.max(axis=0) - cams.min(axis=0)
        print("camera centers (world):")
        for i, c in enumerate(cams):
            print(f"  cam{i}: {np.round(c, 3)}")
        print("baseline spread (m if metric):", np.round(spread, 3),
              " max pairwise:", round(float(np.linalg.norm(spread)), 3))

    # intrinsics があれば glb エクスポートも試す
    glb = os.path.join(out_dir, "scene.glb")
    if pred.intrinsics is not None and pred.extrinsics is not None:
        print("exporting glb ->", out_dir)
        model.inference(images, export_dir=out_dir, export_format="glb", show_cameras=False)
    else:
        print("intrinsics/extrinsics 不足のため glb export スキップ")

    if os.path.exists(glb):
        import trimesh

        scene = trimesh.load(glb)
        npts = 0
        if hasattr(scene, "geometry"):
            for name, g in scene.geometry.items():
                v = getattr(g, "vertices", None)
                n = 0 if v is None else len(v)
                npts += n
                print(f"  glb geom {name}: {type(g).__name__} verts={n}")
        print(f"glb total verts: {npts}  size={os.path.getsize(glb)} bytes")
    else:
        print("NO glb produced")


if __name__ == "__main__":
    main()
