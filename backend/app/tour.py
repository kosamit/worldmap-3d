"""道沿いの複数 Street View 地点を、写真パノラマで繋いだ歩けるツアーに合成する。

深度メッシュ化はせず、各地点のキューブ6面写真をそのまま使う（Street View の見た目）。
各ノードを最初の地点を原点とした実距離で配置し、フロントが順に行き来できるようにする。
"""

import json

from . import storage
from .geo import local_offset_meters
from .route import snap_route_points
from .streetview import fetch_cube_faces

MAX_TOUR_NODES = 20
FACE_FORMAT = "JPEG"
FACE_QUALITY = 85


def build_tour(
    points: list[tuple[float, float]],
    api_key: str | None = None,
    face_size: int = 640,
) -> dict:
    """点列を実パノラマにスナップし、各ノードのキューブ6面を保存してツアー meta を返す。"""
    snapped = snap_route_points(points, api_key=api_key)
    if not snapped:
        raise ValueError("選択した道沿いに Street View が見つかりませんでした")
    snapped = snapped[:MAX_TOUR_NODES]

    tour_id = storage.new_tour_id()
    out_dir = storage.tour_dir(tour_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    origin = (snapped[0]["lat"], snapped[0]["lng"])
    nodes = []
    for i, point in enumerate(snapped):
        faces = fetch_cube_faces(
            point["lat"], point["lng"], size=face_size, api_key=api_key
        )
        face_urls = {}
        for key, image in faces.items():
            fname = f"node{i}_{key}.jpg"
            image.save(str(out_dir / fname), format=FACE_FORMAT, quality=FACE_QUALITY)
            face_urls[key] = f"/tours/{tour_id}/{fname}"

        east, north = local_offset_meters(origin, (point["lat"], point["lng"]))
        nodes.append(
            {
                "index": i,
                "lat": point["lat"],
                "lng": point["lng"],
                "pano_id": point.get("pano_id"),
                "x": east,
                "z": -north,  # 北を -Z に合わせる
                "faces": face_urls,
            }
        )

    meta = {
        "id": tour_id,
        "origin": {"lat": origin[0], "lng": origin[1]},
        "node_count": len(nodes),
        "nodes": nodes,
    }
    (out_dir / "tour.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
