"""道沿いの複数地点を 1 つの歩ける 3D 空間へ合成する。

フロントが地図上で選んだ点列（道沿い）を受け取り、
1. 各点を実在パノラマへスナップ（pano_id で重複除去）
2. 各点で 360° パノラマメッシュを生成
3. 最初の点を原点とした実距離オフセットで平行移動して連結
することで、ルート全体を歩いて移動できる 1 つの glb にする。

ネットワーク取得（スナップ・画像 DL）と推論（深度→メッシュ）を分離し、
呼び出し側が推論部分だけをロックできるようにしている。
"""

import trimesh

from .geo import world_translation
from .panorama import build_panorama
from .streetview import fetch_streetview_panorama, fetch_streetview_metadata

MAX_ROUTE_POINTS = 12


def snap_route_points(
    points: list[tuple[float, float]],
    api_key: str | None = None,
    radius: int = 50,
) -> list[dict]:
    """点列を実在パノラマへスナップし、pano_id で重複を除いた順序付きリストを返す。

    各要素は {"lat", "lng", "pano_id", "date"}。スナップ不能な点は飛ばす。
    """
    snapped: list[dict] = []
    seen_panos: set[str] = set()
    for lat, lng in points:
        meta = fetch_streetview_metadata(lat, lng, radius=radius, api_key=api_key)
        if meta is None:
            continue
        pano_id = meta.get("pano_id")
        if pano_id and pano_id in seen_panos:
            continue
        if pano_id:
            seen_panos.add(pano_id)
        snapped.append(meta)
    return snapped


def fetch_route_panoramas(
    snapped: list[dict],
    num_views: int,
    pitch: float,
    fov: float,
    api_key: str | None = None,
) -> list[list]:
    """スナップ済み各点について Street View を num_views 方位ぶん取得する。

    戻り値は [[(image, heading), ...], ...]（点ごとの視点リスト）。ネットワーク処理。
    """
    headings = [i * 360.0 / num_views for i in range(num_views)]
    panoramas = []
    for point in snapped:
        images = fetch_streetview_panorama(
            point["lat"], point["lng"], headings, pitch=pitch, fov=fov, api_key=api_key
        )
        panoramas.append(images)
    return panoramas


def build_route_scene(
    panoramas: list[list],
    snapped: list[dict],
    fov: float,
    max_width: int | None = None,
):
    """点ごとのパノラマを実距離オフセットで連結し (trimesh.Scene, info) を返す。

    panoramas[i] は snapped[i] に対応する [(image, heading), ...]。推論を含む処理。
    """
    if not panoramas:
        raise ValueError("ルートに有効な地点がありません")

    origin = (snapped[0]["lat"], snapped[0]["lng"])
    meshes = []
    total_vertices = 0
    total_faces = 0
    for images, point in zip(panoramas, snapped):
        mesh, info = build_panorama(images, fov_deg=fov, max_width=max_width)
        mesh.apply_translation(world_translation(origin, (point["lat"], point["lng"])))
        meshes.append(mesh)
        total_vertices += info["vertex_count"]
        total_faces += info["face_count"]

    scene = trimesh.util.concatenate(meshes)
    info = {
        "points": len(meshes),
        "vertex_count": total_vertices,
        "face_count": total_faces,
        "fov_deg": float(fov),
        "origin": {"lat": origin[0], "lng": origin[1]},
    }
    return scene, info
