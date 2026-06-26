"""Google Street View Static API から画像を取得する。

API キーは環境変数 GOOGLE_MAPS_API_KEY、または呼び出し時の引数で渡す。
利用は Google Maps Platform の利用規約に従うこと。
"""

import io
import os

import requests
from PIL import Image

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
DEFAULT_SIZE = "640x640"
DEFAULT_SNAP_RADIUS_M = 50


def fetch_streetview(
    lat: float,
    lng: float,
    heading: float = 0.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    size: str = DEFAULT_SIZE,
    api_key: str | None = None,
    pano: str | None = None,
) -> tuple[Image.Image, float]:
    """(PIL.Image RGB, 使用した fov) を返す。失敗時は ValueError。

    pano を指定すると座標ではなくパノラマ ID で取得する（隣接ノードを正確に取得）。
    """
    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "Google Maps API キーが必要です (環境変数 GOOGLE_MAPS_API_KEY か api_key 引数)"
        )

    params = {
        "size": size,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
        "return_error_code": "true",
    }
    if pano:
        params["pano"] = pano
    else:
        params["location"] = f"{lat},{lng}"
    resp = requests.get(STREETVIEW_URL, params=params, timeout=20)
    if resp.status_code == 404:
        raise ValueError(
            "指定地点に Street View 画像が見つかりません（座標を変えるか heading を調整してください）"
        )
    if resp.status_code == 403:
        raise ValueError(
            "Street View API に拒否されました（API キーの制限・有効化状態を確認してください）"
        )
    if resp.status_code != 200:
        raise ValueError(f"Street View API エラー {resp.status_code}")

    try:
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - 外部レスポンスの破損を明示的に握る
        raise ValueError(f"Street View 画像のデコードに失敗: {exc}") from exc

    return image, fov


def fetch_streetview_panorama(
    lat: float,
    lng: float,
    headings: list[float],
    pitch: float = 0.0,
    fov: float = 90.0,
    size: str = DEFAULT_SIZE,
    api_key: str | None = None,
) -> list[tuple[Image.Image, float]]:
    """指定方位リストぶん Street View を取得して [(image, heading), ...] を返す。"""
    results = []
    for heading in headings:
        image, _ = fetch_streetview(
            lat, lng, heading=heading, pitch=pitch, fov=fov, size=size, api_key=api_key
        )
        results.append((image, heading))
    return results


def fetch_streetview_metadata(
    lat: float,
    lng: float,
    radius: int = DEFAULT_SNAP_RADIUS_M,
    api_key: str | None = None,
) -> dict | None:
    """指定座標の近傍にあるパノラマへスナップする。

    実在するパノラマが見つかれば {"lat", "lng", "pano_id", "date"} を返す。
    無ければ None。メタデータ API は課金されない（実画像取得前のスナップに最適）。
    失敗時は ValueError。
    """
    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "Google Maps API キーが必要です (環境変数 GOOGLE_MAPS_API_KEY か api_key 引数)"
        )

    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": api_key,
    }
    resp = requests.get(METADATA_URL, params=params, timeout=20)
    if resp.status_code != 200:
        raise ValueError(f"Street View metadata API エラー {resp.status_code}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError(f"Street View metadata の JSON 解析に失敗: {exc}") from exc

    status = data.get("status")
    if status == "ZERO_RESULTS" or status == "NOT_FOUND":
        return None
    if status != "OK":
        raise ValueError(f"Street View metadata ステータス異常: {status}")

    location = data.get("location") or {}
    return {
        "lat": float(location["lat"]),
        "lng": float(location["lng"]),
        "pano_id": data.get("pano_id"),
        "date": data.get("date"),
    }


# キューブ6面の定義。キーは Three.js の CubeTexture 並び [px, nx, py, ny, pz, nz]。
# heading は北基準（0=北, 90=東）で、全ノードで軸を揃えるため絶対値を使う。
CUBE_FACES: list[tuple[str, float, float]] = [
    ("px", 90.0, 0.0),    # +X 東
    ("nx", 270.0, 0.0),   # -X 西
    ("py", 0.0, 90.0),    # +Y 上
    ("ny", 0.0, -90.0),   # -Y 下
    ("pz", 180.0, 0.0),   # +Z 南
    ("nz", 0.0, 0.0),     # -Z 北
]


def fetch_cube_faces(
    lat: float,
    lng: float,
    size: int = 640,
    api_key: str | None = None,
    pano: str | None = None,
) -> dict[str, Image.Image]:
    """1 地点のスカイボックス用キューブ6面を取得する。

    pano を指定すると座標ではなくパノラマ ID で取得する。
    {"px","nx","py","ny","pz","nz": PIL.Image} を返す。各面は fov=90 の正方形。
    深度推定は行わず写真をそのまま使うため、Street View の見た目を保てる。
    """
    size_str = f"{size}x{size}"
    faces: dict[str, Image.Image] = {}
    for key, heading, pitch in CUBE_FACES:
        image, _ = fetch_streetview(
            lat,
            lng,
            heading=heading,
            pitch=pitch,
            fov=90.0,
            size=size_str,
            api_key=api_key,
            pano=pano,
        )
        faces[key] = image
    return faces
