"""Google Street View Static API から画像を取得する。

API キーは環境変数 GOOGLE_MAPS_API_KEY、または呼び出し時の引数で渡す。
利用は Google Maps Platform の利用規約に従うこと。
"""

import io
import os

import requests
from PIL import Image

STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
DEFAULT_SIZE = "640x640"


def fetch_streetview(
    lat: float,
    lng: float,
    heading: float = 0.0,
    pitch: float = 0.0,
    fov: float = 90.0,
    size: str = DEFAULT_SIZE,
    api_key: str | None = None,
) -> tuple[Image.Image, float]:
    """(PIL.Image RGB, 使用した fov) を返す。失敗時は ValueError。"""
    api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError(
            "Google Maps API キーが必要です (環境変数 GOOGLE_MAPS_API_KEY か api_key 引数)"
        )

    params = {
        "size": size,
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
        "return_error_code": "true",
    }
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
