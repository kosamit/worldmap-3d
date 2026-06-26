"""Google Street View Static API から画像を取得する。

API キーは環境変数 GOOGLE_MAPS_API_KEY、または呼び出し時の引数で渡す。
利用は Google Maps Platform の利用規約に従うこと。
"""

import io
import math
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


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """2 点間の概算距離 (m)。視点間ベースラインの実測に使う。"""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _offset_latlng(lat: float, lng: float, north_m: float, east_m: float):
    dlat = north_m / 111320.0
    dlng = east_m / (111320.0 * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng


def _is_user_photosphere(meta: dict) -> bool:
    """ユーザー投稿のフォトスフィア（屋内外バラバラで整合しない）を判定。

    公式 Street View は copyright が "© Google"、pano_id は短い。ユーザー投稿は
    copyright が個人名で pano_id が "CAoSF..." 形式。これらは混ぜると壊れる。
    """
    cr = (meta.get("copyright") or "")
    pid = meta.get("pano_id") or ""
    return ("Google" not in cr) or pid.startswith("CAoSF")


def gather_nearby_viewpoints(
    lat: float,
    lng: float,
    radius_m: float = 12.0,
    max_views: int = 5,
    api_key: str | None = None,
    consistent: bool = True,
) -> list[dict]:
    """中心とその周囲オフセットを実在パノラマにスナップし、重複除去して返す。

    マルチビュー再構成に必要な「ベースライン（視差）」を作るため、中心の前後左右を
    少しずらしてスナップし、別地点のパノラマを集める。返り値は中心を先頭にした
    [{"pano_id", "lat", "lng", "date", "copyright"}] のリスト（最大 max_views 件）。

    consistent=True（既定）: 中心と「同一ソース（copyright）かつ同一撮影日（date）」の
    公式パノラマだけを集める。屋内駅にユーザー投稿フォトスフィアや別日の屋外パノラマが
    混ざって「壁・天井が消える／ぐちゃぐちゃ」になるのを防ぐ。条件を満たすものが無ければ
    中心1地点だけ（=きれいな単一視点）になる。
    """
    # 中心＋全周8方向×2リング。柱の裏など遮蔽部を別位置から捉えるため位置を多めに探す
    # （同一撮影の実在パノラマだけが consistent フィルタで残る）。
    seeds = [(0.0, 0.0)]
    for rr in (radius_m, radius_m * 2.0):
        for deg in range(0, 360, 45):
            a = math.radians(deg)
            seeds.append((rr * math.cos(a), rr * math.sin(a)))
    out: list[dict] = []
    seen: set[str] = set()
    ref_copyright: str | None = None
    ref_date: str | None = None
    for north_m, east_m in seeds:
        plat, plng = _offset_latlng(lat, lng, north_m, east_m)
        try:
            meta = fetch_streetview_metadata(plat, plng, api_key=api_key)
        except ValueError:
            continue
        if not meta or not meta.get("pano_id"):
            continue
        pid = meta["pano_id"]
        if pid in seen:
            continue
        is_center = not out
        if consistent:
            if is_center:
                ref_copyright = meta.get("copyright")
                ref_date = meta.get("date")
            else:
                # ユーザー投稿や、中心と別ソース/別撮影日のパノラマは混ぜない。
                if _is_user_photosphere(meta):
                    continue
                if meta.get("copyright") != ref_copyright or meta.get("date") != ref_date:
                    continue
        seen.add(pid)
        out.append({
            "pano_id": pid, "lat": meta["lat"], "lng": meta["lng"],
            "date": meta.get("date"), "copyright": meta.get("copyright"),
        })
        if len(out) >= max_views:
            break
    return out


def sample_viewpoint_images(
    viewpoints: list[dict],
    headings: list[float],
    pitch: float = 0.0,
    fov: float = 90.0,
    size: str = DEFAULT_SIZE,
    api_key: str | None = None,
) -> tuple[list[Image.Image], list[int]]:
    """各視点パノラマから heading ぶん透視画像を取得する。

    (images, view_index) を返す。view_index[i] は images[i] がどの視点(viewpoints の
    インデックス)由来かを示す。隣接 heading が重なるよう fov を広めに使うこと。
    """
    images: list[Image.Image] = []
    view_index: list[int] = []
    for vi, vp in enumerate(viewpoints):
        for h in headings:
            image, _ = fetch_streetview(
                vp["lat"], vp["lng"], heading=h, pitch=pitch, fov=fov,
                size=size, api_key=api_key, pano=vp.get("pano_id"),
            )
            images.append(image)
            view_index.append(vi)
    return images, view_index


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
        "copyright": data.get("copyright"),
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
