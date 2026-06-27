"""複数の狭角 Street View タイルを 1 枚の equirectangular(全天球)画像へ再投影する。

6 面キューブ(各 fov90°/640px)では粗いため、より狭い fov のタイルを多数取得して
角度あたりの画素密度を上げ、滑らかで高精細なパノラマを作る。

座標規約（フロントの球体テクスチャと一致させる）:
- 経度 lon: 0 = 北、東向きに増加
- 緯度 lat: +で上
- ワールド方向: 北 = -Z, 東 = +X, 上 = +Y
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .streetview import fetch_streetview

# タイル取得は I/O 待ちなので並列化して総取得時間を短縮する。
_FETCH_WORKERS = 8

# タイルの取得方位・仰角（度）と fov。緯度帯 + 真上/真下で全天球を覆う。
SOURCE_FOV = 62.0
POLE_FOV = 90.0


def _source_views(hi: bool = False) -> list[tuple[float, float, float]]:
    views: list[tuple[float, float, float]] = []
    if hi:
        # 高精細(26枚): 通常と同じ 8方位×3仰角＋上下 だが、fovを少し狭めて(ズーム)
        # 実解像度を稼ぎ、out_width も上げて鮮明化する。45°間隔に対し fov55 は約10°重なり
        # で隙間なし。
        for heading in range(0, 360, 45):  # 8 方位
            for pitch in (-45.0, 0.0, 45.0):
                views.append((float(heading), pitch, 55.0))
        views.append((0.0, 90.0, 75.0))   # 真上
        views.append((0.0, -90.0, 75.0))  # 真下
        return views
    for heading in range(0, 360, 45):  # 8 方位
        for pitch in (-45.0, 0.0, 45.0):
            views.append((float(heading), pitch, SOURCE_FOV))
    views.append((0.0, 90.0, POLE_FOV))   # 真上
    views.append((0.0, -90.0, POLE_FOV))  # 真下
    return views


def _basis(heading_deg: float, pitch_deg: float):
    """視点の (forward, right, up) 単位ベクトルを返す。"""
    hr = np.radians(heading_deg)
    pr = np.radians(pitch_deg)
    cp = np.cos(pr)
    forward = np.array([cp * np.sin(hr), np.sin(pr), -cp * np.cos(hr)])
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(forward[1]) > 0.99:  # 真上/真下では world_up と平行になるので基準を変える
        world_up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)
    return forward, right, up


def build_equirectangular(
    lat: float,
    lng: float,
    api_key: str | None = None,
    pano: str | None = None,
    out_width: int = 2560,
    tile_size: int = 640,
    hi: bool = False,
) -> Image.Image:
    """複数タイルを取得して equirectangular 画像(RGB)を返す。

    hi=True で狭角タイルを多数集め、Google の実解像度で高精細化する（GPU不要）。
    """
    out_w = out_width
    out_h = out_width // 2
    views = _source_views(hi=hi)

    def _fetch(view: tuple[float, float, float]) -> np.ndarray:
        heading, pitch, fov = view
        image, _ = fetch_streetview(
            lat,
            lng,
            heading=heading,
            pitch=pitch,
            fov=fov,
            size=f"{tile_size}x{tile_size}",
            api_key=api_key,
            pano=pano,
        )
        return np.asarray(image, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        tiles = list(pool.map(_fetch, views))

    # 出力ピクセルごとのワールド方向
    xs = (np.arange(out_w) + 0.5) / out_w
    ys = (np.arange(out_h) + 0.5) / out_h
    lon = xs * 2.0 * np.pi - np.pi          # -pi..pi (0 = 北)
    lat_arr = np.pi / 2.0 - ys * np.pi      # +pi/2..-pi/2
    lon_g, lat_g = np.meshgrid(lon, lat_arr)
    ce = np.cos(lat_g)
    dx = ce * np.sin(lon_g)
    dy = np.sin(lat_g)
    dz = -ce * np.cos(lon_g)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    best = np.full((out_h, out_w), -1.0)  # cz が大きい(視点中心に近い)ほど優先
    half = tile_size / 2.0

    for (heading, pitch, fov), tile in zip(views, tiles):
        forward, right, up = _basis(heading, pitch)
        focal = half / np.tan(np.radians(fov) / 2.0)
        cx = dx * right[0] + dy * right[1] + dz * right[2]
        cy = dx * up[0] + dy * up[1] + dz * up[2]
        cz = dx * forward[0] + dy * forward[1] + dz * forward[2]
        safe = np.where(cz > 1e-3, cz, 1.0)
        u = half + focal * (cx / safe)
        v = half - focal * (cy / safe)
        inside = (cz > 1e-3) & (u >= 0) & (u < tile_size) & (v >= 0) & (v < tile_size)
        take = inside & (cz > best)
        ui = np.clip(u, 0, tile_size - 1).astype(np.int32)
        vi = np.clip(v, 0, tile_size - 1).astype(np.int32)
        sampled = tile[vi, ui]
        out[take] = sampled[take]
        best[take] = cz[take]

    return Image.fromarray(out, "RGB")
