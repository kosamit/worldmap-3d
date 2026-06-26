"""緯度経度をローカルな平面メートル座標へ変換する小さなヘルパー。

ルート上の各地点を「最初の地点を原点とした相対位置」に置くために使う。
数 km 程度の範囲なら等距円筒近似で十分な精度。

座標規約:
- east  : 東向きを + (m)  → ワールドの +X
- north : 北向きを + (m)  → ワールドの -Z（Three.js/glTF は -Z が前方/北）
"""

import math

# 緯度 1 度あたりのメートル（地球平均）。経度方向は cos(緯度) で縮む。
METERS_PER_DEG_LAT = 110_540.0
METERS_PER_DEG_LNG = 111_320.0


def local_offset_meters(
    origin: tuple[float, float], point: tuple[float, float]
) -> tuple[float, float]:
    """origin を基準とした point の (east_m, north_m) を返す。

    origin, point はいずれも (lat, lng)。
    """
    lat0, lng0 = origin
    lat, lng = point
    north = (lat - lat0) * METERS_PER_DEG_LAT
    east = (lng - lng0) * METERS_PER_DEG_LNG * math.cos(math.radians(lat0))
    return east, north


def world_translation(
    origin: tuple[float, float], point: tuple[float, float]
) -> list[float]:
    """origin を基準にした point のワールド平行移動量 [x, y, z] を返す。

    east → +X、north → -Z（前方が -Z）。高さ(Y)は 0。
    """
    east, north = local_offset_meters(origin, point)
    return [east, 0.0, -north]
