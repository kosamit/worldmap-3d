"""geo.py のローカル平面座標変換の単体テスト。"""

import math

from app.geo import (
    METERS_PER_DEG_LAT,
    local_offset_meters,
    world_translation,
)


def test_origin_maps_to_zero():
    # Arrange
    origin = (35.6595, 139.7005)

    # Act
    east, north = local_offset_meters(origin, origin)

    # Assert
    assert east == 0.0
    assert north == 0.0


def test_north_is_positive_for_higher_latitude():
    # Arrange
    origin = (35.0, 139.0)
    one_deg_north = (36.0, 139.0)

    # Act
    east, north = local_offset_meters(origin, one_deg_north)

    # Assert
    assert east == 0.0
    assert north == METERS_PER_DEG_LAT


def test_east_shrinks_with_latitude_cosine():
    # Arrange
    origin = (60.0, 10.0)
    one_deg_east = (60.0, 11.0)

    # Act
    east, _ = local_offset_meters(origin, one_deg_east)

    # Assert: 経度方向は cos(緯度) で縮む
    expected = 111_320.0 * math.cos(math.radians(60.0))
    assert math.isclose(east, expected, rel_tol=1e-9)


def test_world_translation_maps_north_to_negative_z():
    # Arrange
    origin = (35.0, 139.0)
    north_point = (35.01, 139.0)

    # Act
    x, y, z = world_translation(origin, north_point)

    # Assert: 北向きはワールドの -Z、高さは 0
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert y == 0.0
    assert z < 0.0
