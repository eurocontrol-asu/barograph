from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from supertraj_met.domain import Cube, Domain, GridIndex

GLOBAL_LAT = np.arange(90.0, -90.01, -0.25)
GLOBAL_LON = np.arange(0.0, 360.0, 0.25)


def test_domain_takes_the_short_arc_across_the_antimeridian() -> None:
    d = Domain.from_points([10.0, 12.0], [179.0, -179.0], margin=0.5)
    assert (d.lon_min, d.lon_max) == (178.5, 181.5)
    assert (d.lat_min, d.lat_max) == (9.5, 12.5)


def test_domain_across_greenwich_uses_negative_longitudes() -> None:
    d = Domain.from_points([45.0], [359.0], margin=0.0)
    assert d.lon_min == d.lon_max == -1.0


def test_invalid_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="latitude"):
        Domain(10.0, 5.0, 0.0, 1.0)


def test_grid_index_wraps_columns_and_encloses_the_domain() -> None:
    g = GridIndex.locate(Domain(44.1, 45.9, -2.1, 3.1), GLOBAL_LAT, GLOBAL_LON)
    assert g.lat[0] >= 45.9 and g.lat[-1] <= 44.1
    assert g.lon[0] <= -2.1 and g.lon[-1] >= 3.1
    np.testing.assert_allclose(GLOBAL_LON[g.cols], np.mod(g.lon, 360.0))
    assert np.all(np.diff(g.lon) > 0)


def test_grid_index_keeps_two_points_for_a_single_location() -> None:
    g = GridIndex.locate(Domain(45.0, 45.0, 1.0, 1.0), GLOBAL_LAT, GLOBAL_LON)
    assert len(g.lat) >= 2 and len(g.lon) >= 2


def test_cube_round_trips_through_disk(make_cube: Callable[..., Cube], tmp_path: Path) -> None:
    cube = make_cube()
    cube.p_range = (20000.0, 30000.0)
    path = tmp_path / "c.npz"
    cube.save(path)
    back = Cube.load(path)
    assert back.hour == cube.hour and back.p_range == cube.p_range
    np.testing.assert_array_equal(back.fields["u"], cube.fields["u"])
    np.testing.assert_array_equal(back.vertical.ids, cube.vertical.ids)


def test_cube_serves_only_narrower_requests(make_cube: Callable[..., Cube]) -> None:
    cube = make_cube()
    cube.p_range, cube.level_margin = (20000.0, 30000.0), 1
    inner = Domain(41.0, 49.0, 355.5, 356.0)  # same place as -4.5..-4.0
    assert cube.serves(inner, (21000.0, 29000.0), 1)
    assert not cube.serves(inner, (19000.0, 29000.0), 1)
    assert not cube.serves(inner, (21000.0, 29000.0), 2)
    assert not cube.serves(Domain(41.0, 49.0, 9.0, 11.0), (21000.0, 29000.0), 1)


@pytest.mark.parametrize(("lon_min", "lon_max"), [(10.0, 5.0), (0.0, 400.0)])
def test_invalid_longitude_ranges_are_rejected(lon_min: float, lon_max: float) -> None:
    with pytest.raises(ValueError, match="longitude"):
        Domain(0.0, 1.0, lon_min, lon_max)


def test_points_spanning_the_globe_get_the_whole_longitude_range() -> None:
    d = Domain.from_points([0.0, 0.0, 0.0], [0.0, 120.0, 240.0], margin=100.0)
    assert (d.lon_min, d.lon_max) == (-180.0, 180.0)
    with pytest.raises(ValueError, match="no points"):
        Domain.from_points([], [])


def test_enclosing_domains() -> None:
    outer = Domain(40.0, 50.0, 350.0, 370.0)
    assert outer.encloses(Domain(41.0, 49.0, -5.0, 5.0))
    assert not outer.encloses(Domain(41.0, 49.0, 5.0, 15.0))
    assert Domain.whole_globe().encloses(Domain(-10.0, 10.0, 100.0, 200.0))
