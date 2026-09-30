from __future__ import annotations

import numpy as np
import pytest

from baro.met.domain import Domain, GridIndex

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


@pytest.mark.parametrize(("lon_min", "lon_max"), [(10.0, 5.0), (0.0, 400.0)])
def test_invalid_longitude_ranges_are_rejected(lon_min: float, lon_max: float) -> None:
    with pytest.raises(ValueError, match="longitude"):
        Domain(0.0, 1.0, lon_min, lon_max)


def test_points_spanning_the_globe_get_the_whole_longitude_range() -> None:
    d = Domain.from_points([0.0, 0.0, 0.0], [0.0, 120.0, 240.0], margin=100.0)
    assert (d.lon_min, d.lon_max) == (-180.0, 180.0)
    with pytest.raises(ValueError, match="no points"):
        Domain.from_points([], [])


# ARPEGE 0.1 deg Europe as the GRIB decoder returns it: 328 (-32) .. 402 (42) degrees east
ARPEGE_LAT = np.arange(72.0, 19.95, -0.1)
ARPEGE_LON = np.arange(328.0, 402.05, 0.1)


def test_regional_box_across_the_western_edge_is_clipped_not_sent_east() -> None:
    grid = GridIndex.locate(Domain(40, 50, -32.1, 10.0), ARPEGE_LAT, ARPEGE_LON)
    assert grid.lon[0] == pytest.approx(328.0) and grid.lon[-1] == pytest.approx(370.0)


def test_whole_globe_on_a_regional_grid_is_the_whole_grid() -> None:
    grid = GridIndex.locate(Domain.whole_globe(), ARPEGE_LAT, ARPEGE_LON)
    assert grid.lon[0] == pytest.approx(328.0) and grid.lon[-1] == pytest.approx(402.0)
    wide = GridIndex.locate(Domain(40, 50, -60.0, 60.0), ARPEGE_LAT, ARPEGE_LON)
    assert len(wide.lon) == len(ARPEGE_LON)


def test_regional_box_inside_the_grid_is_unchanged() -> None:
    grid = GridIndex.locate(Domain(40, 50, -5.0, 10.0), ARPEGE_LAT, ARPEGE_LON)
    assert grid.lon[0] == pytest.approx(355.0) and grid.lon[-1] == pytest.approx(370.0)
