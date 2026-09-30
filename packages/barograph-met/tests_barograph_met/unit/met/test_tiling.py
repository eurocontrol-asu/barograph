"""Splitting the points of one time slice along a source's tiling."""

from __future__ import annotations

import numpy as np
import pytest

from baro.met.tiling import Tiling, partition

BANDS = Tiling(lat0=-90.0, lat_step=10.0)  # ERA5-like: bands spanning every longitude
SQUARES = Tiling(lat0=-90.0, lat_step=5.0, lon0=-180.0, lon_step=5.0)


def _labels(
    lat: list[float], lon: list[float], tiling: Tiling | None, margin: float = 0.5
) -> list[int]:
    return [int(x) for x in partition(np.array(lat), np.array(lon), margin, tiling)]


def test_distant_bands_are_separate_components_ordered_south_to_north() -> None:
    assert _labels([52.0, -33.0, 51.0, 12.0], [0.0, 150.0, 90.0, 0.0], BANDS) == [2, 0, 2, 1]


def test_points_near_a_band_edge_join_both_bands() -> None:
    # 40.3 N reaches the 30-40 band within a 0.5 margin, which bridges it to 25 N
    assert _labels([25.0, 40.3], [0.0] * 2, BANDS) == [0, 0]
    assert _labels([25.0, 40.3], [0.0] * 2, BANDS, margin=0.1) == [0, 1]
    assert _labels([25.0, 35.0], [0.0] * 2, BANDS, margin=0.1) == [0, 0]  # adjacent bands


def test_square_tiles_split_in_longitude_too() -> None:
    assert _labels([45.0, 45.0, 45.5], [2.0, 60.0, 3.0], SQUARES) == [0, 1, 0]


def test_tiles_on_both_sides_of_the_dateline_are_neighbours() -> None:
    assert _labels([10.0, 10.0], [179.9, -179.9], SQUARES) == [0, 0]
    assert _labels([10.0, 10.0], [179.9, -179.9], SQUARES, margin=0.0) == [0, 0]


def test_no_tiling_keeps_one_component() -> None:
    assert _labels([10.0, -60.0], [0.0, 100.0], None) == [0, 0]
    assert partition(np.array([]), np.array([]), 0.5, BANDS).size == 0


def test_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        Tiling(lat0=0.0, lat_step=0.0)
    assert SQUARES.n_cols == 72 and BANDS.n_cols == 1
