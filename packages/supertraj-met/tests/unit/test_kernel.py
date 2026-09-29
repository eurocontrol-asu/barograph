from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from synthetic import surface_pressure, truth

from supertraj_met.domain import Cube
from supertraj_met.kernel import Flag, sample, track_components
from supertraj_met.vertical import pressure_levels


def _points(n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    lat = rng.uniform(40.0, 50.0, n)
    lon = rng.uniform(-5.0, 10.0, n)
    ps = surface_pressure(lat, lon)
    p = rng.uniform(16000.0, ps * 0.97)
    return lat, lon, p


def test_log_pressure_interpolation_recovers_the_analytic_field(
    make_cube: Callable[..., Cube],
) -> None:
    lat, lon, p = _points()
    values, flags = sample(make_cube(), lat, lon, p)
    assert np.all(flags == Flag.OK)
    for name in ("u", "v", "t"):
        np.testing.assert_allclose(
            values[name], truth(name, lat, lon, p, np.zeros_like(p)), atol=1e-6
        )


def test_standard_surface_pressure_biases_low_levels_only(make_cube: Callable[..., Cube]) -> None:
    cube = make_cube()
    lat = np.array([49.0, 49.0])
    lon = np.array([9.0, 9.0])  # ps ~ 96.8 kPa, far from standard
    p = np.array([25000.0, 85000.0])
    actual, _ = sample(cube, lat, lon, p)
    standard, _ = sample(cube, lat, lon, p, surface_pressure="standard")
    err = np.abs(standard["t"] - actual["t"])
    assert err[1] > 3 * err[0] > 0.1  # ~0.45 K at 250 hPa vs ~1.9 K at 850 hPa


def test_points_are_flagged_instead_of_extrapolated(make_cube: Callable[..., Cube]) -> None:
    cube = make_cube(level_ids=np.arange(70, 138))
    lat = np.array([45.0, 45.0, 45.0, 60.0, np.nan])
    lon = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    top = cube.vertical.pressure(surface_pressure(np.array(45.0), np.array(0.0)))[0]
    p = np.array([top * 0.9, 106000.0, 30000.0, 30000.0, 30000.0])
    values, flags = sample(cube, lat, lon, p)
    assert list(flags) == [
        Flag.ABOVE_TOP,
        Flag.BELOW_BOTTOM,
        Flag.OK,
        Flag.OUTSIDE_DOMAIN,
        Flag.INVALID_INPUT,
    ]
    assert np.isnan(values["u"][[0, 1, 3, 4]]).all() and np.isfinite(values["u"][2])


def test_missing_surface_pressure_is_flagged(make_cube: Callable[..., Cube]) -> None:
    cube = make_cube()
    assert cube.ps is not None
    cube.ps[:, :] = np.nan
    _, flags = sample(cube, [45.0], [0.0], [30000.0])
    assert flags[0] == Flag.MISSING_DATA


def test_longitudes_are_matched_modulo_360(make_cube: Callable[..., Cube]) -> None:
    cube = make_cube()
    a, _ = sample(cube, [45.0], [-2.3], [30000.0])
    b, _ = sample(cube, [45.0], [357.7], [30000.0])
    np.testing.assert_allclose(a["u"], b["u"])


def test_pressure_levels_cube_is_supported(make_cube: Callable[..., Cube]) -> None:
    ml = make_cube()
    layers = np.stack(
        [np.full((len(ml.lat), len(ml.lon)), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)]
    )
    pl = Cube(ml.source, ml.hour, ml.lat, ml.lon, pressure_levels([200, 250, 300]), {"x": layers})
    values, flags = sample(pl, [45.0], [1.0], [np.sqrt(20000.0 * 25000.0)], vertical="log")
    assert flags[0] == Flag.OK
    assert values["x"][0] == pytest.approx(1.5)


def test_track_components() -> None:
    along, cross = track_components([10.0, 0.0], [0.0, 10.0], [90.0, 90.0])
    np.testing.assert_allclose(along, [10.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cross, [0.0, -10.0], atol=1e-12)
