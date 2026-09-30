from __future__ import annotations

import numpy as np
import pytest

from baro.met.isa import (
    altitude_from_pressure,
    pressure_from_altitude,
    pressure_from_altitude_ft,
)


def test_sea_level_and_tropopause_match_icao_values() -> None:
    p = pressure_from_altitude([0.0, 11000.0, 20000.0])
    np.testing.assert_allclose(p, [101325.0, 22632.06, 5474.89], rtol=2e-5)


def test_flight_level_360_is_about_227_hpa() -> None:
    assert pressure_from_altitude_ft(36000.0) == pytest.approx(22716.0, rel=1e-3)


def test_inverse_round_trips_across_layers() -> None:
    h = np.array([-300.0, 0.0, 5000.0, 10999.0, 11000.0, 15000.0, 25000.0])
    np.testing.assert_allclose(altitude_from_pressure(pressure_from_altitude(h)), h, atol=1e-6)


def test_above_model_range_is_nan() -> None:
    assert np.isnan(pressure_from_altitude(40000.0))
