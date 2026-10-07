"""Physical surface conversions and input selections without archive I/O."""

from __future__ import annotations

import numpy as np
import pytest

from baro.met.openmeteo_surface import (
    _field_mapping,
    _native_variables,
    _output_values,
    _validate_storage,
)


def test_rain_preserves_native_period_total_and_temperature_converts_to_kelvin() -> None:
    native = {
        "precipitation": np.array([[4.0, 1.0], [0.0, np.nan]], dtype=np.float32),
        "snowfall_water_equivalent": np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32),
        "temperature_2m": np.array([[-10.0, 0.0], [10.0, 20.0]], dtype=np.float32),
    }
    before = {name: value.copy() for name, value in native.items()}
    result = _output_values(
        native, {"liquid": "rain", "total": "precipitation", "temperature": "t2m"}
    )
    np.testing.assert_array_equal(result["liquid"], [[2.0, 0.0], [0.0, np.nan]])
    np.testing.assert_array_equal(result["total"], before["precipitation"])
    np.testing.assert_allclose(result["temperature"], [[263.15, 273.15], [283.15, 293.15]])
    for name in native:
        np.testing.assert_array_equal(native[name], before[name])


def test_shared_precipitation_dependency_is_read_once_and_wind_is_not_requested() -> None:
    assert _native_variables({"liquid": "rain", "total": "precipitation"}) == [
        "precipitation",
        "snowfall_water_equivalent",
    ]
    assert _native_variables({"eastward": "u10"}) == ["wind_u_component_10m"]


def test_selection_is_validated_and_copied_before_data_access() -> None:
    fields = {"temperature": "t2m"}
    result = _field_mapping(fields)
    result["temperature"] = "u10"
    assert fields == {"temperature": "t2m"}
    assert set(_field_mapping(None)) == {"u10", "v10", "t2m", "precipitation", "rain", "gusts"}
    with pytest.raises(ValueError, match="at least one"):
        _field_mapping({})
    with pytest.raises(ValueError, match="unknown surface variables"):
        _field_mapping({"bad": "temperature_at_unknown_height"})


@pytest.mark.parametrize("tile,cache", [(0, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_spatial_or_cache_budget_is_rejected(tile: float, cache: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        _validate_storage(tile, cache)
