"""Actual ecCodes messages exercise response semantics and interval differencing."""

import eccodes as ec
import numpy as np
import pytest

from baro.met.grib import GribFormatError
from baro.met.immediate_grib import ImmediateField, decode_surface_response, normalize_precipitation

RUN = np.datetime64("2026-10-07T12:00", "ns")


def message(field: str = "u10", start: int = 0, end: int = 15, **overrides: int) -> bytes:
    handle = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    param = {
        "u10": (2, 2, 103, 10),
        "gusts": (2, 22, 103, 10),
        "precipitation": (1, 52, 1, 0),
        "rain": (1, 65, 1, 0),
    }[field]
    keys = {
        "centre": 85,
        "Ni": 2,
        "Nj": 2,
        "latitudeOfFirstGridPointInDegrees": 49,
        "latitudeOfLastGridPointInDegrees": 48,
        "longitudeOfFirstGridPointInDegrees": 2,
        "longitudeOfLastGridPointInDegrees": 3,
        "iDirectionIncrementInDegrees": 1,
        "jDirectionIncrementInDegrees": 1,
        "dataDate": 20261007,
        "dataTime": 1200,
        "parameterCategory": param[0],
        "parameterNumber": param[1],
        "typeOfFirstFixedSurface": param[2],
        "scaledValueOfFirstFixedSurface": param[3],
        "scaleFactorOfFirstFixedSurface": 0,
        "stepType": "instant" if field == "u10" else "max" if field == "gusts" else "accum",
        "stepUnits": "m",
        "startStep": start if field != "u10" else end,
        "endStep": end,
    }
    try:
        for key, value in keys.items():
            ec.codes_set(handle, key, value)
        for key, value in overrides.items():
            ec.codes_set(handle, key, value)
        ec.codes_set_values(handle, [1, 2, 3, 4])
        return bytes(ec.codes_get_message(handle))
    finally:
        ec.codes_release(handle)


def decode(
    buf: bytes, field: str = "u10", minutes: int = 15, interval: int = 900
) -> ImmediateField:
    return decode_surface_response(
        buf,
        field=field,
        run=RUN,
        valid_time=RUN + np.timedelta64(minutes, "m"),
        interval_seconds=interval,
    )


def test_instant_orientation_and_native_intervals() -> None:
    result = decode(message())
    assert result.interval_start is None
    np.testing.assert_array_equal(result.lat, [48, 49])
    np.testing.assert_array_equal(result.values, [[3, 4], [1, 2]])
    for field in ["gusts", "precipitation", "rain"]:
        result = decode(message(field), field)
        assert result.interval_start == RUN
    five = decode(message("precipitation", end=5), "precipitation", minutes=5, interval=300)
    assert five.interval_start is not None
    assert five.interval_end - five.interval_start == np.timedelta64(5, "m")


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataTime": 1100},
        {"scaledValueOfFirstFixedSurface": 20},
        {"parameterNumber": 3},
        {"productDefinitionTemplateNumber": 5},
    ],
)
def test_reject_wrong_response(overrides: dict[str, int]) -> None:
    with pytest.raises(GribFormatError):
        decode(message("u10", 0, 15, **overrides))


def test_reject_validity_and_statistical_mismatch() -> None:
    with pytest.raises(GribFormatError, match="validity"):
        decode(message(end=30))
    with pytest.raises(GribFormatError, match="interval"):
        decode(message("gusts", end=30), "gusts", minutes=30)
    with pytest.raises(GribFormatError, match="exactly one"):
        decode(message() + message())
    with pytest.raises(GribFormatError):
        decode(message("precipitation", typeOfStatisticalProcessing=2), "precipitation")


def test_cumulative_and_grid_safety() -> None:
    current = decode(message("precipitation", end=30), "precipitation", minutes=30)
    previous = decode(message("precipitation"), "precipitation")
    with pytest.raises(GribFormatError, match="preceding"):
        normalize_precipitation(current, interval_seconds=900)
    result = normalize_precipitation(current, previous=previous, interval_seconds=900)
    np.testing.assert_array_equal(result.values, 0)
    assert result.interval_start == RUN + np.timedelta64(15, "m")
    previous.values[0, 0] = 100
    result = normalize_precipitation(current, previous=previous, interval_seconds=900)
    assert np.isnan(result.values[0, 0])
    assert np.isfinite(result.values[1, 1])
    previous.lon[0] += 0.01
    with pytest.raises(GribFormatError, match="grids"):
        normalize_precipitation(current, previous=previous, interval_seconds=900)


def test_grid_relative_wind_and_native_amount() -> None:
    with pytest.raises(GribFormatError, match="grid-relative"):
        decode(message(uvRelativeToGrid=1))
    current = decode(message("precipitation"), "precipitation")
    amount = normalize_precipitation(current, interval_seconds=900)
    np.testing.assert_array_equal(amount.values, current.values)
    with pytest.raises(ValueError):
        normalize_precipitation(current, interval_seconds=0)
    with pytest.raises(ValueError):
        decode(message(), field="unknown")
    with pytest.raises(ValueError):
        decode(message(), interval=0)


def test_counters_require_same_reference_interval() -> None:
    current = decode(message("precipitation", end=45), "precipitation", minutes=45)
    wrong_previous = decode(message("precipitation"), "precipitation")
    with pytest.raises(GribFormatError, match="incompatible intervals"):
        normalize_precipitation(current, interval_seconds=900, previous=wrong_previous)
