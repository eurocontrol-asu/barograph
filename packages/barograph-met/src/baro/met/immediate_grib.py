"""Strict normalization of deterministic immediate-forecast surface responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from baro.met.grib import GribFormatError, GribMessage, decode_message, parse_header

_PARAMS = {
    "u10": ((0, 2, 2), 103, 10, "m s**-1"),
    "v10": ((0, 2, 3), 103, 10, "m s**-1"),
    "t2m": ((0, 0, 0), 103, 2, "K"),
    "precipitation": ((0, 1, 52), 1, 0, "kg m**-2"),
    "rain": ((0, 1, 65), 1, 0, "kg m**-2"),
    "gusts": ((0, 2, 22), 103, 10, "m s**-1"),
}
_NEGATIVE_TOLERANCE = -0.001
_AMOUNTS = {"precipitation", "rain"}


@dataclass(frozen=True)
class ImmediateField:
    """One validated field with ascending axes and explicit accumulation bounds."""

    values: NDArray[np.float32]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    interval_start: np.datetime64 | None
    interval_end: np.datetime64


def _reference(date: int, time: int) -> np.datetime64:
    value = f"{date:08d}{time:04d}"
    return np.datetime64(
        f"{value[:4]}-{value[4:6]}-{value[6:8]}T{value[8:10]}:{value[10:12]}", "ns"
    )


def _semantics(buf: bytes, field: str, run: np.datetime64) -> GribMessage:
    import eccodes

    msg = parse_header(buf, 0)
    if msg.length != len(buf) or buf[-4:] != b"7777":
        raise GribFormatError("expected exactly one complete GRIB2 message")
    if not np.isfinite(msg.step_h):
        raise GribFormatError("unsupported GRIB forecast time unit")
    handle = eccodes.codes_new_from_message(buf)
    try:
        template = eccodes.codes_get(handle, "productDefinitionTemplateNumber")
        if template not in {0, 8}:
            raise GribFormatError("only deterministic GRIB templates 4.0 and 4.8 are supported")
        if field in {"u10", "v10"} and eccodes.codes_get(handle, "uvRelativeToGrid"):
            raise GribFormatError("grid-relative wind components are unsupported")
        actual_run = _reference(
            int(eccodes.codes_get(handle, "dataDate")),
            int(eccodes.codes_get(handle, "dataTime")),
        )
        if actual_run != run:
            raise GribFormatError("response reference run differs from requested run")
        actual_valid = _reference(
            int(eccodes.codes_get(handle, "validityDate")),
            int(eccodes.codes_get(handle, "validityTime")),
        )
        expected_valid = run + np.timedelta64(round(msg.step_h * 3600), "s")
        if actual_valid != expected_valid:
            raise GribFormatError("inconsistent GRIB validity and forecast interval")
        units = eccodes.codes_get(handle, "units")
    finally:
        eccodes.codes_release(handle)
    param, level_type, level, unit = _PARAMS[field]
    if msg.param != param or msg.level_type != level_type or msg.level != level:
        raise GribFormatError(f"unexpected parameter or vertical level for {field}")
    if units != unit:
        raise GribFormatError(f"unexpected unit {units!r} for {field}; expected {unit}")
    return msg


def _bounds(
    msg: GribMessage,
    field: str,
    run: np.datetime64,
    valid_time: np.datetime64,
    interval_seconds: int,
) -> np.datetime64 | None:
    if not np.isfinite(msg.step_h):
        raise GribFormatError("unsupported GRIB forecast time unit")
    end = run + np.timedelta64(round(msg.step_h * 3600), "s")
    if end != valid_time or end < run:
        raise GribFormatError("response validity differs from requested time")
    if field not in _AMOUNTS | {"gusts"}:
        if msg.statistical_process is not None:
            raise GribFormatError("instantaneous field has statistical processing")
        return None
    process = 1 if field in _AMOUNTS else 2
    if msg.statistical_process != process or msg.start_step_h is None:
        raise GribFormatError("expected an accumulation or interval maximum")
    if not np.isfinite(msg.start_step_h):
        raise GribFormatError("unsupported GRIB interval time unit")
    start = run + np.timedelta64(round(msg.start_step_h * 3600), "s")
    native_start = end - np.timedelta64(interval_seconds, "s")
    if start >= end or start < run:
        raise GribFormatError("invalid statistical interval")
    if start != native_start and not (field in _AMOUNTS and start == run):
        raise GribFormatError("statistical interval is neither native nor cumulative from run")
    return start


def decode_surface_response(
    buf: bytes,
    *,
    field: str,
    run: np.datetime64,
    valid_time: np.datetime64,
    interval_seconds: int,
) -> ImmediateField:
    """Decode a single field; cumulative precipitation still needs differencing.

    Precipitation is water-equivalent millimetres (1 kg/m² = 1 mm). Statistical
    probabilities, percentiles and multi-range products are rejected, including
    probabilistic PIAF products. Gusts must be a scalar speed interval maximum.
    """
    if field not in _PARAMS:
        raise ValueError(f"unsupported surface field: {field}")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    run, valid_time = np.datetime64(run, "ns"), np.datetime64(valid_time, "ns")
    msg = _semantics(buf, field, run)
    start = _bounds(msg, field, run, valid_time, interval_seconds)
    values, lat, lon = decode_message(buf)
    return ImmediateField(values[::-1], lat[::-1], lon, start, valid_time)


def normalize_precipitation(
    current: ImmediateField,
    *,
    interval_seconds: int,
    previous: ImmediateField | None = None,
) -> ImmediateField:
    """Return a native interval amount, subtracting a compatible cumulative field.

    Negative counter differences below numerical tolerance are missing locally.
    A previous field is required only when the current interval is cumulative.
    """
    if interval_seconds <= 0 or current.interval_start is None:
        raise ValueError("expected positive interval and statistical field")
    start = current.interval_end - np.timedelta64(interval_seconds, "s")
    values = current.values.copy()
    if current.interval_start != start:
        if previous is None:
            raise GribFormatError("cumulative precipitation requires preceding counter")
        if previous.interval_start != current.interval_start or previous.interval_end != start:
            raise GribFormatError("precipitation counters have incompatible intervals")
        if not np.array_equal(current.lat, previous.lat) or not np.array_equal(
            current.lon, previous.lon
        ):
            raise GribFormatError("precipitation counters have different grids")
        values -= previous.values
    values = np.where(values < _NEGATIVE_TOLERANCE, np.nan, np.maximum(values, 0)).astype(
        np.float32
    )
    return ImmediateField(values, current.lat, current.lon, start, current.interval_end)
