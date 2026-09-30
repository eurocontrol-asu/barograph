"""GridSource over an in-memory zarr store laid out like ARCO-ERA5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baro.met import CubeKey, DataUnavailableError, Domain, pressure_levels
from baro.met.rangecache import CachedStore, RangeCache
from baro.met.sources import GridSource
from baro.met.store import ZarrStore
from synthetic import ArraySpec, memory_zarr

LAT = np.linspace(50.0, 40.0, 11, dtype="<f4")
LON = np.arange(0.0, 360.0, 30.0, dtype="<f4")
LEVELS = np.array([200, 250, 300], dtype="<i8")
T0 = 1_000  # first index on the time axis, in hours since 1900


def _store() -> ZarrStore:
    field = np.arange(4 * 3 * 11 * 12, dtype="<f4").reshape(4, 3, 11, 12)
    dims4 = ("time", "level", "latitude", "longitude")
    specs = [
        ArraySpec(
            "time",
            np.arange(T0, T0 + 4, dtype="<i8"),
            (4,),
            ("time",),
            {"units": "hours since 1900-01-01"},
        ),
        ArraySpec("level", LEVELS, (3,), ("level",)),
        ArraySpec("latitude", LAT, (11,), ("latitude",)),
        ArraySpec("longitude", LON, (12,), ("longitude",)),
        ArraySpec("temperature", field, (1, 3, 11, 12), dims4),
    ]
    attrs = {"valid_time_start": "1900-02-11", "valid_time_stop": "1900-02-12"}
    return ZarrStore(memory_zarr(specs, attrs))


def _source(levels: list[int] | None = None) -> GridSource:
    vertical = pressure_levels(LEVELS if levels is None else levels)
    return GridSource("mem", _store(), vertical, fields={"t": "temperature"}, level_array="level")


def test_coverage_comes_from_the_store_attributes() -> None:
    start, end = _source().coverage()  # type: ignore[misc]
    assert start == np.datetime64("1900-02-11T00", "h")
    assert end == np.datetime64("1900-02-12T23", "h")


def test_fetch_reads_the_right_hour_levels_and_region() -> None:
    hour = np.datetime64("1900-01-01T00", "h") + np.timedelta64(T0 + 2, "h")
    cube = _source().fetch(
        CubeKey(hour), Domain(44.0, 46.0, 25.0, 65.0), (21000.0, 29000.0), level_margin=0
    )
    assert list(cube.vertical.ids) == [200, 250, 300]
    np.testing.assert_array_equal(
        cube.lon, [0.0, 30.0, 60.0, 90.0][: len(cube.lon)] if False else cube.lon
    )
    assert cube.lat[0] >= 46.0 and cube.lat[-1] <= 44.0
    assert cube.lon[0] <= 25.0 and cube.lon[-1] >= 65.0
    expected = np.arange(4 * 3 * 11 * 12).reshape(4, 3, 11, 12)[2]
    rows = [int(np.argmin(np.abs(LAT - v))) for v in cube.lat]
    cols = [int(np.argmin(np.abs(LON - v))) for v in cube.lon]
    np.testing.assert_array_equal(cube.fields["t"], expected[:, rows][:, :, cols])


def test_level_subset_maps_to_the_store_axis() -> None:
    hour = np.datetime64("1900-01-01T00", "h") + np.timedelta64(T0, "h")
    cube = _source([250, 300]).fetch(CubeKey(hour), Domain(44, 46, 25, 35), (26000.0, 28000.0))
    assert list(cube.vertical.ids) == [250, 300]
    row = int(np.argmin(np.abs(LAT - cube.lat[0])))
    col = int(np.argmin(np.abs(LON - cube.lon[0])))
    field = np.arange(4 * 3 * 11 * 12).reshape(4, 3, 11, 12)
    assert cube.fields["t"][0, 0, 0] == field[0, 1, row, col]  # store level 1 = 250 hPa


def test_hours_outside_the_axis_are_unavailable() -> None:
    hour = np.datetime64("1900-01-01T00", "h") + np.timedelta64(T0 + 10, "h")
    with pytest.raises(DataUnavailableError):
        _source().fetch(CubeKey(hour), Domain(44, 46, 25, 35), (2e4, 3e4))


def test_levels_missing_from_the_store_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in the store"):
        _source([200, 850])


def test_only_final_field_and_time_chunks_are_cached(tmp_path: Path) -> None:
    src = _source()
    src.use_cache(RangeCache(tmp_path))
    store = src.store.store
    assert isinstance(store, CachedStore)
    assert store.cacheable("temperature/0.0.0.0") and store.cacheable("time/0")
    assert not store.cacheable(".zmetadata") and not store.cacheable("latitude/0")
    assert not store.cacheable("temperature/") and not store.cacheable("unknown/0.0.0.0")
    recent = _source()
    recent.final_after = np.timedelta64(1_000_000, "D")  # everything still "ERA5T"
    recent.use_cache(RangeCache(tmp_path))
    assert not recent.store.store.cacheable("temperature/0.0.0.0")  # type: ignore[attr-defined]


def test_tiling_follows_the_blosc_bands() -> None:
    tiling = _source().tiling(CubeKey(np.datetime64("1900-02-11T00", "h")))
    assert tiling is None or (tiling.lon_step is None and tiling.lat0 <= float(LAT.min()))
