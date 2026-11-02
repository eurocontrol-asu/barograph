"""Deferred native reads, exact indexing and bounded retention without network I/O."""

from typing import Any

import numpy as np
import pytest

from baro.met.base import DataUnavailableError
from baro.met.pycontrails import PycontrailsWeather
from unit.met.test_materialize import START, source


def test_configuration_and_periodic_coordinates() -> None:
    with pytest.raises(ValueError, match="does not supply"):
        PycontrailsWeather(source(truncated=True))
    with pytest.raises(ValueError, match="unique"):
        PycontrailsWeather(source(truncated=True), fields=[])
    with pytest.raises(ValueError, match="nonnegative"):
        PycontrailsWeather(source(truncated=True), fields=["t"], max_cache_bytes=-1)
    src = source(truncated=True).select_fields(["u"])
    with pytest.raises(ValueError, match="does not supply"):
        PycontrailsWeather(src, fields=["tau_cirrus"])


def test_lazy_native_indexing_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pycontrails")
    bridge = PycontrailsWeather(source(truncated=True), fields=["t", "u"], max_cache_bytes=256)
    before = bridge.source.store.store.stats.bytes
    met, rad = bridge.datasets(START, START + np.timedelta64(1, "h"))
    assert bridge.source.store.store.stats.bytes == before
    assert rad is None
    assert bridge.cache_info()["misses"] == 0
    assert met.data.attrs["provider"] == "unknown"
    field = met.data.air_temperature
    selected = field.isel(longitude=[2, 0], latitude=1, level=[6, 5], time=0).values
    assert selected.shape == (2, 2)
    # Native fixture: source latitude is descending and pressure level IDs
    # select the original array positions, including the truncated view offset.
    pattern = np.arange(4 * 11 * 3 * 3).reshape(4, 11, 3, 3)
    expected = (220 + pattern.astype("float32") * 0.001)[0, [9, 8], 1][:, [2, 0]].T
    np.testing.assert_array_equal(selected, expected)
    assert bridge.cache_info()["field_reads"] == {"t": 1}
    np.testing.assert_array_equal(
        selected, field.isel(longitude=[2, 0], latitude=1, level=[6, 5], time=0).values
    )
    assert bridge.cache_info()["hits"] == 1
    _ = met.data.eastward_wind.isel(longitude=0, latitude=0, level=5, time=0).values
    assert bridge.cache_info()["bytes"] <= bridge.max_cache_bytes
    bridge.clear()
    assert bridge.cache_info()["entries"] == 0
    assert field.isel(longitude=slice(0, 0)).values.size == 0
    with pytest.raises(ValueError, match="coverage"):
        bridge.datasets(START - np.timedelta64(1, "h"), START)

    def broken(*args: Any, **kwargs: Any) -> Any:
        return np.full((1, 1, 1, 1), np.nan)

    monkeypatch.setattr(bridge, "_materialize", broken)
    with pytest.raises(DataUnavailableError, match="Nonfinite"):
        _ = field.isel(longitude=0, latitude=0, level=0, time=0).values


def test_cache_eviction_owned_arrays() -> None:
    bridge = PycontrailsWeather(source(truncated=True), fields=["u"], max_cache_bytes=8)
    axes = (np.array([1.0]), np.array([45.0]), np.array([250.0]), np.array([START]))
    a = bridge._read("u", axes)
    assert a.flags.owndata and not a.flags.writeable
    other = (*axes[:2], np.array([300.0]), axes[3])
    bridge._read("u", other)
    assert bridge.cache_info()["evictions"] == 1
    assert bridge.cache_info()["bytes"] == bridge.max_cache_bytes


def test_lazy_radiation_only_reads_selected_field(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pycontrails")
    from unit.met.test_arco_surface import SOLAR, THERMAL
    from unit.met.test_arco_surface import source as radiation_source

    radiation = radiation_source()
    configured = tuple(radiation.variables)
    reads = []
    fetch = radiation.fetch

    def record(*args: Any, **kwargs: Any) -> Any:
        reads.append(kwargs["fields"])
        return fetch(*args, **kwargs)

    monkeypatch.setattr(radiation, "fetch", record)
    bridge = PycontrailsWeather(source(truncated=True), radiation, fields=["t"])
    _, rad = bridge.datasets(START, START + np.timedelta64(1, "h"))
    assert rad is not None
    assert bridge.cache_info()["misses"] == 0
    np.testing.assert_allclose(rad.data[SOLAR].isel(longitude=1, latitude=1).values, [[1, 2, 3]])
    assert bridge.cache_info()["field_reads"] == {SOLAR: 1}
    assert tuple(radiation.variables) == configured
    assert reads and all(fields == (SOLAR,) for fields in reads)
    assert THERMAL in rad.data
    assert rad.data[SOLAR].attrs["units"] == "W m**-2"
    assert rad.data.time.attrs["shift_radiation_time"]


def test_native_phase_origin_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pycontrails")
    from baro.met import materialize as module
    from baro.met.materialize import materialize_weather
    from unit.met.test_materialize import request

    src = source(truncated=True)
    origin = START + np.timedelta64(30, "m")
    monkeypatch.setattr(src, "time_origin", origin, raising=False)
    bridge = PycontrailsWeather(src, fields=["t"])
    met, _ = bridge.datasets(START + np.timedelta64(40, "m"), START + np.timedelta64(70, "m"))
    np.testing.assert_array_equal(met.data.time, origin + np.arange(2) * np.timedelta64(1, "h"))
    actual = module._times
    calls = []

    def record(req: Any, epoch: Any, *args: Any) -> Any:
        calls.append(epoch)
        # Native fixture fetch is hourly; only test forwarded origin independently.
        return actual(req, START, *args)

    monkeypatch.setattr(module, "_times", record)
    materialize_weather(src, request(), fields=["t"])
    assert calls == [origin]


def test_shared_byte_cache_and_disabled_decoded_retention(tmp_path: Any) -> None:
    from baro.met.vertical import model_levels
    from unit.met.test_arco_surface import source as radiation_source

    src = source(truncated=True)
    src.vertical = model_levels([1, 2])
    with pytest.raises(ValueError, match="pressure levels"):
        PycontrailsWeather(src, fields=["u"])
    src = source(truncated=True)
    from baro.met.sources import GridSource

    src = GridSource(
        "cache-test", src.store, src.vertical, fields={"u": "wind"}, level_array="level"
    )
    bridge = PycontrailsWeather(
        src,
        radiation_source(),
        fields=["u"],
        cache_dir=tmp_path,
        max_cache_bytes=0,
    )
    axes = (np.array([1.0]), np.array([45.0]), np.array([250.0]), np.array([START]))
    bridge._read("u", axes)
    before = sum(s.bytes for s in bridge.source.stats)
    bridge._read("u", axes)
    assert sum(s.bytes for s in bridge.source.stats) == before
    assert bridge.cache_info()["entries"] == 0


def test_recent_weather_bypasses_bridge_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = PycontrailsWeather(source(truncated=True), fields=["u"])
    axes = (
        np.array([1.0]),
        np.array([45.0]),
        np.array([250.0]),
        np.array([np.datetime64("now", "ns")]),
    )
    calls = []

    def materialize(*args: Any) -> Any:
        calls.append(args)
        return np.ones((1, 1, 1, 1)) * len(calls)

    monkeypatch.setattr(bridge, "_materialize", materialize)
    assert bridge._read("u", axes).item() == 1
    assert bridge._read("u", axes).item() == 2
    assert bridge.cache_info()["entries"] == 0


@pytest.mark.filterwarnings("ignore:Pressure level is below.*:UserWarning")
def test_lazy_tau_uses_complete_native_upper_column() -> None:
    pytest.importorskip("pycontrails")
    from baro.met.materialize import materialize_weather
    from unit.met.test_materialize import request

    src = source()
    bridge = PycontrailsWeather(src, fields=["t", "tau_cirrus"])
    axes = (np.array([1.0]), np.array([45.0]), np.array([250.0]), np.array([START]))
    actual = bridge._read("tau_cirrus", axes)
    full = materialize_weather(src, request("00:00", "00:00"), fields=["t"], tau_cirrus=True)
    native = (
        np.asarray(full.lon),
        np.asarray(full.lat),
        np.asarray(full.level_hpa),
        np.asarray(full.time_ns).astype("datetime64[ns]"),
    )
    ids = [np.searchsorted(axis, wanted) for axis, wanted in zip(native, axes, strict=True)]
    expected = np.asarray(full.fields["tau_cirrus"])[np.ix_(*ids)]
    np.testing.assert_array_equal(actual, expected)
    assert bridge.cache_info()["field_reads"] == {"tau_cirrus": 1}


def test_optional_engines_are_imported_only_for_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    from baro.met import pycontrails as module

    def missing(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module, "import_module", missing)
    bridge = PycontrailsWeather(source(truncated=True), fields=["u"])
    assert bridge.cache_info()["misses"] == 0
    with pytest.raises(ModuleNotFoundError):
        bridge.datasets(START, START)


def test_native_longitude_metadata_validates_seam() -> None:
    pytest.importorskip("pycontrails")
    src = source(truncated=True)
    src.lon = np.arange(0, 360, 90, dtype=np.float64)
    met, _ = PycontrailsWeather(src, fields=["u"]).datasets(START, START)
    np.testing.assert_array_equal(met.data.longitude, [-180, -90, 0, 90, 180])
    for lon, message in (([0, 0], "unique"), ([170, 180, 190], "antimeridian")):
        src.lon = np.array(lon, dtype=np.float64)
        with pytest.raises(ValueError, match=message):
            PycontrailsWeather(src, fields=["u"]).datasets(START, START)
