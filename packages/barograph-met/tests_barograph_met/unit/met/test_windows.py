"""Bounded native-window reuse and forward-driver demand conversion, offline."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from baro.met.diffcocip import WeatherDemand
from baro.met.domain import Domain
from baro.met.materialize import WeatherRequest
from baro.met.weather import WeatherCube
from baro.met.windows import WeatherWindowProvider

START = np.datetime64("2020-01-01T00", "ns")
END = np.datetime64("2020-01-02T00", "ns")


def request(
    hour: int = 1,
    domain: Domain | None = None,
    pressure: tuple[float, float] = (25000, 30000),
) -> WeatherRequest:
    time = START + np.timedelta64(hour, "h")
    return WeatherRequest(domain or Domain(40, 41, 2, 3), time, time, pressure)


@pytest.fixture
def provider_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    from baro.met.weather import MET_FIELDS, RAD_FIELDS, RadiationCube

    calls = []
    source: Any = SimpleNamespace(
        vertical=SimpleNamespace(a=np.array([10000, 100000]), kind="pressure"),
        coverage=lambda: (START, END),
        fields=dict.fromkeys(("u", "v", "t", "q", "omega", "ciwc"), "native"),
    )
    radiation: Any = SimpleNamespace(coverage=lambda: (START, END))

    def materialize(
        source: Any, req: WeatherRequest, *, fields: tuple[str, ...], tau_cirrus: bool
    ) -> WeatherCube:
        calls.append(req)
        lon = np.array([np.floor(req.domain.lon_min), np.ceil(req.domain.lon_max)])
        lat = np.array([np.floor(req.domain.lat_min), np.ceil(req.domain.lat_max)])
        times = np.array([req.start, req.end + np.timedelta64(1, "h")]).astype(np.int64)
        names = (*fields, "tau_cirrus") if tau_cirrus else fields
        return WeatherCube(
            lon=lon,
            lat=lat,
            level_hpa=np.array(req.pressure_pa) / 100,
            time_ns=times,
            fields={k: np.zeros((2, 2, 2, 2)) for k in names},
            units={k: MET_FIELDS[k][1] for k in names},
            provenance={},
        )

    def materialize_rad(source: Any, req: WeatherRequest) -> RadiationCube:
        return RadiationCube(
            lon=[np.floor(req.domain.lon_min), np.ceil(req.domain.lon_max)],
            lat=[np.floor(req.domain.lat_min), np.ceil(req.domain.lat_max)],
            time_ns=np.array([req.start, req.end + np.timedelta64(1, "h")]).astype(np.int64),
            fields={k: np.zeros((2, 2, 2)) for k in RAD_FIELDS},
            units=dict.fromkeys(RAD_FIELDS, "W m-2"),
            provenance={},
            time_semantics="interval_midpoint",
        )

    monkeypatch.setattr("baro.met.windows.materialize_weather", materialize)
    monkeypatch.setattr("baro.met.windows.materialize_radiation", materialize_rad)

    def factory(**kwargs: Any) -> tuple[WeatherWindowProvider, list[WeatherRequest]]:
        return WeatherWindowProvider(source, radiation, **kwargs), calls

    return factory


def test_reuses_actual_native_support_and_clear(provider_factory: Any) -> None:
    provider, calls = provider_factory()
    first = provider.get(request())
    assert provider.get(request(2)) is first  # native bracket beyond requested time
    assert len(calls) == 1
    assert provider.cache_info()["hits"] == 1
    with pytest.raises(ValueError, match="read-only"):
        first[0].fields["u"][0, 0, 0, 0] = 1
    provider.clear()
    assert provider.cache_info()["bytes"] == 0
    provider.get(request())
    assert len(calls) == 2


def test_pressure_and_spatial_expansion_fetch(provider_factory: Any) -> None:
    provider, calls = provider_factory()
    provider.get(request())
    provider.get(request(pressure=(24000, 31000)))
    provider.get(request(domain=Domain(41, 42, 3, 4)))
    assert len(calls) == 3


def test_cache_budget_includes_axes_and_lru_eviction(provider_factory: Any) -> None:
    measure, _ = provider_factory()
    measure.get(request())
    size = measure.cache_info()["bytes"]
    assert size == 6 * 2**4 * 8 + 2 * 2**3 * 8 + (4 + 3) * 2 * 8
    provider, _ = provider_factory(max_cache_bytes=size)
    provider.get(request(1))
    provider.get(request(4))
    assert provider.cache_info()["entries"] == 1
    assert provider.cache_info()["evictions"] == 1
    assert provider.cache_info()["bytes"] == size
    tiny, _ = provider_factory(max_cache_bytes=1)
    assert tiny.get(request())[0].fields
    assert tiny.cache_info()["entries"] == 0


def test_prefetch_and_pressure_margin_clip_to_source_bounds(provider_factory: Any) -> None:
    provider, calls = provider_factory(
        margin_deg=0.25, pressure_margin_pa=20000, prefetch=np.timedelta64(48, "h")
    )
    provider.get(request())
    assert calls[0].domain == Domain(39.75, 41.25, 1.75, 3.25)
    assert calls[0].pressure_pa == (10000, 50000)
    assert calls[0].end == END
    with pytest.raises(ValueError, match="pressure coverage"):
        provider.get(request(pressure=(1000, 2000)))
    with pytest.raises(ValueError, match="time coverage"):
        provider.get(request(100))


def test_failed_materialization_does_not_enter_cache(
    provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _ = provider_factory()

    def failed(*args: Any, **kwargs: Any) -> None:
        raise ValueError("missing weather")

    monkeypatch.setattr("baro.met.windows.materialize_weather", failed)
    with pytest.raises(ValueError, match="missing weather"):
        provider.get(request())
    assert provider.cache_info()["entries"] == 0


def test_incomplete_materialized_window_refused(
    provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _ = provider_factory()
    old = provider.get(request())[0]
    provider.clear()
    monkeypatch.setattr("baro.met.windows.materialize_weather", lambda *a, **kw: old)
    with pytest.raises(ValueError, match="complete demand"):
        provider.get(request(pressure=(20000, 40000)))
    assert provider.cache_info()["entries"] == 0


def test_driver_callback_units_and_independent_export(provider_factory: Any) -> None:
    provider, calls = provider_factory()
    seconds = float((START - np.datetime64("2000-01-01", "ns")) / np.timedelta64(1, "s"))
    demand = WeatherDemand(np.array([[2, 40, 250, seconds + 3600], [3, 41, 300, seconds + 3600]]))
    met, rad = provider.for_diffcocip(demand)
    assert calls[0].pressure_pa == (25000, 30000)
    assert calls[0].start == START + np.timedelta64(1, "h")
    np.testing.assert_array_equal(met["level"], [250, 300])
    assert rad["time_s"][0] == seconds + 3600
    met["eastward_wind"].fill(99)
    assert provider.get(calls[0])[0].fields["u"].max() == 0


def test_nanosecond_conversion_rounds_outward_without_float_unix_epoch(
    provider_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _ = provider_factory()
    requests = []
    exported = SimpleNamespace(to_diffcocip=lambda: {})

    def capture(request: WeatherRequest) -> Any:
        requests.append(request)
        return exported, exported

    monkeypatch.setattr(provider, "get", capture)

    def demand(seconds: float) -> WeatherDemand:
        return WeatherDemand(np.array([[2, 40, 250, seconds]]))

    provider.for_diffcocip(demand(0.000000001))
    assert requests[-1].start == np.datetime64("2000-01-01T00:00:00.000000001")
    provider.for_diffcocip(demand(0.0000000011))
    assert requests[-1].end == np.datetime64("2000-01-01T00:00:00.000000002")
    provider.for_diffcocip(demand(-0.5))
    assert requests[-1].start == np.datetime64("1999-12-31T23:59:59.500000000")
    with pytest.raises(ValueError, match="finite"):
        provider.for_diffcocip(demand(float("nan")))
    with pytest.raises(ValueError, match="range"):
        provider.for_diffcocip(demand(1e100))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fields": []},
        {"max_cache_bytes": -1},
        {"margin_deg": -1},
        {"pressure_margin_pa": float("nan")},
        {"prefetch": np.timedelta64(-1, "h")},
    ],
)
def test_configuration_errors(provider_factory: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        provider_factory(**kwargs)


def test_native_reader_window_and_radiation_share_targeted_store() -> None:
    from baro.met.arco_surface import ArcoRadiationSource
    from baro.met.sources import GridSource
    from baro.met.store import ZarrStore
    from baro.met.vertical import pressure_levels
    from baro.met.weather import MET_FIELDS, RAD_FIELDS
    from synthetic import ArraySpec, memory_zarr

    specs = [
        ArraySpec(
            "time", np.arange(4, dtype="<i8"), (4,), ("time",), {"units": "hours since 2020-01-01"}
        ),
        ArraySpec("level", np.array([200, 250, 300, 400], dtype="<i8"), (4,), ("level",)),
        ArraySpec("latitude", np.array([42, 41, 40], dtype="<f4"), (3,), ("latitude",)),
        ArraySpec("longitude", np.arange(4, dtype="<f4"), (4,), ("longitude",)),
    ]
    names = ("u", "v", "t", "q", "omega", "tau_cirrus")
    for name in names:
        specs.append(
            ArraySpec(
                name,
                np.ones((4, 4, 3, 4), dtype="<f4"),
                (1, 4, 3, 4),
                ("time", "level", "latitude", "longitude"),
                {"units": MET_FIELDS[name][1]},
            )
        )
    for name in RAD_FIELDS:
        specs.append(
            ArraySpec(
                name,
                np.full((4, 3, 4), 7200, dtype="<f4"),
                (1, 3, 4),
                ("time", "latitude", "longitude"),
                {"units": "J m**-2"},
            )
        )
    store = ZarrStore(
        memory_zarr(
            specs,
            {
                "valid_time_start": "2020-01-01T00",
                "valid_time_stop": "2020-01-01T03",
            },
        )
    )
    grid = GridSource(
        "synthetic",
        store,
        pressure_levels([200, 250, 300, 400]),
        fields={n: n for n in (*names, *RAD_FIELDS)},
        level_array="level",
    )
    radiation = ArcoRadiationSource(grid_source=grid.select_fields(tuple(RAD_FIELDS)))
    provider = WeatherWindowProvider(grid, radiation)
    req = WeatherRequest(
        Domain(40.2, 40.8, 1.2, 1.8),
        START + np.timedelta64(1, "h"),
        START + np.timedelta64(2, "h"),
        (25000, 30000),
    )
    weather, rad = provider.get(req)
    assert np.asarray(weather.fields["u"]).min() == 1
    assert np.asarray(rad.fields["top_net_solar_radiation"]).min() == 2
    before = sum(stat.bytes for stat in grid.stats)
    assert provider.get(req)[0] is weather
    assert sum(stat.bytes for stat in grid.stats) == before
    assert len(weather.to_diffcocip()["time_s"]) == 2
