"""Native window selection, time bracketing and full-column cirrus parity."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np
import pytest

from baro.met import Domain, model_levels, pressure_levels
from baro.met.materialize import WeatherRequest, materialize_weather
from baro.met.sources import GridSource
from baro.met.store import ZarrStore
from synthetic import ArraySpec, memory_zarr

LEVELS = np.array([1, 10, 50, 100, 200, 250, 300, 400, 600, 800, 1000], dtype="<i8")
START = np.datetime64("2020-01-01T00", "ns")
DOMAIN = Domain(44, 45, 1, 2)


def source(*, bad_units: bool = False, truncated: bool = False) -> GridSource:
    shape = (4, len(LEVELS), 3, 3)
    pattern = np.arange(np.prod(shape), dtype="<f4").reshape(shape)
    dims = ("time", "level", "latitude", "longitude")
    specs = [
        ArraySpec(
            "time", np.arange(4, dtype="<i8"), (4,), ("time",), {"units": "hours since 2020-01-01"}
        ),
        ArraySpec("level", LEVELS, (len(LEVELS),), ("level",)),
        ArraySpec("latitude", np.array([46, 45, 44], dtype="<f4"), (3,), ("latitude",)),
        ArraySpec("longitude", np.array([0, 1, 2], dtype="<f4"), (3,), ("longitude",)),
        ArraySpec(
            "temperature",
            220 + pattern * 0.001,
            (1, 1, 3, 3),
            dims,
            {"units": "C" if bad_units else "K"},
        ),
        ArraySpec("ciwc", 1e-5 + pattern * 1e-9, (1, 1, 3, 3), dims, {"units": "kg kg**-1"}),
        ArraySpec("wind", pattern, (1, 1, 3, 3), dims, {"units": "m s**-1"}),
    ]
    store = ZarrStore(
        memory_zarr(
            specs, {"valid_time_start": "2020-01-01T00", "valid_time_stop": "2020-01-01T03"}
        )
    )
    return GridSource(
        "synthetic",
        store,
        pressure_levels(LEVELS[3:] if truncated else LEVELS),
        fields={"t": "temperature", "u": "wind", "ciwc": "ciwc", "unused": "missing"},
        level_array="level",
    )


def request(start: str = "00:20", end: str = "01:40") -> WeatherRequest:
    return WeatherRequest(
        DOMAIN,
        np.datetime64(f"2020-01-01T{start}"),
        np.datetime64(f"2020-01-01T{end}"),
        (25000, 30000),
    )


def test_missing_cirrus_dependency_fails_before_field_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = source()
    original = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "pycontrails.models.tau_cirrus":
            raise ImportError("not installed")
        return original(name, package)

    def no_fetch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("field reads must not precede dependency validation")

    monkeypatch.setattr(importlib, "import_module", missing)
    monkeypatch.setattr(GridSource, "fetch", no_fetch)
    with pytest.raises(ImportError, match="optional pycontrails"):
        materialize_weather(src, request(), fields=["u"], tau_cirrus=True)


def test_selective_fields_levels_and_axis_order(monkeypatch: pytest.MonkeyPatch) -> None:
    original = GridSource.fetch
    calls: list[Any] = []

    def record(self: GridSource, *args: Any, **kwargs: Any) -> Any:
        cube = original(self, *args, **kwargs)
        calls.append((tuple(self.fields), cube))
        return cube

    monkeypatch.setattr(GridSource, "fetch", record)
    src = source()
    cube = materialize_weather(src, request(), fields=["u"])
    assert set(cube.fields) == {"u"}
    assert all(names == ("u",) for names, _ in calls)
    assert len(calls) == 3
    assert np.asarray(cube.level_hpa).tolist() == [200, 250, 300, 400]
    assert np.asarray(cube.lat).tolist() == [44, 45]
    assert np.asarray(cube.lon).tolist() == [1, 2]
    for i, (_, native) in enumerate(calls):
        np.testing.assert_array_equal(
            np.asarray(cube.fields["u"])[..., i], native.fields["u"][:, ::-1].transpose(2, 1, 0)
        )
    assert "unused" in src.fields


@pytest.mark.parametrize(
    "start,end,hours",
    [
        ("00:00", "01:00", [0, 1]),
        ("01:00", "01:00", [1, 2]),
        ("00:20", "01:40", [0, 1, 2]),
        ("03:00", "03:00", [2, 3]),
    ],
)
def test_native_time_bracketing(start: str, end: str, hours: list[int]) -> None:
    cube = materialize_weather(source(), request(start, end), fields=["t"])
    np.testing.assert_array_equal(
        cube.time_ns, (START + np.array(hours) * np.timedelta64(1, "h")).astype("int64")
    )


def test_invalid_native_metadata_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="native units"):
        materialize_weather(source(bad_units=True), request(), fields=["t"])
    src = source()
    src.vertical = model_levels([60, 70, 80])
    with pytest.raises(ValueError, match="pressure levels"):
        materialize_weather(src, request(), fields=["t"])
    with pytest.raises(ValueError, match="does not supply"):
        materialize_weather(source(), request(), fields=["q"])
    with pytest.raises(ValueError, match="unique"):
        materialize_weather(source(), request(), fields=["u", "u"])
    with pytest.raises(ValueError, match="units must"):
        materialize_weather(source(), request(), fields=["t"], units={})


def test_cirrus_rejects_truncated_native_column() -> None:
    with pytest.raises(ValueError, match="complete native"):
        materialize_weather(source(truncated=True), request(), fields=["t"], tau_cirrus=True)


@pytest.mark.parametrize("pressure", [(0, 1), (30000, 25000), (np.nan, 30000)])
def test_request_rejects_bad_pressure(pressure: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="positive pressures"):
        WeatherRequest(DOMAIN, START, START, pressure)


@pytest.mark.parametrize(
    "start,end", [(np.datetime64("NaT", "ns"), START), (START + np.timedelta64(1, "h"), START)]
)
def test_request_rejects_bad_time(start: np.datetime64, end: np.datetime64) -> None:
    with pytest.raises(ValueError, match="time bounds"):
        WeatherRequest(DOMAIN, start, end, (25000, 30000))


@pytest.mark.skipif(importlib.util.find_spec("pycontrails") is None, reason="optional engine")
@pytest.mark.filterwarnings("ignore:Pressure level is below.*:UserWarning")
def test_cirrus_matches_complete_native_column() -> None:
    tau = importlib.import_module("pycontrails.models.tau_cirrus").tau_cirrus
    src = source()
    full = materialize_weather(
        src,
        WeatherRequest(DOMAIN, START, START + np.timedelta64(2, "h"), (100, 100000)),
        fields=["t", "ciwc"],
    )
    reference = (
        tau(full.to_pycontrails(require_cocip=False))
        .transpose("longitude", "latitude", "level", "time")
        .values
    )
    cropped = materialize_weather(src, request(), fields=["t"], tau_cirrus=True)
    positions = np.searchsorted(full.level_hpa, cropped.level_hpa)
    np.testing.assert_allclose(
        np.asarray(cropped.fields["tau_cirrus"]), reference[:, :, positions, :], rtol=1e-14, atol=0
    )
    assert cropped.provenance["cirrus_height"] == "ISA"


def test_rejects_out_of_source_coverage() -> None:
    for req, match in [
        (WeatherRequest(Domain(40, 41, 1, 2), START, START, (25000, 30000)), "spatial coverage"),
        (WeatherRequest(DOMAIN, START, START, (50, 30000)), "pressure coverage"),
        (request("03:00", "04:00"), "time coverage"),
    ]:
        with pytest.raises(ValueError, match=match):
            materialize_weather(source(), req, fields=["t"])


def test_rejects_grid_changes_across_time(monkeypatch: pytest.MonkeyPatch) -> None:
    original = GridSource.fetch

    def changed(self: GridSource, *args: Any, **kwargs: Any) -> Any:
        cube = original(self, *args, **kwargs)
        if cube.hour > START:
            cube.lon = cube.lon + 0.1
        return cube

    monkeypatch.setattr(GridSource, "fetch", changed)
    with pytest.raises(ValueError, match="grids must agree"):
        materialize_weather(source(), request(), fields=["t"])
