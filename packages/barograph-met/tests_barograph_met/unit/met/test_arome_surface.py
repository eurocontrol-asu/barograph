"""Native fixed-run surface semantics without network access."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from baro.met.arome_surface import AromeSurfaceSource
from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain
from baro.met.grib import GribIndex, GribMessage


def test_precise_run_and_lead_files() -> None:
    with pytest.raises(ValueError, match="exact"):
        AromeSurfaceSource(run=np.datetime64("2026-10-06T00:01"))
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"))
    assert "00H06H" in source.file_key(source.run, 6)
    assert "07H12H" in source.file_key(source.run, 7)
    assert "49H51H" in source.file_key(source.run, 51)
    with pytest.raises(DataUnavailableError):
        source.file_key(source.run, 52)


def test_hourly_precipitation_subtracts_same_run_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"), fields={"p": "precipitation"})
    message = GribMessage(0, 8, 0, 1, 52, 1, 0, 7, 0, 0, 4, 0, 1)
    monkeypatch.setattr(source.indexes, "get", lambda key: GribIndex(key, 8, [message]))
    lat, lon = np.array([49.0, 48.0]), np.array([2.0, 3.0])
    monkeypatch.setattr("baro.met.arome_surface.fetch_messages", lambda *args: [b"field"])
    monkeypatch.setattr(
        "baro.met.arome_surface.decode_message",
        lambda b: (np.full((2, 2), 9, dtype=np.float32), lat, lon),
    )
    calls = []

    def cumulative(name: str, lead: int) -> tuple[Any, Any, Any]:
        calls.append((name, lead))
        return np.full((2, 2), 6, dtype=np.float32), lat, lon

    monkeypatch.setattr(source._reader, "cumulative", cumulative)
    field, _, _ = source._reader.field("precipitation", 7)
    assert calls == [("precipitation", 6)]
    assert np.all(field == 3)


def test_subset_skips_unavailable_initial_period_and_reverses_latitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"))
    calls = []

    def get_field(name: str, lead: int) -> tuple[Any, Any, Any]:
        calls.append((name, lead))
        return (
            np.array([[10, 10], [20, 20]], dtype=np.float32),
            np.array([49.0, 48.0]),
            np.array([2.0, 3.0]),
        )

    monkeypatch.setattr(source._reader, "field", get_field)
    cube = source.fetch(
        CubeKey(source.run, source.run), Domain.from_points([48.5], [2.5]), fields=["t2m"]
    )
    assert calls == [("t2m", 0)]
    assert cube.lat.tolist() == [48.0, 49.0]
    assert cube.fields["t2m"][:, 0].tolist() == [20, 10]


def test_latest_checks_previous_file_and_skips_initial_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    newest = np.datetime64("2026-10-06T09")
    previous = np.datetime64("2026-10-06T06")
    older = np.datetime64("2026-10-06T03")
    monkeypatch.setattr(
        AromeSurfaceSource, "published_runs", lambda self: [older, previous, newest]
    )
    checked = []

    def published(self: Any, key: str) -> bool:
        checked.append(key)
        # Previous run reaches lead7, but its previouslead6 file is absent.
        return not ("2026-10-06T06" in key and "00H06H" in key)

    monkeypatch.setattr("baro.met.arpege.ArpegeSource._published", published)
    source = AromeSurfaceSource.latest(
        np.datetime64("2026-10-06T13"), fields={"p": "precipitation"}
    )
    # newest09 lead4 file exists; remove it then ensure previous06 lead7
    # also cannot be selected without previous06 lead6 counter.
    assert source.run == newest

    def published_without_newest(self: Any, key: str) -> bool:
        return "2026-10-06T09" not in key and published(self, key)

    monkeypatch.setattr("baro.met.arpege.ArpegeSource._published", published_without_newest)
    assert (
        AromeSurfaceSource.latest(np.datetime64("2026-10-06T13"), fields={"p": "precipitation"}).run
        == older
    )
    monkeypatch.setattr("baro.met.arpege.ArpegeSource._published", lambda self, key: True)
    assert AromeSurfaceSource.latest(newest, fields={"p": "precipitation"}).run == previous
    with pytest.raises(ValueError, match="NaT"):
        AromeSurfaceSource.latest(np.datetime64("NaT", "s"))


def test_subsecond_initialization_is_not_rounded() -> None:
    with pytest.raises(ValueError, match="exact"):
        AromeSurfaceSource(run=np.datetime64("2026-10-06T00:00:00.000000001"))


def test_material_counter_decrease_only_invalidates_affected_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"))
    message = GribMessage(0, 8, 0, 1, 52, 1, 0, 6, 0, 0, 4, 0, 1)
    monkeypatch.setattr(source.indexes, "get", lambda key: GribIndex(key, 8, [message]))
    lat, lon = np.array([49.0, 48.0]), np.array([2.0, 3.0])
    monkeypatch.setattr("baro.met.arome_surface.fetch_messages", lambda *args: [b"field"])
    monkeypatch.setattr(
        "baro.met.arome_surface.decode_message",
        lambda b: (np.array([[9.0, 1.0], [9.0, 9.0]], dtype=np.float32), lat, lon),
    )
    monkeypatch.setattr(
        source._reader,
        "cumulative",
        lambda name, lead: (np.full((2, 2), 6, dtype=np.float32), lat, lon),
    )
    field, _, _ = source._reader.field("precipitation", 6)
    assert field[0, 0] == 3
    assert np.isnan(field[0, 1])


@pytest.mark.parametrize(
    ("start", "process", "lead", "valid"),
    [(5, 1, 6, True), (0, 1, 1, True), (2, 1, 6, False), (0, 2, 6, False)],
)
def test_precipitation_interval_validation(
    start: int, process: int, lead: int, valid: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"))
    message = GribMessage(0, 8, 0, 1, 52, 1, 0, lead, 0, 0, 4, start, process)
    decoded = np.ones((2, 2), dtype=np.float32), np.array([49.0, 48.0]), np.array([2.0, 3.0])
    monkeypatch.setattr(source._reader, "message", lambda *args: message)
    monkeypatch.setattr(source._reader, "decode", lambda *args: decoded)
    key = CubeKey(source.run + np.timedelta64(lead, "h"), source.run)
    domain = Domain.from_points([48.5], [2.5])
    if valid:
        cube = source.fetch(key, domain, fields=["precipitation"])
        assert np.all(cube.fields["precipitation"] == 1)
    else:
        with pytest.raises(DataUnavailableError, match="interval"):
            source.fetch(key, domain, fields=["precipitation"])


def test_native_gust_selection_rejects_wrong_statistical_meaning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = AromeSurfaceSource(run=np.datetime64("2026-10-06T00"))
    mean = GribMessage(0, 8, 0, 2, 22, 103, 10, 6, 0, 0, 4, 5, 0)
    maximum = GribMessage(8, 8, 0, 2, 22, 103, 10, 6, 0, 0, 4, 5, 2)
    monkeypatch.setattr(source.indexes, "get", lambda key: GribIndex(key, 16, [mean, maximum]))
    assert source._reader.message("package", "gusts", 6) is maximum
    with pytest.raises(DataUnavailableError, match="missing or ambiguous"):
        source._reader.message("package", "gusts", 7)
