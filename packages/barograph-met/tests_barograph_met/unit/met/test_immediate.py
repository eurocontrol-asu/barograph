"""Fixed-run immediate-source configuration and metadata selection contracts."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain
from baro.met.immediate import AromePISource, PiafSource, _CoverageReader, _subsets
from baro.met.wcs_catalog import CoverageGrid

_RUN = np.datetime64("2026-10-07T12:00", "ns")


def _grid(leads: tuple[int, ...], heights: tuple[float, ...] = ()) -> CoverageGrid:
    return CoverageGrid(np.array([49.0, 48.0]), np.array([2.0, 3.0]), leads, heights)


@pytest.mark.parametrize("run", ["NaT", "2026-10-07T12:00:00.000000001", "2026-10-07T12:15"])
def test_arome_rejects_noninitialization(run: str) -> None:
    with pytest.raises(ValueError, match="initialization"):
        AromePISource(run=np.datetime64(run, "ns"), token="fixture")


@pytest.mark.parametrize("tile_deg", [0, -1, float("nan"), float("inf")])
def test_invalid_tile_does_not_close_injected_client(tile_deg: float) -> None:
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="tile_deg"):
            AromePISource(run=_RUN, token="fixture", tile_deg=tile_deg, client=client)
        assert not client.is_closed


def test_configuration_and_native_endpoints() -> None:
    with AromePISource(run=_RUN, token="fixture", resolution="0.01") as source:
        assert set(source.fields) == {"t2m", "precipitation", "gusts"}
        left, right = source.cube_keys(_RUN + np.timedelta64(901, "s"))
        assert left == CubeKey(_RUN + np.timedelta64(900, "s"), _RUN)
        assert right == CubeKey(_RUN + np.timedelta64(1800, "s"), _RUN)
        assert source.coverage() == (_RUN + np.timedelta64(1, "ns"), _RUN + np.timedelta64(6, "h"))
    with PiafSource(run=_RUN + np.timedelta64(5, "m"), token="fixture") as source:
        assert source.time_step == np.timedelta64(300, "s")
        assert source.coverage()[1] == _RUN + np.timedelta64(200, "m")
    with pytest.raises(ValueError, match="fields"):
        PiafSource(run=_RUN, token="fixture", fields={"wind": "u10"})


def test_subsets_require_height_and_interpolation_neighbours() -> None:
    domain = Domain(48.4, 48.6, 2.4, 2.6)
    assert _subsets(_grid((900,), (10,)), domain, 900, 10) == [
        "time(900)",
        "lat(48.00000000,49.00000000)",
        "long(2.00000000,3.00000000)",
        "height(10)",
    ]
    with pytest.raises(DataUnavailableError, match="height"):
        _subsets(_grid((900,), (2,)), domain, 900, 10)
    with pytest.raises(DataUnavailableError, match="height"):
        _subsets(_grid((900,)), domain, 900, 10)


def test_latest_skips_run_missing_instant_left_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    older = _RUN - np.timedelta64(1, "h")
    monkeypatch.setattr(AromePISource, "published_runs", lambda self: [older, _RUN])
    calls = []

    def metadata(self: _CoverageReader, field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        calls.append((field, run))
        return "fixture", _grid((900,) if run == _RUN else (3600, 4500), (2,))

    monkeypatch.setattr(_CoverageReader, "metadata", metadata)
    with AromePISource.latest(
        _RUN + np.timedelta64(1, "s"), token="fixture", fields={"temperature": "t2m"}
    ) as source:
        assert source.run == older
        assert source.observed_at is not None
        assert [run for _, run in calls] == [_RUN, older]


def test_latest_period_needs_only_right_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PiafSource, "published_runs", lambda self: [_RUN])
    monkeypatch.setattr(_CoverageReader, "metadata", lambda *args: ("fixture", _grid((300,))))
    with PiafSource.latest(_RUN + np.timedelta64(1, "ns"), token="fixture") as source:
        assert source.run == _RUN
    with pytest.raises(DataUnavailableError):
        PiafSource.latest(_RUN, token="fixture")


def test_fetch_rejects_wrong_run_and_time_before_network() -> None:
    with PiafSource(run=_RUN, token="fixture") as source:
        domain = Domain(48, 49, 2, 3)
        for key in [CubeKey(_RUN, None), CubeKey(_RUN + np.timedelta64(1, "ns"), _RUN)]:
            with pytest.raises(ValueError, match="cube"):
                source.fetch(key, domain)
        with pytest.raises(DataUnavailableError, match="horizon"):
            source.fetch(CubeKey(_RUN + np.timedelta64(4, "h"), _RUN), domain)


def test_latest_skips_missing_height(monkeypatch: pytest.MonkeyPatch) -> None:
    older = _RUN - np.timedelta64(1, "h")
    monkeypatch.setattr(AromePISource, "published_runs", lambda self: [older, _RUN])

    def metadata(self: _CoverageReader, field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        return "fixture", _grid((0, 3600), (10,) if run == _RUN else (2,))

    monkeypatch.setattr(_CoverageReader, "metadata", metadata)
    with AromePISource.latest(_RUN, token="fixture", fields={"t": "t2m"}) as source:
        assert source.run == older
