"""Fixed-run immediate-source configuration and metadata selection contracts."""

from __future__ import annotations

from collections.abc import Callable
from xml.etree.ElementTree import Element, SubElement

import httpx
import numpy as np
import pytest

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain
from baro.met.immediate import AromePISource, PiafSource
from baro.met.wcs import WCSClient, coverage_run
from baro.met.wcs_catalog import CoverageGrid

_RUN = np.datetime64("2026-10-07T12:00", "ns")


def _grid(leads: tuple[int, ...], heights: tuple[float, ...] = ()) -> CoverageGrid:
    return CoverageGrid(np.array([49.0, 48.0]), np.array([2.0, 3.0]), leads, heights)


def _description(grid: CoverageGrid) -> Element:
    root = Element("root")
    labels = "long lat height time" if grid.heights else "long lat time"
    names = labels.split()
    for index, name in enumerate(names):
        axis = SubElement(root, "GeneralGridAxis")
        vector = ["0"] * len(names)
        vector[index] = "1"
        units = "deg deg m s" if grid.heights else "deg deg s"
        SubElement(axis, "offsetVector", axisLabels=labels, uomLabels=units).text = " ".join(vector)
        SubElement(axis, "gridAxesSpanned").text = name
        values = grid.heights if name == "height" else grid.lead_seconds
        SubElement(axis, "coefficients").text = " ".join(map(str, values))
    envelope = SubElement(root, "EnvelopeWithTimePeriod", axisLabels=labels)
    extra = f" {grid.heights[0]}" if grid.heights else ""
    SubElement(envelope, "lowerCorner").text = "2 48" + extra
    SubElement(envelope, "upperCorner").text = "3 49" + extra
    return root


def _advertise(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    metadata: Callable[[str, np.datetime64], tuple[str, CoverageGrid]],
    *,
    step: int = 900,
    cumulative: bool = False,
) -> None:
    name = {
        "t2m": "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
        "u10": "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
        "precipitation": "TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE",
    }[field]
    suffix = f"_PT{step // 60}M" if field == "precipitation" and not cumulative else ""
    ids = tuple(
        name + "___" + str(np.datetime_as_string(run, unit="s")).replace(":", ".") + "Z" + suffix
        for run in (_RUN - np.timedelta64(1, "h"), _RUN)
    )
    monkeypatch.setattr(WCSClient, "capabilities", lambda self: ids)

    def describe(self: WCSClient, identifier: str) -> Element:
        run = coverage_run(identifier)
        assert run is not None
        return _description(metadata(field, run)[1])

    monkeypatch.setattr(WCSClient, "describe", describe)


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


def test_subsets_require_height_and_interpolation_neighbours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _advertise(monkeypatch, "u10", lambda *args: ("", _grid((900,), (10,))))
    subsets_seen = []

    def coverage(self: WCSClient, identifier: str, **kwargs: object) -> bytes:
        subsets_seen.append(kwargs["subsets"])
        raise DataUnavailableError("fixture has no field")

    monkeypatch.setattr(WCSClient, "coverage", coverage)
    with AromePISource(run=_RUN, token="fixture", fields={"u": "u10"}) as source:
        with pytest.raises(DataUnavailableError, match="fixture"):
            source.fetch(
                CubeKey(_RUN + np.timedelta64(900, "s"), _RUN), Domain(48.4, 48.6, 2.4, 2.6)
            )
    assert subsets_seen == [
        ["time(900)", "lat(48.00000000,49.00000000)", "long(2.00000000,3.00000000)", "height(10)"]
    ]
    _advertise(monkeypatch, "u10", lambda *args: ("", _grid((900,), (2,))))
    with AromePISource(run=_RUN, token="fixture", fields={"u": "u10"}) as source:
        with pytest.raises(DataUnavailableError, match="height"):
            source.fetch(
                CubeKey(_RUN + np.timedelta64(900, "s"), _RUN), Domain(48.4, 48.6, 2.4, 2.6)
            )


def test_latest_skips_run_missing_instant_left_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    older = _RUN - np.timedelta64(1, "h")
    monkeypatch.setattr(AromePISource, "published_runs", lambda self: [older, _RUN])
    calls = []

    def metadata(field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        calls.append((field, run))
        return "fixture", _grid((900,) if run == _RUN else (3600, 4500), (2,))

    _advertise(monkeypatch, "t2m", metadata)
    with AromePISource.latest(
        _RUN + np.timedelta64(1, "s"), token="fixture", fields={"temperature": "t2m"}
    ) as source:
        assert source.run == older
        assert source.observed_at is not None
        assert [run for _, run in calls] == [_RUN, older]


def test_latest_period_needs_only_right_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PiafSource, "published_runs", lambda self: [_RUN])
    _advertise(monkeypatch, "precipitation", lambda *args: ("fixture", _grid((300,))), step=300)
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

    def metadata(field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        return "fixture", _grid((0, 3600), (10,) if run == _RUN else (2,))

    _advertise(monkeypatch, "t2m", metadata)
    with AromePISource.latest(_RUN, token="fixture", fields={"t": "t2m"}) as source:
        assert source.run == older


def test_latest_cumulative_product_needs_previous_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    older = _RUN - np.timedelta64(1, "h")
    monkeypatch.setattr(AromePISource, "published_runs", lambda self: [older, _RUN])

    def metadata(field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        return "cumulativeZ", _grid((1800,) if run == _RUN else (4500, 5400))

    _advertise(monkeypatch, "precipitation", metadata, cumulative=True)
    with AromePISource.latest(
        _RUN + np.timedelta64(30, "m"), token="fixture", fields={"p": "precipitation"}
    ) as source:
        assert source.run == older
