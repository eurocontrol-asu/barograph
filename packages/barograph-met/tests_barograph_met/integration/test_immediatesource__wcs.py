"""Surface sampling through real GRIB normalization and a local WCS transport."""

from __future__ import annotations

from pathlib import Path

import eccodes as ec
import httpx
import numpy as np
import pytest

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain
from baro.met.immediate import AromePISource, PiafSource
from baro.met.surface import SurfaceEnricher

pytestmark = pytest.mark.integration
RUN = np.datetime64("2026-10-07T12:00", "ns")
_NAMES = {
    "u10": "U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "v10": "V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "t2m": "TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "gusts": "WIND_SPEED_GUST_15MIN__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    "precipitation": "TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE",
    "rain": "TOTAL_WATER_PRECIPITATION__GROUND_OR_WATER_SURFACE",
}
_PARAMS = {
    "u10": (2, 2, 103, 10),
    "v10": (2, 3, 103, 10),
    "t2m": (0, 0, 103, 2),
    "gusts": (2, 22, 103, 10),
    "precipitation": (1, 52, 1, 0),
    "rain": (1, 65, 1, 0),
}
_PERIODS = {"gusts", "precipitation", "rain"}


def _description(field: str, step: int, horizon: int) -> str:
    height = _PARAMS[field][3]
    names = ["long", "lat", *(["height"] if height else []), "time"]
    labels = " ".join(names)
    units = "deg deg m s" if height else "deg deg s"
    axes = []
    for index, name in enumerate(names):
        vector = ["0"] * len(names)
        vector[index] = "0.25" if name in {"long", "lat"} else "1"
        coeff = (
            str(height)
            if name == "height"
            else " ".join(
                str(t) for t in range(step if field in _PERIODS else 0, horizon + 1, step)
            )
        )
        axes.append(
            f'<g:GeneralGridAxis><g:offsetVector axisLabels="{labels}" '
            f'uomLabels="{units}">{" ".join(vector)}</g:offsetVector>'
            f"<g:gridAxesSpanned>{name}</g:gridAxesSpanned>"
            f"<g:coefficients>{coeff}</g:coefficients></g:GeneralGridAxis>"
        )
    extra = f" {height}" if height else ""
    return (
        '<root xmlns:g="urn:g">'
        + "".join(axes)
        + (
            f'<g:EnvelopeWithTimePeriod axisLabels="{labels}">'
            f"<g:lowerCorner>1 48{extra}</g:lowerCorner>"
            f"<g:upperCorner>3 50{extra}</g:upperCorner>"
            "</g:EnvelopeWithTimePeriod></root>"
        )
    )


def _grib(field: str, lead: int, step: int, subsets: dict[str, str]) -> bytes:
    south, north = (float(v) for v in subsets["lat"].split(","))
    west, east = (float(v) for v in subsets["long"].split(","))
    lat = np.linspace(north, south, round((north - south) / 0.25) + 1)
    lon = np.linspace(west, east, round((east - west) / 0.25) + 1)
    category, number, level_type, level = _PARAMS[field]
    handle = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    keys = {
        "centre": 85,
        "Ni": len(lon),
        "Nj": len(lat),
        "latitudeOfFirstGridPointInDegrees": north,
        "latitudeOfLastGridPointInDegrees": south,
        "longitudeOfFirstGridPointInDegrees": west,
        "longitudeOfLastGridPointInDegrees": east,
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
        "dataDate": 20261007,
        "dataTime": 1200,
        "parameterCategory": category,
        "parameterNumber": number,
        "typeOfFirstFixedSurface": level_type,
        "scaledValueOfFirstFixedSurface": level,
        "scaleFactorOfFirstFixedSurface": 0,
        "stepType": "max" if field == "gusts" else "accum" if field in _PERIODS else "instant",
        "stepUnits": "s",
        "startStep": lead - step if field in _PERIODS else lead,
        "endStep": lead,
    }
    values = np.full((len(lat), len(lon)), lead / step, dtype=float)
    if field == "u10":
        values += 10 + 2 * (lat[:, None] - 49) + (lon[None, :] - 2)
    elif field == "v10":
        values[:] = 3
    elif field == "t2m":
        values += 280
    elif field == "gusts":
        values += 100
    try:
        for key, value in keys.items():
            ec.codes_set(handle, key, value)
        ec.codes_set_values(handle, values.ravel())
        return bytes(ec.codes_get_message(handle))
    finally:
        ec.codes_release(handle)


class _Service:
    def __init__(self, piaf: bool = False) -> None:
        self.step = 300 if piaf else 900
        self.horizon = 11700 if piaf else 21600
        names = (
            {
                "precipitation": _NAMES["precipitation"].replace(
                    "TOTAL_PRECIPITATION__", "TOTAL_PRECIPITATION_RATE__"
                )
            }
            if piaf
            else _NAMES
        )
        self.ids = {
            name
            + "___2026-10-07T12.00.00Z"
            + (f"_PT{self.step // 60}M" if field in _PERIODS else ""): field
            for field, name in names.items()
        }
        self.calls: list[tuple[str, int, dict[str, str]]] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "fixture-token"
        operation = request.url.path.rsplit("/", 1)[-1]
        if operation == "GetCapabilities":
            xml = "<root>" + "".join(f"<CoverageId>{i}</CoverageId>" for i in self.ids) + "</root>"
            return httpx.Response(200, text=xml)
        field = self.ids[request.url.params["coverageid"]]
        if operation == "DescribeCoverage":
            return httpx.Response(200, text=_description(field, self.step, self.horizon))
        assert operation == "GetCoverage"
        subsets = {
            s.split("(")[0]: s.split("(")[1][:-1] for s in request.url.params.get_list("subset")
        }
        raw_time = subsets["time"].strip('"')
        lead = (
            int(raw_time)
            if raw_time.isdigit()
            else int((np.datetime64(raw_time.removesuffix("Z")) - RUN) / np.timedelta64(1, "s"))
        )
        assert 0 <= lead <= self.horizon
        assert lead % self.step == 0
        assert field not in _PERIODS or lead >= self.step
        self.calls.append((field, lead, subsets))
        return httpx.Response(200, content=_grib(field, lead, self.step, subsets))


def test_arome_interpolation_periods_provenance_and_cache(tmp_path: Path) -> None:
    service = _Service()
    with (
        service.client,
        AromePISource(run=RUN, token="fixture-token", client=service.client) as source,
    ):
        sampler = SurfaceEnricher(source, cache_dir=tmp_path)
        times = RUN + np.array([450, 900, 1350, 21600], dtype="timedelta64[s]")
        result = sampler.enrich(times, 49.1, 2.1)
        assert result.valid.all()
        np.testing.assert_allclose(result["u10"], [10.8, 11.3, 11.8, 34.3], atol=1e-5)
        np.testing.assert_allclose(result["wind_speed"], np.hypot(result["u10"], 3))
        np.testing.assert_allclose(
            result["wind_direction"], np.degrees(np.arctan2(-result["u10"], -3)) % 360
        )
        np.testing.assert_allclose(result["precipitation"], [1, 1, 2, 24])
        np.testing.assert_allclose(result["gusts"], [101, 101, 102, 124])
        np.testing.assert_array_equal(result.run, np.full(4, RUN))
        np.testing.assert_array_equal(result.lead, times - RUN)
        np.testing.assert_array_equal(
            result.interval_end["precipitation"],
            RUN + np.array([900, 900, 1800, 21600], dtype="timedelta64[s]"),
        )
        count = len(service.calls)
        repeat = sampler.enrich(times, 49.1, 2.1)
        assert repeat.stats["bytes"] == 0
        assert len(service.calls) == count
        assert not any(field in _PERIODS and lead == 0 for field, lead, _ in service.calls)
        assert not any(lead > 21600 for _, lead, _ in service.calls)
        for _, _, subsets in service.calls:
            bounds = [float(v) for v in subsets["lat"].split(",")]
            assert bounds[1] - bounds[0] <= 0.5


def test_piaf_five_minute_hold_and_final_endpoint(tmp_path: Path) -> None:
    service = _Service(piaf=True)
    with (
        service.client,
        PiafSource(run=RUN, token="fixture-token", client=service.client) as source,
    ):
        sampler = SurfaceEnricher(source, cache_dir=tmp_path)
        result = sampler.enrich(
            RUN + np.array([1, 150, 300, 301, 11700], dtype="timedelta64[s]"), 49.1, 2.1
        )
        assert result.valid.all()
        np.testing.assert_array_equal(result["precipitation"], [1, 1, 1, 2, 39])
        np.testing.assert_array_equal(
            result.interval_end["precipitation"] - result.interval_start["precipitation"],
            np.timedelta64(300, "s"),
        )
        assert {lead for _, lead, _ in service.calls} == {300, 600, 11700}


def test_requested_aliases_share_one_response_and_fixed_run_validation() -> None:
    service = _Service()
    with (
        service.client,
        AromePISource(
            run=RUN,
            token="fixture-token",
            client=service.client,
            fields={"east": "u10", "copy": "u10", "rain": "rain"},
        ) as source,
    ):
        domain = Domain.from_points([49.1], [2.1], 0)
        key = CubeKey(RUN + np.timedelta64(900, "s"), RUN)
        cube = source.fetch(key, domain, fields=["east", "copy"])
        assert set(cube.fields) == {"east", "copy"}
        assert len(service.calls) == 1
        assert service.calls[0][0] == "u10"
        np.testing.assert_array_equal(cube.fields["east"], cube.fields["copy"])
        with pytest.raises(ValueError, match="belong"):
            source.fetch(CubeKey(key.valid, RUN - np.timedelta64(1, "h")), domain)
        with pytest.raises(DataUnavailableError, match="horizon"):
            source.fetch(CubeKey(RUN + np.timedelta64(7, "h"), RUN), domain)


def test_latest_checks_advertised_leads_without_downloading_fields() -> None:
    service = _Service()
    service.horizon = 1800  # The catalog is not proof that the full nominal run is ready.
    with service.client:
        with AromePISource.latest(
            RUN + np.timedelta64(1350, "s"), token="fixture-token", client=service.client
        ) as source:
            assert source.run == RUN
            assert source.observed_at is not None
            assert not service.calls
        with pytest.raises(DataUnavailableError, match="no published run"):
            AromePISource.latest(
                RUN + np.timedelta64(2250, "s"), token="fixture-token", client=service.client
            )
        with pytest.raises(DataUnavailableError, match="no published run"):
            AromePISource.latest(RUN, token="fixture-token", client=service.client)


def test_persistent_cache_survives_source_recreation(tmp_path: Path) -> None:
    service = _Service(piaf=True)
    valid = RUN + np.timedelta64(150, "s")
    with service.client:
        for attempt in range(2):
            with PiafSource(run=RUN, token="fixture-token", client=service.client) as source:
                sampler = SurfaceEnricher(source, cache_dir=tmp_path)
                result = sampler.enrich(valid, 49.1, 2.1)
                assert result.valid.all()
                # Metadata is refreshed per source, immutable GRIB bytes are reused.
                assert len(service.calls) == 1
                if attempt:
                    assert source.stats[0].bytes > 0
