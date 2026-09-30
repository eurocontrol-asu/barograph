"""Open-Meteo source on a local mirror: both archive layouts, coverage, tile cache."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from baro.met import Enricher, Flag
from baro.met.enrich import UncoveredPointsWarning
from baro.met.openmeteo import OpenMeteoSource
from synthetic import EPOCH, OM_MODEL, truth, write_openmeteo_mirror

pytestmark = pytest.mark.integration
pytest.importorskip("omfiles")


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_openmeteo_mirror(tmp_path_factory.mktemp("openmeteo"))


def _flight(start_h: float, stop_h: float) -> tuple[np.ndarray, ...]:
    n = 120
    hours = np.linspace(start_h, stop_h, n)
    time = EPOCH + (hours * 3600).astype("timedelta64[s]")
    lat = np.linspace(42.0, 48.0, n)
    lon = np.linspace(-3.0, 8.0, n)
    p = np.linspace(22000.0, 45000.0, n)
    return time, lat, lon, p, hours


def test_catalog_reads_the_mirror(mirror: Path) -> None:
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror)) as src:
        assert src.levels == [200, 300, 500]
        assert src.chunk_hours == 24
        start, end = src.coverage()
        assert start == np.datetime64("2020-01-01T00", "h")
        assert end == np.datetime64("2020-01-02T23", "h")


def test_both_archive_layouts_recover_the_field(mirror: Path) -> None:
    time, lat, lon, p, hours = _flight(20.2, 27.8)  # crosses the chunk boundary at 24 h
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror), tile_deg=2.0) as src:
        res = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
    assert np.all(res.flags == Flag.OK)
    for name in ("u", "v", "t"):
        np.testing.assert_allclose(res[name], truth(name, lat, lon, p, hours), atol=0.02)


def test_tiles_are_reused_between_flights(mirror: Path) -> None:
    time, lat, lon, p, _ = _flight(2.0, 6.0)
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror)) as src:
        first = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
        second = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
    assert first.stats["bytes"] > 0 and second.stats["bytes"] == 0
    np.testing.assert_array_equal(first["u"], second["u"])


def test_points_after_the_archive_are_flagged(mirror: Path) -> None:
    time, lat, lon, p, _ = _flight(50.0, 51.0)
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror)) as src, warnings.catch_warnings():
        warnings.simplefilter("ignore", UncoveredPointsWarning)
        res = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
    assert np.all(res.flags == Flag.OUT_OF_COVERAGE)


def test_unknown_variable_is_rejected(mirror: Path) -> None:
    with pytest.raises(ValueError, match="unknown variables"):
        OpenMeteoSource(OM_MODEL, base_url=str(mirror), fields={"x": "nope"})
