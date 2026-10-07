"""Open-Meteo source on a local mirror: both archive layouts, coverage, tile cache."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from baro.met import Enricher, Flag
from baro.met.enrich import UncoveredPointsWarning
from baro.met.openmeteo import OpenMeteoSource
from baro.met.rangecache import RangeCache
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


def test_tiles_and_immutable_chunks(mirror: Path, tmp_path: Path) -> None:
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror), tile_deg=2.0) as src:
        tiling = src.tiling(src.cube_keys(EPOCH)[0])
        assert tiling is not None and tiling.lon_step == pytest.approx(tiling.lat_step)
        src.use_cache(RangeCache(tmp_path))
        assert src.reader.disk is not None and src.reader.disk.cacheable("x/chunk_0.om")
        assert not src.reader.disk.cacheable("x/chunk_99999999.om")  # still being rewritten
        assert not src.reader.disk.cacheable("data/model/static/meta.json")


def test_tiles_of_recent_chunks_are_read_again_after_a_while(
    mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    time, lat, lon, p, _ = _flight(2.0, 6.0)
    with OpenMeteoSource(OM_MODEL, base_url=str(mirror)) as src:
        src.reader.final = lambda _path: False  # pretend the chunks are still rewritten
        Enricher(src).enrich(time, lat, lon, pressure_pa=p)
        assert Enricher(src).enrich(time, lat, lon, pressure_pa=p).stats["bytes"] == 0
        monkeypatch.setattr("baro.met.openmeteo.REVALIDATE_S", 0.0)
        assert Enricher(src).enrich(time, lat, lon, pressure_pa=p).stats["bytes"] > 0


@pytest.fixture(scope="module")
def three_hour_mirror(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_openmeteo_mirror(tmp_path_factory.mktemp("ifs"), step_hours=3)


@pytest.mark.parametrize("tile_deg", [0.5, 1.0, 2.0, 5.0])
def test_three_hour_interpolation_across_chunks(three_hour_mirror: Path, tile_deg: float) -> None:
    time, lat, lon, p, hours = _flight(44.2, 51.8)
    with OpenMeteoSource(OM_MODEL, base_url=str(three_hour_mirror), tile_deg=tile_deg) as src:
        assert src.time_step == np.timedelta64(3, "h")
        assert src.chunk_hours == 72
        assert src.catalog.chunk_steps == 24
        keys = src.cube_keys(EPOCH + np.timedelta64(71, "h"))
        assert keys[0].valid == EPOCH + np.timedelta64(69, "h")
        assert keys[1].valid == EPOCH + np.timedelta64(72, "h")
        assert all(k.run is None and k.lead is None for k in keys)
        first = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
        second = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
        assert second.stats["bytes"] == 0
    assert np.all(first.flags == Flag.OK)
    for name in ("u", "v", "t"):
        np.testing.assert_allclose(first[name], truth(name, lat, lon, p, hours), atol=0.02)
        np.testing.assert_array_equal(first[name], second[name])


def test_three_hour_coverage_endpoints(three_hour_mirror: Path) -> None:
    with OpenMeteoSource(OM_MODEL, base_url=str(three_hour_mirror)) as src:
        start, end = src.coverage()
        assert start == EPOCH - np.timedelta64(24, "h")
        assert end == start + np.timedelta64(141, "h")
        times = np.array(
            [
                start - np.timedelta64(1, "ns"),
                start,
                end - np.timedelta64(90, "m"),
                end,
                end + np.timedelta64(1, "ns"),
            ]
        )
        res = Enricher(src, on_uncovered="flag").enrich(
            times, np.full(5, 45.0), np.full(5, 2.0), pressure_pa=np.full(5, 30000.0)
        )
        np.testing.assert_array_equal(
            res.flags, [Flag.OUT_OF_COVERAGE, Flag.OK, Flag.OK, Flag.OK, Flag.OUT_OF_COVERAGE]
        )
        from baro.met.base import CubeKey
        from baro.met.domain import Domain

        with pytest.raises(ValueError, match="not aligned"):
            src.fetch(
                CubeKey(start + np.timedelta64(1, "h")),
                Domain.from_points([45.0], [2.0], 0.5),
                (30000.0, 30000.0),
            )


def test_three_hour_cache_finality(three_hour_mirror: Path) -> None:
    from baro.met.openmeteo import STITCH_WINDOW

    with OpenMeteoSource(OM_MODEL, base_url=str(three_hour_mirror)) as src:
        cutoff = np.datetime64("now", "s") - STITCH_WINDOW
        current = src.catalog.time_index(cutoff) // src.catalog.chunk_steps
        # The chunk containing the cutoff is still mutable, even though its
        # index interpreted as hours would put it decades in the past.
        assert not src._final_path(f"x/chunk_{current}.om")
        assert src._final_path(f"x/chunk_{current - 1}.om")


def test_three_hour_reads_only_needed_tiles_and_levels(three_hour_mirror: Path) -> None:
    from baro.met.domain import Domain

    with OpenMeteoSource(OM_MODEL, base_url=str(three_hour_mirror), tile_deg=1.0) as src:
        key = src.cube_keys(EPOCH)[0]
        cube = src.fetch(
            key, Domain.from_points([45.0], [2.0], 0.1), (25000.0, 25000.0), level_margin=0
        )
        assert set(cube.vertical.ids) == {200, 300}
        cached = list(src.reader._cache)
        assert {k[1] for k in cached} == {200, 300}
        assert {k[2] for k in cached} == {src.catalog.time_index(EPOCH) // 24}
        assert len({k[3] for k in cached}) <= 4
        assert cube.fields["u"].shape[1] < len(src.reader.grid()[0])
        assert cube.fields["u"].shape[2] < len(src.reader.grid()[1])
