from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baro.met import Enricher, Flag, GridSource, model_levels
from baro.met.sources import SurfacePressure
from baro.met.store import FileStore, ZarrStore
from synthetic import EPOCH, truth, write_stores

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_stores(tmp_path_factory.mktemp("zarr"), n_hours=4, missing_hours=(3,))


def _source(stores: tuple[Path, Path]) -> GridSource:
    ml, pl = stores
    return GridSource(
        "synthetic-ml",
        ZarrStore(FileStore(ml)),
        model_levels(),
        level_array="hybrid",
        surface_pressure=SurfacePressure(ZarrStore(FileStore(pl))),
    )


def _flights() -> tuple[np.ndarray, ...]:
    """Two flights between 00:10 and 01:50 with climbs and cruise."""
    n = 400
    minutes = np.linspace(10, 110, n)
    time = EPOCH + (minutes * 60).astype("timedelta64[s]")
    lat = np.concatenate([np.linspace(42.0, 48.0, n // 2), np.linspace(47.0, 43.5, n // 2)])
    lon = np.concatenate([np.linspace(21.0, 33.0, n // 2), np.linspace(35.0, 25.0, n // 2)])
    pressure = np.concatenate([np.linspace(85000.0, 22000.0, n // 2), np.full(n // 2, 25000.0)])
    return time, lat, lon, pressure, minutes / 60.0


def test_enrichment_matches_the_analytic_atmosphere(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    time, lat, lon, pressure, hours = _flights()
    result = Enricher(_source(stores), io_workers=4).enrich(time, lat, lon, pressure_pa=pressure)
    assert np.all(result.flags == Flag.OK)
    for name in ("u", "v", "t"):
        np.testing.assert_allclose(result[name], truth(name, lat, lon, pressure, hours), atol=2e-3)
    assert result.stats["cubes"] == 3 and result.stats["fetched"] == 3


def test_second_run_is_served_by_the_cache_without_transfer(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    time, lat, lon, pressure, _ = _flights()
    first = Enricher(_source(stores), cache_dir=tmp_path).enrich(
        time, lat, lon, pressure_pa=pressure
    )
    second = Enricher(_source(stores), cache_dir=tmp_path).enrich(
        time, lat, lon, pressure_pa=pressure
    )
    assert first.stats["bytes"] > 0
    assert second.stats["bytes"] == 0 and second.stats["cache_bytes"] > 0
    np.testing.assert_array_equal(first["u"], second["u"])


def test_flights_already_covered_by_an_earlier_call_are_free(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    time, lat, lon, pressure, _ = _flights()
    whole = Enricher(_source(stores), cache_dir=tmp_path).enrich(
        time, lat, lon, pressure_pa=pressure
    )
    for part in (slice(200, None), slice(0, 200)):  # each flight on its own, any order
        alone = Enricher(_source(stores), cache_dir=tmp_path).enrich(
            time[part], lat[part], lon[part], pressure_pa=pressure[part]
        )
        assert alone.stats["bytes"] == 0
        np.testing.assert_allclose(alone["t"], whole["t"][part], rtol=1e-12)


def test_unavailable_hour_is_flagged_not_extrapolated(stores: tuple[Path, Path]) -> None:
    time = np.array([EPOCH + np.timedelta64(150, "m"), EPOCH + np.timedelta64(60, "m")])
    result = Enricher(_source(stores)).enrich(
        time, [45.0, 45.0], [30.0, 30.0], altitude_ft=[35000.0, 35000.0]
    )
    assert list(result.flags) == [Flag.MISSING_DATA, Flag.OK]
    assert np.isnan(result["t"][0]) and np.isfinite(result["t"][1])


def test_concurrent_enrichers_download_each_byte_once(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    time, lat, lon, pressure, _ = _flights()

    def run(cache: Path) -> dict[str, int]:
        res = Enricher(_source(stores), cache_dir=cache).enrich(
            time, lat, lon, pressure_pa=pressure
        )
        return res.stats

    reference = run(tmp_path / "alone")["bytes"]
    with ThreadPoolExecutor(4) as ex:
        stats = list(ex.map(run, [tmp_path / "shared"] * 4))
    # every field byte once; only tiny metadata reads made before any claim may race:
    # the time-axis check and the 4 KiB header probe that sizes the tiling
    overhead = (stores[0] / "time" / "0").stat().st_size + 4096
    assert reference <= sum(s["bytes"] for s in stats) <= reference + 4 * overhead
