from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("eccodes")

from supertraj_met import ArpegeSource, Enricher, Flag
from supertraj_met.grib import GribIndex
from supertraj_met.store import FileStore
from synthetic import ARPEGE_RUN, EPOCH, arpege_truth, write_arpege_mirror

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mirror(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_arpege_mirror(tmp_path_factory.mktemp("arpege"))


def _file(mirror: Path) -> str:
    stamp = str(ARPEGE_RUN) + ":00:00Z"
    return f"{stamp}/arpege/01/IP1/arpege__01__IP1__000H012H__{stamp}.grib2"


def test_parallel_index_equals_the_sequential_walk(mirror: Path) -> None:
    store = FileStore(mirror)
    seq = GribIndex.build(store, _file(mirror))
    with ThreadPoolExecutor(8) as ex:
        par = GribIndex.build(store, _file(mirror), executor=ex, segments=8)
    assert [m.offset for m in par.messages] == [m.offset for m in seq.messages]
    assert len(seq.messages) == 3 * 5 * 13
    assert par.levels((0, 0, 0), 100) == [20000.0, 25000.0, 30000.0, 40000.0, 50000.0]


def _flight() -> tuple[np.ndarray, ...]:
    n = 200
    minutes = np.linspace(65, 235, n)  # 01:05 -> 03:55 UTC, crosses Greenwich
    time = ARPEGE_RUN + (minutes * 60).astype("timedelta64[s]")
    lat = np.linspace(48.0, 44.0, n)
    lon = np.linspace(-3.0, 4.0, n)
    p = np.linspace(21000.0, 45000.0, n)
    hours = (time - EPOCH) / np.timedelta64(1, "h")
    return time, lat, lon, p, hours.astype(np.float64)


def test_enrichment_recovers_the_analytic_forecast(mirror: Path, tmp_path: Path) -> None:
    time, lat, lon, p, hours = _flight()
    src = ArpegeSource(resolution="01", base_url=str(mirror), index_dir=tmp_path)
    res = Enricher(src).enrich(time, lat, lon, pressure_pa=p)
    assert np.all(res.flags == Flag.OK)
    for name in ("u", "v", "t"):
        np.testing.assert_allclose(res[name], arpege_truth(name, lat, lon, p, hours), atol=2e-3)
    assert res.stats["cubes"] == 4  # 01h, 02h, 03h, 04h from the 00 UTC run


def test_latest_policy_falls_back_to_the_published_run(mirror: Path) -> None:
    src = ArpegeSource(resolution="01", base_url=str(mirror))  # the 06 UTC run is not published
    t0 = ARPEGE_RUN + np.timedelta64(7, "h")
    k0, k1 = src.cube_keys(t0)
    assert k0.run == k1.run == ARPEGE_RUN


def test_unpublished_data_is_flagged_missing(mirror: Path) -> None:
    src = ArpegeSource(resolution="01", base_url=str(mirror), run="daily")
    t = np.array([ARPEGE_RUN + np.timedelta64(30, "h")])  # next day's 00 run absent
    res = Enricher(src).enrich(t, [45.0], [2.0], pressure_pa=[30000.0])
    assert res.flags[0] == Flag.MISSING_DATA and np.isnan(res["t"][0])


def test_points_outside_the_regional_grid_are_flagged(mirror: Path) -> None:
    src = ArpegeSource(resolution="01", base_url=str(mirror))
    t = np.array([ARPEGE_RUN + np.timedelta64(90, "m")] * 2)
    res = Enricher(src).enrich(t, [45.0, 10.0], [2.0, 2.0], pressure_pa=[30000.0, 30000.0])
    assert list(res.flags) == [Flag.OK, Flag.OUTSIDE_DOMAIN]


def test_cached_index_avoids_rewalking_the_file(mirror: Path, tmp_path: Path) -> None:
    time, lat, lon, p, _ = _flight()
    first = Enricher(ArpegeSource(resolution="01", base_url=str(mirror), index_dir=tmp_path))
    r1 = first.enrich(time, lat, lon, pressure_pa=p)
    second = Enricher(ArpegeSource(resolution="01", base_url=str(mirror), index_dir=tmp_path))
    r2 = second.enrich(time, lat, lon, pressure_pa=p)
    messages_in_file = 3 * 5 * 13
    assert r1.stats["requests"] - r2.stats["requests"] >= messages_in_file  # no header walk
    assert r2.stats["requests"] <= r2.stats["cubes"] * 3 * 5 + 2  # only data messages (+ HEADs)
    np.testing.assert_array_equal(r1["u"], r2["u"])
