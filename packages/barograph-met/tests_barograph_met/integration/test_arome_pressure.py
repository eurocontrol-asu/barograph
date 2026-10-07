"""AROME archive contracts on a small local mirror; no network or QAR data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import synthetic
from baro.met import Enricher, Flag
from baro.met.base import CubeKey
from baro.met.domain import Domain
from baro.met.openmeteo import OpenMeteoSource

pytestmark = pytest.mark.integration
pytest.importorskip("omfiles")

MODEL = "meteofrance_arome_france0025"
LEVELS = [
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    275,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    925,
    950,
    1000,
]


@pytest.fixture
def arome_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(synthetic, "OM_MODEL", MODEL)
    monkeypatch.setattr(synthetic, "OM_CHUNK_HOURS", 108)
    monkeypatch.setattr(synthetic, "OM_LEVELS", LEVELS)
    return synthetic.write_openmeteo_mirror(tmp_path)


def test_arome_native_chunks_and_selective_levels(arome_mirror: Path) -> None:
    with OpenMeteoSource(MODEL, base_url=str(arome_mirror), tile_deg=0.5) as src:
        assert src.time_step == np.timedelta64(1, "h")
        assert src.chunk_hours == 108
        assert src.levels == LEVELS
        start, _ = src.coverage()
        times = start + np.array([107 * 3600 + 1800, 108 * 3600 + 1800]).astype("timedelta64[s]")
        res = Enricher(src, level_margin=0).enrich(
            times, [45.0, 45.0], [2.0, 2.0], pressure_pa=[26000.0, 26000.0]
        )
        assert np.all(res.flags == Flag.OK)
        hours = (times - synthetic.EPOCH) / np.timedelta64(1, "h")
        for name in ("u", "v", "t"):
            np.testing.assert_allclose(
                res[name],
                synthetic.truth(
                    name, np.array([45.0] * 2), np.array([2.0] * 2), np.array([26000.0] * 2), hours
                ),
                atol=0.02,
            )
        assert {key[1] for key in src.reader._cache} == {250, 275}
        assert len({key[2] for key in src.reader._cache}) == 2
        assert (
            Enricher(src, level_margin=0)
            .enrich(times, [45.0, 45.0], [2.0, 2.0], pressure_pa=[26000.0, 26000.0])
            .stats["bytes"]
            == 0
        )


def test_regional_outside_points_are_flagged(arome_mirror: Path) -> None:
    with OpenMeteoSource(MODEL, base_url=str(arome_mirror), tile_deg=0.5) as src:
        start, _ = src.coverage()
        res = Enricher(src).enrich(
            np.full(3, start),
            [55.0, 45.0, 45.0],
            [2.0, 20.0, 2.0],
            pressure_pa=np.full(3, 30000.0),
        )
        np.testing.assert_array_equal(
            res.flags, [Flag.OUTSIDE_DOMAIN, Flag.OUTSIDE_DOMAIN, Flag.OK]
        )
        for values in res.fields.values():
            assert np.all(np.isnan(values[:2]))


def test_missing_variable_level_uses_common_levels(arome_mirror: Path) -> None:
    import shutil

    shutil.rmtree(arome_mirror / "data" / MODEL / "temperature_275hPa")
    with OpenMeteoSource(MODEL, base_url=str(arome_mirror)) as src:
        assert 275 not in src.levels
        start, _ = src.coverage()
        cube = src.fetch(
            CubeKey(start), Domain(44.0, 46.0, 1.0, 3.0), (26000.0, 26000.0), level_margin=0
        )
        np.testing.assert_array_equal(cube.vertical.ids, [250, 300])


def test_missing_chunk_is_unavailable(arome_mirror: Path) -> None:
    with OpenMeteoSource(MODEL, base_url=str(arome_mirror)) as src:
        start, _ = src.coverage()
        index = src.catalog.time_index(start) // src.catalog.chunk_steps
        missing = src.catalog.path("temperature", 300, index)
        (arome_mirror / missing).unlink()
        result = Enricher(src, level_margin=0).enrich([start], [45.0], [2.0], pressure_pa=[26000.0])
        # 260 hPa selects 250/275, so the unneeded missing 300-hPa file has no cost.
        assert result.flags[0] == Flag.OK
        missing = src.catalog.path("temperature", 275, index)
        (arome_mirror / missing).unlink()
    with OpenMeteoSource(MODEL, base_url=str(arome_mirror)) as src:
        result = Enricher(src, level_margin=0).enrich([start], [45.0], [2.0], pressure_pa=[26000.0])
        assert result.flags[0] == Flag.MISSING_DATA
