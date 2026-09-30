"""Enricher planning, prefetch and failure handling, with in-memory sources."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from baro.met import DataUnavailableError, Domain, Enricher, Flag, Tiling
from baro.met.domain import Cube
from baro.met.enrich import UncoveredPointsWarning
from synthetic import EPOCH, FakeSource


def _times(hours: list[float]) -> np.ndarray:
    return np.array([EPOCH + np.timedelta64(int(h * 60), "m") for h in hours])


@pytest.fixture
def source(make_cube: Callable[..., Cube]) -> FakeSource:
    return FakeSource("fake", 3.0, (0, 10), make_cube)


def test_each_cube_is_fetched_once_for_all_points(source: FakeSource) -> None:
    t = _times([0.2, 0.4, 0.6, 1.5, 1.7, 2.9])
    res = Enricher(source, prefetch=1).enrich(t, [45.0] * 6, [0.0] * 6, pressure_pa=[3e4] * 6)
    assert np.all(res.flags == Flag.OK) and np.allclose(res["u"], 3.0)
    assert sorted(k.valid for k in source.fetched) == [
        EPOCH + np.timedelta64(h, "h") for h in range(4)
    ]
    assert res.stats["fetched"] == 4 == res.stats["cubes"]


def test_missing_later_cube_flags_only_the_points_that_need_it(source: FakeSource) -> None:
    t = _times([10.0, 10.5])  # 10:00 exactly needs one cube; 10:30 needs 11:00 (absent)
    with pytest.warns(UncoveredPointsWarning, match="1 after"):
        res = Enricher(source).enrich(t, [45.0, 45.0], [0.0, 0.0], pressure_pa=[3e4, 3e4])
    assert list(res.flags) == [Flag.OK, Flag.OUT_OF_COVERAGE]


def test_invalid_inputs_are_flagged(source: FakeSource) -> None:
    t = _times([1.0, 1.0])
    res = Enricher(source).enrich(t, [45.0, np.nan], [0.0, 0.0], pressure_pa=[3e4, 3e4])
    assert list(res.flags) == [Flag.OK, Flag.INVALID_INPUT]


def test_global_mode_reads_whole_fields(source: FakeSource) -> None:
    enricher = Enricher(source, mode="global")
    assert enricher.domain == Domain.whole_globe() and enricher.level_margin > 137


def test_configuration_errors(source: FakeSource) -> None:
    with pytest.raises(ValueError, match="mode"):
        Enricher(source, mode="everything")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_uncovered"):
        Enricher(source, on_uncovered="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one"):
        Enricher(source).enrich(_times([1.0]), [45.0], [0.0])
    with pytest.raises(DataUnavailableError):
        Enricher(source, on_uncovered="raise").enrich(
            _times([20.0]), [45.0], [0.0], pressure_pa=[3e4]
        )


def test_distant_groups_of_one_hour_get_their_own_cubes(source: FakeSource) -> None:
    source.tiles = Tiling(lat0=-90.0, lat_step=2.0)  # 2-degree latitude bands
    t = _times([1.0, 1.0, 1.0, 1.0])
    lat = [41.0, 41.3, 48.8, 49.0]  # two groups, several empty bands apart
    res = Enricher(source).enrich(t, lat, [0.0] * 4, pressure_pa=[3e4] * 4)
    assert np.all(res.flags == Flag.OK) and res.stats["cubes"] == 2
    boxes = sorted((d.lat_min, d.lat_max) for d in source.domains)
    assert boxes == [(40.5, 41.8), (48.3, 49.5)]  # never the empty bands in between


def test_touching_groups_share_one_cube(source: FakeSource) -> None:
    source.tiles = Tiling(lat0=-90.0, lat_step=2.0)
    res = Enricher(source).enrich(
        _times([1.0, 1.0]), [41.0, 43.0], [0.0, 0.0], pressure_pa=[3e4, 3e4]
    )
    assert res.stats["cubes"] == 1


def test_a_fixed_domain_disables_the_split(source: FakeSource) -> None:
    source.tiles = Tiling(lat0=-90.0, lat_step=2.0)
    box = Domain(40.0, 50.0, -5.0, 10.0)
    res = Enricher(source, domain=box).enrich(
        _times([1.0, 1.0]), [41.0, 49.0], [0.0, 0.0], pressure_pa=[3e4, 3e4]
    )
    assert res.stats["cubes"] == 1 and source.domains == [box]


def test_the_cache_is_handed_to_the_source(source: FakeSource, tmp_path: Path) -> None:
    enricher = Enricher(source, cache_dir=tmp_path)
    assert source.cache is enricher.cache and source.cache is not None
