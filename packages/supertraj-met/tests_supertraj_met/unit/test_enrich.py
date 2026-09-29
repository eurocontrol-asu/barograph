"""Enricher planning, prefetch and failure handling, with in-memory sources."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from supertraj_met import DataUnavailableError, Domain, Enricher, Flag
from supertraj_met.domain import Cube
from supertraj_met.enrich import UncoveredPointsWarning
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
