"""Chained sources and coverage guardrails, with in-memory fake sources."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from baro.met import DataUnavailableError, Enricher, Flag
from baro.met.base import CubeKey
from baro.met.chain import SourceChain
from baro.met.domain import Cube, Domain
from baro.met.enrich import UncoveredPointsWarning
from synthetic import EPOCH, FakeSource

H = np.timedelta64(1, "h")


@pytest.fixture
def pair(make_cube: Callable[..., Cube]) -> tuple[FakeSource, FakeSource]:
    recent = FakeSource("recent", 2.0, (100, 200), make_cube)
    archive = FakeSource("archive", 1.0, (0, 150), make_cube)
    return recent, archive


def _at(hours: list[float]) -> np.ndarray:
    return np.array([EPOCH + np.timedelta64(int(h * 60), "m") for h in hours])


def test_first_source_serving_the_interval_wins(pair: tuple[FakeSource, FakeSource]) -> None:
    recent, archive = pair
    t = _at([10.5, 120.5, 149.5])
    res = Enricher(SourceChain(recent, archive)).enrich(
        t, [45.0] * 3, [0.0] * 3, pressure_pa=[3e4] * 3
    )
    np.testing.assert_allclose(res["u"], [1.0, 2.0, 2.0])
    assert all(k.source == "recent" for k in recent.fetched) or not recent.fetched


def test_an_interval_is_never_split_between_sources(pair: tuple[FakeSource, FakeSource]) -> None:
    recent, archive = pair
    # 99.5 h: its interval [99, 100] is not fully inside "recent" (starts at 100) -> archive
    res = Enricher(SourceChain(recent, archive)).enrich(
        _at([99.5]), [45.0], [0.0], pressure_pa=[3e4]
    )
    assert res["u"][0] == 1.0
    assert all(k.valid <= EPOCH + 100 * H for k in archive.fetched)


def test_points_outside_every_source_are_flagged_with_a_warning(
    pair: tuple[FakeSource, FakeSource],
) -> None:
    recent, archive = pair
    t = _at([-5.0, 10.5, 300.0])
    with pytest.warns(UncoveredPointsWarning, match=r"1 before, 1 after"):
        res = Enricher(SourceChain(recent, archive)).enrich(
            t, [45.0] * 3, [0.0] * 3, pressure_pa=[3e4] * 3
        )
    assert list(res.flags) == [Flag.OUT_OF_COVERAGE, Flag.OK, Flag.OUT_OF_COVERAGE]


def test_raise_mode_refuses_uncovered_points(pair: tuple[FakeSource, FakeSource]) -> None:
    recent, _ = pair
    with pytest.raises(DataUnavailableError, match="outside the period"):
        Enricher(recent, on_uncovered="raise").enrich(_at([5.0]), [45.0], [0.0], pressure_pa=[3e4])


def test_chain_rejects_incompatible_sources(pair: tuple[FakeSource, FakeSource]) -> None:
    recent, archive = pair
    archive.fields = {"u": "u"}
    with pytest.raises(ValueError, match="same output fields"):
        SourceChain(recent, archive)


def test_chain_requires_consistent_distinct_sources(pair: tuple[FakeSource, FakeSource]) -> None:
    recent, archive = pair
    with pytest.raises(ValueError, match="at least one"):
        SourceChain()
    with pytest.raises(ValueError, match="distinct ids"):
        SourceChain(recent, recent)
    archive.time_step = np.timedelta64(3, "h")
    with pytest.raises(ValueError, match="time step"):
        SourceChain(recent, archive)


def test_unrouted_keys_are_refused(pair: tuple[FakeSource, FakeSource]) -> None:
    chain = SourceChain(*pair)
    with pytest.raises(DataUnavailableError, match="not routed"):
        chain.fetch(CubeKey(EPOCH), Domain(40, 50, 0, 10), (2e4, 3e4))
