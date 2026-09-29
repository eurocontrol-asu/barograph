"""Chained sources and coverage guardrails, with in-memory fake sources."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor

import numpy as np
import pytest
from synthetic import EPOCH

from supertraj_met import CubeKey, DataUnavailableError, Enricher, Flag
from supertraj_met.chain import SourceChain
from supertraj_met.domain import Cube, Domain
from supertraj_met.enrich import UncoveredPointsWarning
from supertraj_met.store import TransferStats

H = np.timedelta64(1, "h")


class FakeSource:
    """Uniform fields whose value identifies the source; fixed coverage."""

    time_step = H

    def __init__(self, name: str, value: float, start: int, stop: int, make: Callable[..., Cube]):
        self.source_id = name
        self.fields = {"u": "u", "v": "v", "t": "t"}
        self.value = value
        self.span = (EPOCH + start * H, EPOCH + stop * H)
        self.make = make
        self.fetched: list[CubeKey] = []

    @property
    def stats(self) -> list[TransferStats]:
        return [TransferStats()]

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        return self.span

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        h = np.datetime64(t0, "h")
        return CubeKey(h), CubeKey(h + H)

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        if not self.span[0] <= key.valid <= self.span[1]:
            raise DataUnavailableError(str(key))
        self.fetched.append(key)
        cube = self.make(hour=key.valid)
        cube.fields = {k: np.full_like(v, self.value) for k, v in cube.fields.items()}
        cube.source = self.source_id
        return cube


@pytest.fixture
def pair(make_cube: Callable[..., Cube]) -> tuple[FakeSource, FakeSource]:
    recent = FakeSource("recent", 2.0, 100, 200, make_cube)
    archive = FakeSource("archive", 1.0, 0, 150, make_cube)
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
