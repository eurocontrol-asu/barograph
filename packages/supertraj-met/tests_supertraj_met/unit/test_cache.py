"""Cache layout and queries (paths only, no disk access)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from supertraj_met.base import CubeKey
from supertraj_met.cache import CubeCache, CubeQuery
from supertraj_met.domain import Domain

HOUR = np.datetime64("2026-09-29T07", "h")


def test_slots_are_organised_by_source_date_hour_and_run() -> None:
    cache = CubeCache("/cache")
    assert cache._key_dir("src", CubeKey(HOUR)) == Path("/cache/src/2026/09/29/07")
    run = np.datetime64("2026-09-29T06", "h")
    assert cache._key_dir("src", CubeKey(HOUR, run)) == Path(
        "/cache/src/2026/09/29/07/run-2026092906"
    )


def test_query_defaults_to_one_spare_level() -> None:
    query = CubeQuery(CubeKey(HOUR), Domain(40, 50, 0, 10), (2e4, 3e4))
    assert query.level_margin == 1


def test_home_directory_is_expanded() -> None:
    assert "~" not in str(CubeCache("~/cubes").root)
