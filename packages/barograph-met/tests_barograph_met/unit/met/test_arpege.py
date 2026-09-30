"""ARPEGE file naming and run selection, over an in-memory store."""

from __future__ import annotations

import numpy as np
import pytest

from baro.met import ArpegeSource, DataUnavailableError
from synthetic import MemoryStore

RUN = np.datetime64("2026-09-29T00", "h")


def _source(runs: list[np.datetime64], **kw: object) -> ArpegeSource:
    probe = ArpegeSource(store=MemoryStore({}), listing_url=None)
    objects = {probe.file_key(r, 0): b"GRIB" for r in runs}
    return ArpegeSource(store=MemoryStore(objects), listing_url=None, **kw)  # type: ignore[arg-type]


def test_file_key_follows_the_open_data_layout() -> None:
    src = ArpegeSource(store=MemoryStore({}), listing_url=None)
    assert src.file_key(RUN, 13) == (
        "2026-09-29T00:00:00Z/arpege/01/IP1/arpege__01__IP1__013H024H__2026-09-29T00:00:00Z.grib2"
    )
    wide = ArpegeSource(resolution="025", store=MemoryStore({}), listing_url=None)
    assert "__025H048H__" in wide.file_key(RUN, 30)
    with pytest.raises(DataUnavailableError, match="beyond"):
        src.file_key(RUN, 200)


def test_latest_policy_uses_the_newest_published_run() -> None:
    six = RUN + np.timedelta64(6, "h")
    assert _source([RUN, six]).run_for(RUN + np.timedelta64(7, "h")) == six
    assert _source([RUN]).run_for(RUN + np.timedelta64(7, "h")) == RUN  # 06 not published


def test_daily_and_fixed_run_policies() -> None:
    t0 = RUN + np.timedelta64(9, "h")
    assert _source([RUN], run="daily").run_for(t0) == RUN
    fixed = RUN - np.timedelta64(12, "h")
    k0, k1 = _source([fixed], run=fixed).cube_keys(t0)
    assert k0.run == k1.run == fixed and k1.valid - k0.valid == np.timedelta64(1, "h")


def test_unknown_variables_and_resolution_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ARPEGE variables"):
        ArpegeSource(fields={"x": "nope"}, store=MemoryStore({}), listing_url=None)
    with pytest.raises(ValueError, match="unknown resolution"):
        ArpegeSource(resolution="05", store=MemoryStore({}), listing_url=None)  # type: ignore[arg-type]


def test_coverage_is_unknown_without_a_listing() -> None:
    assert ArpegeSource(store=MemoryStore({}), listing_url=None).coverage() is None


def test_a_file_not_yet_published_is_checked_again_later(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore({})
    src = ArpegeSource(store=store, listing_url=None)
    key = src.file_key(RUN, 0)
    assert not src._published(key)
    store.objects[key] = b"GRIB"  # the run is published a bit later
    assert not src._published(key)  # trusted for a while
    monkeypatch.setattr("baro.met.arpege._RECHECK_S", 0.0)
    assert src._published(key)
    del store.objects[key]
    assert src._published(key)  # a published file is never re-checked
