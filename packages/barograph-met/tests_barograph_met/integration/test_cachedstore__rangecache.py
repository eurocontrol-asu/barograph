"""Byte ranges kept on disk: only the missing sub-ranges go over the wire."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from baro.met.rangecache import CachedStore, RangeCache
from synthetic import MemoryStore

pytestmark = pytest.mark.integration

BLOB = bytes(range(256)) * 40  # 10 240 bytes


def _store(tmp_path: Path, cacheable: bool = True) -> tuple[CachedStore, MemoryStore]:
    inner = MemoryStore({"a/0.0": BLOB, ".zmetadata": b"{}"})
    cache = RangeCache(tmp_path)
    return CachedStore(inner, cache, "ns", lambda key: cacheable and key != ".zmetadata"), inner


def test_overlapping_reads_download_each_byte_once(tmp_path: Path) -> None:
    store, inner = _store(tmp_path)
    assert store.get("a/0.0", 1000, 3000) == BLOB[1000:3000]
    assert store.get("a/0.0", 2000, 5000) == BLOB[2000:5000]
    assert store.get("a/0.0", 500, 4000) == BLOB[500:4000]
    assert inner.stats.bytes == 2000 + 2000 + 500  # only the gaps
    assert store.cache.stats.snapshot()["hit_bytes"] == 1000 + 3000


def test_a_new_cache_instance_sees_what_another_one_stored(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    store.get("a/0.0", 0, 4096)
    again, inner = _store(tmp_path)
    assert again.get("a/0.0", 100, 200) == BLOB[100:200] and inner.stats.bytes == 0


def test_a_short_answer_marks_the_end_of_the_object(tmp_path: Path) -> None:
    store, inner = _store(tmp_path)
    assert store.get("a/0.0", 10_000, 14_096) == BLOB[10_000:]  # like a header prefix guess
    assert store.get("a/0.0", 10_000, 14_096) == BLOB[10_000:]
    assert inner.stats.requests == 1  # the missing tail is never requested again


def test_whole_objects_are_cached_and_serve_later_ranges(tmp_path: Path) -> None:
    store, inner = _store(tmp_path)
    assert store.get("a/0.0") == BLOB and store.get("a/0.0") == BLOB
    assert store.get("a/0.0", 7, 70) == BLOB[7:70]
    assert inner.stats.requests == 1


def test_mutable_keys_always_go_to_the_inner_store(tmp_path: Path) -> None:
    store, inner = _store(tmp_path)
    store.get(".zmetadata")
    store.get(".zmetadata")
    assert inner.stats.requests == 2 and store.size("a/0.0") == len(BLOB)
    store.get("a/0.0", 5, None)  # open-ended ranges are not cached either
    assert inner.stats.requests == 3 and store.location == inner.location


def test_lookup_and_put_serve_page_level_callers(tmp_path: Path) -> None:
    cache = RangeCache(tmp_path)
    assert cache.lookup("ns", "f.om", 0, 10) is None
    cache.put("ns", "f.om", 0, BLOB[:4096])
    assert cache.lookup("ns", "f.om", 100, 200) == BLOB[100:200]
    assert cache.lookup("ns", "f.om", 4000, 5000) is None  # partly missing
    assert (cache.object_dir("ns", "f.om") / "key.txt").read_text() == "f.om"


def test_claims_serialise_the_same_token(tmp_path: Path) -> None:
    cache = RangeCache(tmp_path)
    inside: list[int] = []
    overlaps: list[bool] = []

    def work(i: int) -> None:
        with cache.claim("hour-7"):
            overlaps.append(bool(inside))
            inside.append(i)
            inside.pop()

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, range(20)))
    assert overlaps == [False] * 20
