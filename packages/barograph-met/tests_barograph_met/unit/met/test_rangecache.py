"""Range arithmetic of the byte-range cache (no disk access)."""

from __future__ import annotations

from pathlib import Path

from baro.met.rangecache import CacheStats, cover, namespace_for

A, B = Path("/c/0-100.bin"), Path("/c/150-300.bin")


def testcover_splits_a_request_into_cached_pieces_and_gaps() -> None:
    extents = [(0, 100, A), (150, 300, B)]
    assert cover(extents, 50, 200) == [(50, 100, A), (100, 150, None), (150, 200, B)]
    assert cover(extents, 300, 320) == [(300, 320, None)]
    assert cover([], 10, 20) == [(10, 20, None)]


def testcover_prefers_the_extent_reaching_furthest() -> None:
    long = Path("/c/0-500.bin")
    assert cover([(0, 100, A), (0, 500, long)], 10, 400) == [(10, 400, long)]


def test_namespaces_are_stable_readable_and_distinct() -> None:
    a = namespace_for("https://storage.googleapis.com/bucket/ar/store.zarr-v1")
    assert a == namespace_for("https://storage.googleapis.com/bucket/ar/store.zarr-v1/")
    assert a.startswith("store.zarr-v1-")
    assert a != namespace_for("https://mirror.example/ar/store.zarr-v1")


def test_stats_accumulate() -> None:
    stats = CacheStats()
    stats.add(hit=10)
    stats.add(stored=5, hit=1)
    assert stats.snapshot() == {"hit_bytes": 11, "stored_bytes": 5}
