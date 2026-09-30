"""Bounded mapping and durable writes."""

from __future__ import annotations

import pytest

from baro.met.common import Lru


def test_lru_keeps_the_most_recently_used_entries() -> None:
    lru: Lru[str, int | None] = Lru(2)
    lru["a"], lru["b"] = 1, None
    assert lru["a"] == 1  # "a" is now the most recent
    lru["c"] = 3
    assert "b" not in lru and lru.get("a") == 1 and lru.get("c") == 3 and len(lru) == 2
    assert lru.get("b") is None
    with pytest.raises(ValueError, match="positive"):
        Lru(0)
