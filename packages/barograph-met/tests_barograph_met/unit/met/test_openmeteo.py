"""Reassembling Open-Meteo tiles (no network)."""

from __future__ import annotations

import numpy as np

from baro.met.openmeteo import gather_tiles


def test_gather_reassembles_tiles_across_boundaries() -> None:
    tp, hours = 4, 3
    full = np.arange(10 * 12 * hours, dtype=np.float32).reshape(10, 12, hours)
    tiles = {
        (r // tp, c // tp): full[r : r + tp, c : c + tp]
        for r in range(0, 10, tp)
        for c in range(0, 12, tp)
    }
    rows = np.array([9, 5, 3, 2])
    cols = np.array([10, 11, 0, 1, 7])  # wraps around like a global grid
    out = gather_tiles(tiles, rows, cols, tp, t_in=2)
    np.testing.assert_array_equal(out, full[np.ix_(rows, cols)][..., 2])
