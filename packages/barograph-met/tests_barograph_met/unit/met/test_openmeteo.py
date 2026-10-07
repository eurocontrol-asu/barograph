"""Reassembling Open-Meteo tiles (no network)."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize("step_seconds,chunk_steps", [(3600, 210), (10800, 104)])
def test_catalog_native_indices(
    monkeypatch: pytest.MonkeyPatch, step_seconds: int, chunk_steps: int
) -> None:
    import json

    from baro.met.openmeteo import ArchiveCatalog
    from baro.met.store import TransferStats

    boundary = 1582 * chunk_steps * step_seconds
    meta = {
        "chunk_time_length": chunk_steps,
        "temporal_resolution_seconds": step_seconds,
        "data_end_time": boundary,
        "crs_wkt": "BBOX[-90,-180,90,179.75]",
    }
    monkeypatch.setattr(ArchiveCatalog, "_read_text", lambda *_: json.dumps(meta))
    monkeypatch.setattr(ArchiveCatalog, "_children", lambda *_args, **_kwargs: [])
    cat = ArchiveCatalog("unused", "model", [], TransferStats())
    assert cat.chunk_seconds == chunk_steps * step_seconds
    assert cat.end == np.datetime64(boundary - step_seconds, "s")
    assert divmod(cat.time_index(cat.end), cat.chunk_steps) == (1581, chunk_steps - 1)
    assert divmod(cat.time_index(np.datetime64(boundary, "s")), cat.chunk_steps) == (1582, 0)
    before = np.datetime64(boundary, "s").astype("datetime64[ns]") - np.timedelta64(1, "ns")
    assert divmod(cat.time_index(before), cat.chunk_steps) == (1581, chunk_steps - 1)
