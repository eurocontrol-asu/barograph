"""Page cache of the Open-Meteo reader, with a mock HTTP transport (no network)."""

from __future__ import annotations

import asyncio

import httpx
import numpy as np

from supertraj_met.openmeteo import _AsyncHttp, _gather
from supertraj_met.store import TransferStats

BLOB = bytes(range(256)) * 64  # 16 KiB object


def _handler(calls: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(BLOB))})
        a, b = request.headers["range"].removeprefix("bytes=").split("-")
        return httpx.Response(206, content=BLOB[int(a) : int(b) + 1])

    return httpx.MockTransport(handle)


def test_small_reads_are_served_from_shared_pages() -> None:
    calls: list[str] = []

    async def run() -> list[bytes]:
        fs = _AsyncHttp("https://x.example", TransferStats(), 8, page_size=4096)
        fs.client = httpx.AsyncClient(transport=_handler(calls))
        reads = [fs._cat_file("f.om", s, s + 100) for s in range(0, 8000, 200)]
        reads.append(fs._cat_file("f.om", 4000, 4200))  # straddles two pages
        out = await asyncio.gather(*reads)
        await fs.aclose()
        return list(out)

    out = asyncio.run(run())
    starts = [*range(0, 8000, 200), 4000]
    ends = [s + 100 for s in range(0, 8000, 200)] + [4200]
    assert all(o == BLOB[s:e] for o, s, e in zip(out, starts, ends, strict=True))
    assert calls.count("GET") == 2 and calls.count("HEAD") == 1  # 41 reads -> 2 pages


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
    out = _gather(tiles, rows, cols, tp, t_in=2)
    np.testing.assert_array_equal(out, full[np.ix_(rows, cols)][..., 2])
