"""Open-Meteo pages of immutable chunks are kept on disk across readers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from baro.met.asyncfs import DiskPages, PagedAsyncFS
from baro.met.rangecache import RangeCache
from baro.met.store import TransferStats

pytestmark = pytest.mark.integration

BLOB = bytes(range(256)) * 64  # 16 KiB object


def _transport(calls: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(BLOB))})
        a, b = request.headers["range"].removeprefix("bytes=").split("-")
        return httpx.Response(206, content=BLOB[int(a) : int(b) + 1])

    return httpx.MockTransport(handle)


def _read(disk: DiskPages, calls: list[str], path: str) -> bytes:
    async def run() -> bytes:
        fs = PagedAsyncFS("https://x.example", TransferStats(), 8, page_size=4096)
        fs.client = httpx.AsyncClient(transport=_transport(calls))
        fs.disk = disk
        data = await fs._cat_file(path, 1000, 9000)
        await fs.aclose()
        return data

    return asyncio.run(run())


def test_a_second_reader_gets_immutable_pages_from_disk(tmp_path: Path) -> None:
    disk = DiskPages(RangeCache(tmp_path), "om", lambda path: path.endswith("chunk_1.om"))
    calls: list[str] = []
    assert _read(disk, calls, "chunk_1.om") == BLOB[1000:9000]
    assert calls.count("GET") == 3
    assert _read(disk, calls, "chunk_1.om") == BLOB[1000:9000]
    assert calls.count("GET") == 3  # all three pages came from disk


def test_recent_chunks_are_always_downloaded(tmp_path: Path) -> None:
    disk = DiskPages(RangeCache(tmp_path), "om", lambda path: path.endswith("chunk_1.om"))
    calls: list[str] = []
    _read(disk, calls, "chunk_9.om")
    _read(disk, calls, "chunk_9.om")
    assert calls.count("GET") == 6
