"""Page cache of the Open-Meteo reader, with a mock HTTP transport (no network)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from baro.met.asyncfs import PagedAsyncFS
from baro.met.store import TransferStats

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
        fs = PagedAsyncFS("https://x.example", TransferStats(), 8, page_size=4096)
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


def _serve(blobs: list[bytes], calls: list[str], status: int = 206) -> httpx.MockTransport:
    """Serve ``blobs[-1]`` (the current version of the file), with an ETag per version."""

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        blob, tag = blobs[-1], f'"v{len(blobs)}"'
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(blob)), "etag": tag})
        if status != 206:
            return httpx.Response(status, content=blob)
        a, b = request.headers["range"].removeprefix("bytes=").split("-")
        return httpx.Response(206, content=blob[int(a) : int(b) + 1], headers={"etag": tag})

    return httpx.MockTransport(handle)


def _fs(transport: httpx.MockTransport) -> PagedAsyncFS:
    fs = PagedAsyncFS("https://x.example", TransferStats(), 8, page_size=4096)
    fs.client = httpx.AsyncClient(transport=transport)
    return fs


def test_a_server_ignoring_ranges_is_an_error() -> None:
    async def run() -> None:
        fs = _fs(_serve([BLOB], [], status=200))
        try:
            await fs._cat_file("f.om", 0, 100)
        finally:
            await fs.aclose()

    with pytest.raises(RuntimeError, match="range ignored"):
        asyncio.run(run())


def test_a_rewritten_file_is_read_again_never_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    blobs: list[bytes] = [BLOB]
    calls: list[str] = []
    newer = bytes(255 - x for x in BLOB)

    async def run() -> tuple[bytes, bytes, bytes]:
        fs = _fs(_serve(blobs, calls))
        first = await fs._cat_file("f.om", 0, 8000)
        blobs.append(newer)  # upstream rewrites the file
        cached = await fs._cat_file("f.om", 0, 8000)  # version trusted for a while
        monkeypatch.setattr("baro.met.asyncfs.REVALIDATE_S", 0.0)
        fresh = await fs._cat_file("f.om", 0, 8000)
        await fs.aclose()
        return first, cached, fresh

    first, cached, fresh = asyncio.run(run())
    assert first == cached == BLOB[:8000] and fresh == newer[:8000]


def test_immutable_files_are_never_revalidated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("baro.met.asyncfs.REVALIDATE_S", 0.0)

    async def run() -> None:
        fs = _fs(_serve([BLOB], calls))
        fs.immutable = lambda _path: True
        for _ in range(3):
            await fs._cat_file("f.om", 0, 100)
        await fs.aclose()

    asyncio.run(run())
    assert calls.count("HEAD") == 1 and calls.count("GET") == 1
