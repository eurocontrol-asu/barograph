"""Async byte access for ``omfiles``: HTTP pages with memory and disk caches, or a mirror.

``omfiles`` reads an archive file through an fsspec-like object exposing
``_cat_file``/``_size``. Its reads are tiny and many, so :class:`PagedAsyncFS`
serves them from aligned pages (one download per page, shared by concurrent
readers), kept in memory and, for files that no longer change, on disk
(:class:`DiskPages`). :class:`LocalAsyncFS` reads a local mirror directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx

from baro.met.limits import async_slot
from baro.met.rangecache import RangeCache, namespace_for
from baro.met.store import TransferStats

__all__ = ["DiskPages", "LocalAsyncFS", "PagedAsyncFS"]

_ATTEMPTS = 5
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_log = logging.getLogger("baro.met.http")


class PagedAsyncFS:
    """Async fsspec-like object for ``omfiles`` (``_cat_file``/``_size``), with a page cache.

    ``omfiles`` issues one range request per internal data chunk, and archive
    chunks are tiny (1 latitude row x 14 longitudes, a few hundred bytes).
    Requests are therefore served from aligned pages of ``page_size`` bytes
    (a page holds several whole latitude rows); concurrent requests for the
    same page share one download. This trades a few extra bytes for ~50x fewer
    round trips, which dominate the cost.
    """

    def __init__(
        self,
        base_url: str,
        stats: TransferStats,
        max_connections: int,
        page_size: int = 512 << 10,
        cache_mb: float = 256.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.stats = stats
        self.page_size = page_size
        self.client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max_connections),
        )
        self._pages: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._inflight: dict[tuple[str, int], asyncio.Future[bytes]] = {}
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._limit = int(cache_mb * 1e6)
        self.disk: DiskPages | None = None  # immutable pages also kept on disk

    async def _request(self, method: str, path: str, headers: dict[str, str]) -> httpx.Response:
        url = f"{self.base_url}/{path}"
        for attempt in range(_ATTEMPTS):
            try:
                t0 = time.perf_counter()
                async with async_slot(url):
                    resp = await self.client.request(method, url, headers=headers)
            except httpx.TransportError:
                if attempt == _ATTEMPTS - 1:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code in _RETRY_STATUS and attempt < _ATTEMPTS - 1:
                self.stats.throttle()
                _log.warning(
                    "HTTP %s from %s (attempt %d), backing off", resp.status_code, url, attempt + 1
                )
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code == HTTPStatus.NOT_FOUND:
                raise FileNotFoundError(path)
            resp.raise_for_status()
            self.stats.add(len(resp.content), time.perf_counter() - t0)
            return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _page(self, path: str, index: int) -> bytes:
        key = (path, index)
        if key in self._pages:
            self._pages.move_to_end(key)
            return self._pages[key]
        if key in self._inflight:
            return await self._inflight[key]
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            data = await self._load(path, index)
            self._remember(key, data)
            future.set_result(data)
            return data
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            del self._inflight[key]

    async def _load(self, path: str, index: int) -> bytes:
        """Page ``index`` of ``path``: from disk if kept there, else downloaded (and kept)."""
        size = await self._size(path)
        start = index * self.page_size
        stop = min(start + self.page_size, size)
        disk = self.disk if self.disk is not None and self.disk.cacheable(path) else None
        data = None if disk is None else disk.cache.lookup(disk.namespace, path, start, stop)
        if data is None:
            resp = await self._request("GET", path, {"Range": f"bytes={start}-{stop - 1}"})
            data = resp.content
            if disk is not None:
                disk.cache.put(disk.namespace, path, start, data)
        return data

    def _remember(self, key: tuple[str, int], data: bytes) -> None:
        """Keep a page in memory, evicting the least recently used beyond the limit."""
        self._pages[key] = data
        self._bytes += len(data)
        while self._bytes > self._limit and len(self._pages) > 1:
            _, old = self._pages.popitem(last=False)
            self._bytes -= len(old)

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **_: Any
    ) -> bytes:
        size = await self._size(path)
        start = 0 if start is None else start
        end = size if end is None else min(end, size)
        first, last = start // self.page_size, max(end - 1, start) // self.page_size
        pages = await asyncio.gather(*(self._page(path, i) for i in range(first, last + 1)))
        blob = b"".join(pages)
        offset = first * self.page_size
        return blob[start - offset : end - offset]

    async def _size(self, path: str, **_: Any) -> int:
        if path not in self._sizes:
            resp = await self._request("HEAD", path, {})
            self._sizes[path] = int(resp.headers["content-length"])
        return self._sizes[path]

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self.client.aclose()


@dataclass(frozen=True)
class DiskPages:
    """Where :class:`PagedAsyncFS` keeps the pages of immutable files."""

    cache: RangeCache
    namespace: str
    cacheable: Callable[[str], bool]

    @classmethod
    def at(cls, cache: RangeCache, location: str, cacheable: Callable[[str], bool]) -> DiskPages:
        """Pages of the files under ``location`` (a URL) whose path passes ``cacheable``."""
        return cls(cache, namespace_for(location), cacheable)


class LocalAsyncFS:
    """The same interface over a local mirror directory."""

    def __init__(self, root: str | Path, stats: TransferStats) -> None:
        self.root = Path(root)
        self.stats = stats

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **_: Any
    ) -> bytes:
        file = self.root / path
        if not file.is_file():
            raise FileNotFoundError(path)
        with file.open("rb") as fh:
            fh.seek(start or 0)
            data = fh.read(None if end is None else end - (start or 0))
        self.stats.add(len(data), 0.0)
        return data

    async def _size(self, path: str, **_: Any) -> int:
        file = self.root / path
        if not file.is_file():
            raise FileNotFoundError(path)
        return file.stat().st_size

    async def aclose(self) -> None:
        """Nothing to release."""
