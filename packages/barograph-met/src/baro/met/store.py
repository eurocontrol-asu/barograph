"""Minimal zarr-v2 reader that fetches only the bytes a regional read needs.

The ARCO-ERA5 0.25 deg stores keep a *global* latitude x longitude field in each
chunk (``[1, n_levels, 721, 1440]``), so a small region never makes a chunk
smaller. Chunks are Blosc frames split into independently compressed blocks of
~512 KiB, i.e. bands of ~90 latitude rows of a single level. This reader
fetches the frame header, then only the block runs covering the requested
(level, latitude band) slabs, with HTTP range requests.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote

import httpx
import numcodecs
import numpy as np
from numpy.typing import NDArray

from baro.met.blosc import (
    BloscHeader,
    UnsupportedFrameError,
    decode_run,
    header_prefix_size,
    parse_header,
)
from baro.met.limits import slot

__all__ = [
    "ArrayMeta",
    "ByteStore",
    "FileStore",
    "HttpStore",
    "MissingChunkError",
    "TransferStats",
    "ZarrStore",
]

_PREFIX_GUESS = 4096
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_log = logging.getLogger("baro.met.http")


class MissingChunkError(KeyError):
    """The chunk does not exist in the store (reads as fill value)."""


@dataclass
class TransferStats:
    """Thread-safe counters of what went over the wire."""

    requests: int = 0
    bytes: int = 0
    seconds: float = 0.0
    throttled: int = 0  # 429/5xx answers that were retried (server pushing back)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, nbytes: int, seconds: float) -> None:
        """Account for one request of ``nbytes`` bytes taking ``seconds``."""
        with self._lock:
            self.requests += 1
            self.bytes += nbytes
            self.seconds += seconds

    def throttle(self) -> None:
        """Account for one 429/5xx answer."""
        with self._lock:
            self.throttled += 1

    def snapshot(self) -> dict[str, float]:
        """Current counters as a dict."""
        with self._lock:
            return {
                "requests": self.requests,
                "bytes": self.bytes,
                "seconds": self.seconds,
                "throttled": self.throttled,
            }


class ByteStore(Protocol):
    """Key/value byte access with optional ``[start, stop)`` ranges."""

    stats: TransferStats

    def get(self, key: str, start: int | None = None, stop: int | None = None) -> bytes:
        """Object bytes, or the ``[start, stop)`` range of them."""
        ...

    def size(self, key: str) -> int:
        """Object size in bytes; raises :class:`MissingChunkError` if absent."""
        ...


class HttpStore:
    """Anonymous HTTP(S) access (honours ``HTTPS_PROXY``), with retries."""

    def __init__(  # noqa: PLR0913 - keyword-only HTTP client settings
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        retries: int = 4,
        backoff: float = 0.5,
        timeout: float = 120.0,
        max_connections: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        )
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True, limits=limits)
        self.retries = retries
        self.backoff = backoff
        self.stats = TransferStats()
        self._base = self.base_url  # becomes the redirect target once one is observed

    def _url(self, key: str) -> str:
        return f"{self._base}/{key}"

    def _learn_redirect(self, resp: httpx.Response, key: str) -> None:
        """Remember a permanent host redirect (e.g. data.gouv.fr -> object storage).

        Only a redirect that preserves the object path is learnt, so every later
        request goes straight to the target instead of paying one extra round trip.
        """
        final = unquote(str(resp.url).split("?", 1)[0])  # targets may percent-encode ':'
        if resp.history and final.endswith("/" + key):
            self._base = final[: -len(key) - 1]

    def get(self, key: str, start: int | None = None, stop: int | None = None) -> bytes:
        """Object bytes, or the ``[start, stop)`` range of them."""
        headers = {}
        if start is not None:
            headers["Range"] = f"bytes={start}-" + ("" if stop is None else str(stop - 1))
        t0 = time.perf_counter()
        resp = self._request("GET", key, headers)
        if start is not None and resp.status_code != HTTPStatus.PARTIAL_CONTENT:
            raise RuntimeError(f"server ignored the range request for {key}")
        self.stats.add(len(resp.content), time.perf_counter() - t0)
        return resp.content

    def size(self, key: str) -> int:
        """Object size in bytes (HEAD request)."""
        return int(self._request("HEAD", key, {}).headers["content-length"])

    def _request(self, method: str, key: str, headers: dict[str, str]) -> httpx.Response:
        """One request with retries: transport errors and 429/5xx back off, 404 raises."""
        for attempt in range(self.retries + 1):
            url = self._url(key)
            try:
                with slot(url):
                    resp = self.client.request(method, url, headers=headers)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * 2**attempt)
                continue
            if resp.status_code == HTTPStatus.NOT_FOUND:
                raise MissingChunkError(key)
            if resp.status_code in _RETRY_STATUS and attempt < self.retries:
                self._pushed_back(resp, url, attempt)
                continue
            resp.raise_for_status()
            self._learn_redirect(resp, key)
            return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    def _pushed_back(self, resp: httpx.Response, url: str, attempt: int) -> None:
        """Record a 429/5xx, honour ``Retry-After`` when given, then back off."""
        self.stats.throttle()
        _log.warning(
            "HTTP %s from %s (attempt %d), backing off", resp.status_code, url, attempt + 1
        )
        retry_after = resp.headers.get("retry-after", "")
        delay = float(retry_after) if retry_after.isdigit() else self.backoff * 2**attempt
        time.sleep(min(delay, 60.0))


class FileStore:
    """Local directory store (tests, mirrored stores)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.stats = TransferStats()

    def get(self, key: str, start: int | None = None, stop: int | None = None) -> bytes:
        """Object bytes, or the ``[start, stop)`` range of them."""
        path = self.root / key
        t0 = time.perf_counter()
        try:
            with path.open("rb") as fh:
                if start is not None:
                    fh.seek(start)
                    data = fh.read(None if stop is None else stop - start)
                else:
                    data = fh.read()
        except FileNotFoundError as exc:
            raise MissingChunkError(key) from exc
        self.stats.add(len(data), time.perf_counter() - t0)
        return data

    def size(self, key: str) -> int:
        """Object size in bytes."""
        try:
            return (self.root / key).stat().st_size
        except FileNotFoundError as exc:
            raise MissingChunkError(key) from exc


@dataclass(frozen=True)
class ArrayMeta:
    """Metadata of one zarr-v2 array."""

    name: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype[Any]
    compressor: dict[str, Any] | None
    fill_value: float | None
    separator: str
    dims: tuple[str, ...]
    attrs: dict[str, Any]

    def chunk_key(self, index: Sequence[int]) -> str:
        """Store key of the chunk at ``index``."""
        return f"{self.name}/" + self.separator.join(str(i) for i in index)

    @property
    def fill(self) -> float:
        """Fill value as a float (NaN when unset)."""
        return float("nan") if self.fill_value is None else float(self.fill_value)


def _parse_fill(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        return float(value.replace("Infinity", "inf"))
    return float(value)


class ZarrStore:
    """Consolidated zarr-v2 store with region reads over a ``ByteStore``."""

    def __init__(self, store: ByteStore, *, full_read_ratio: float = 0.8) -> None:
        self.store = store
        self.full_read_ratio = full_read_ratio
        meta = json.loads(store.get(".zmetadata"))["metadata"]
        self._meta = meta
        self._headers: dict[str, BloscHeader | None] = {}
        self._lock = threading.Lock()

    @property
    def attrs(self) -> dict[str, Any]:
        """Root attributes of the store."""
        return dict(self._meta.get(".zattrs", {}))

    def array(self, name: str) -> ArrayMeta:
        """Metadata of array ``name``."""
        try:
            za = self._meta[f"{name}/.zarray"]
        except KeyError as exc:
            raise KeyError(f"array {name!r} not in store") from exc
        attrs = self._meta.get(f"{name}/.zattrs", {})
        if za.get("filters"):
            raise NotImplementedError(f"{name}: zarr filters are not supported")
        if za.get("order", "C") != "C":
            raise NotImplementedError(f"{name}: only C order is supported")
        return ArrayMeta(
            name=name,
            shape=tuple(za["shape"]),
            chunks=tuple(za["chunks"]),
            dtype=np.dtype(za["dtype"]),
            compressor=za.get("compressor"),
            fill_value=_parse_fill(za.get("fill_value")),
            separator=za.get("dimension_separator", "."),
            dims=tuple(attrs.get("_ARRAY_DIMENSIONS", ())),
            attrs=attrs,
        )

    # -- whole-array reads (coordinates) ------------------------------------------------
    def read_1d(self, name: str, start: int = 0, stop: int | None = None) -> NDArray[Any]:
        """Values ``[start, stop)`` of a 1-D array (e.g. a coordinate)."""
        meta = self.array(name)
        if len(meta.shape) != 1:
            raise ValueError(f"{name} is not 1-D")
        stop = meta.shape[0] if stop is None else stop
        size = meta.chunks[0]
        parts = []
        for c in range(start // size, (stop - 1) // size + 1):
            chunk = self._decode_full(meta, (c,)).reshape(-1)
            lo, hi = max(start - c * size, 0), min(stop - c * size, size)
            parts.append(chunk[lo:hi])
        return np.concatenate(parts)

    def _decode_full(self, meta: ArrayMeta, index: tuple[int, ...]) -> NDArray[Any]:
        try:
            raw = self.store.get(meta.chunk_key(index))
        except MissingChunkError:
            return np.full(meta.chunks, meta.fill, dtype=meta.dtype)
        if meta.compressor is not None:
            raw = bytes(numcodecs.get_codec(meta.compressor).decode(raw))
        expected = int(np.prod(meta.chunks)) * meta.dtype.itemsize
        if len(raw) != expected:
            raise ValueError(
                f"{meta.chunk_key(index)}: {len(raw)} bytes decoded, expected {expected} "
                "(zarr v2 edge chunks must be padded to the full chunk shape)"
            )
        return np.frombuffer(raw, dtype=meta.dtype).reshape(meta.chunks)

    # -- regional reads -------------------------------------------------------------------
    def read_rows(
        self,
        name: str,
        lead: Sequence[Sequence[int]],
        rows: slice,
        *,
        executor: Executor | None = None,
    ) -> NDArray[np.float32]:
        """Read ``array[*lead_i, rows, :]`` for every leading index tuple in ``lead``.

        For an array ``(time, [level,] lat, lon)`` each ``lead_i`` gives the global
        ``(time, level)`` (or ``(time,)``) indices. The two trailing dimensions
        must be stored unchunked (true for ARCO-ERA5 0.25 deg). Every header and
        block-run request is independent and runs on ``executor`` if given.
        Returns ``float32`` of shape ``(len(lead), n_rows, n_lon)``.
        """
        meta = self.array(name)
        if meta.chunks[-2:] != meta.shape[-2:]:
            raise NotImplementedError(f"{name}: lat/lon must be unchunked")
        r0, r1, _ = rows.indices(meta.shape[-2])
        by_chunk = _plan_slabs(meta, lead, r0, r1)
        out = np.empty((len(lead), r1 - r0, meta.shape[-1]), dtype=np.float32)

        chunks = list(by_chunk)
        headers = _map(executor, lambda c: self._safe_header(meta, meta.chunk_key(c)), chunks)
        tasks: list[_Task] = []
        for cidx, header in zip(chunks, headers, strict=True):
            if header is _MISSING:
                for slab in by_chunk[cidx]:
                    out[slab.pos] = meta.fill
            else:
                tasks.extend(self._tasks_for(cidx, by_chunk[cidx], header))
        _map(executor, lambda t: self._execute(meta, t, by_chunk[t.chunk], out), tasks)
        return out

    def _tasks_for(self, cidx: ChunkIndex, slabs: list[_Slab], header: object) -> list[_Task]:
        if not isinstance(header, BloscHeader):
            return [_Task(cidx, None, None)]
        runs = _block_runs(header, [(sl.start, sl.stop) for sl in slabs])
        spans = [header.compressed_span(f, la) for f, la in runs]
        if sum(c1 - c0 for c0, c1 in spans) > self.full_read_ratio * header.cbytes:
            return [_Task(cidx, None, None)]
        return [_Task(cidx, header, run) for run in runs]

    def _execute(
        self, meta: ArrayMeta, task: _Task, slabs: list[_Slab], out: NDArray[np.float32]
    ) -> None:
        if task.header is None or task.run is None:
            data = self._decode_full(meta, task.chunk).tobytes()
            u0 = 0
        else:
            first, last = task.run
            c0, c1 = task.header.compressed_span(first, last)
            payload = self.store.get(meta.chunk_key(task.chunk), c0, c1)
            data = decode_run(task.header, first, last, payload)
            u0 = task.header.block_span(first)[0]
        for slab in slabs:
            if u0 <= slab.start and slab.stop <= u0 + len(data):
                buf = data[slab.start - u0 : slab.stop - u0]
                out[slab.pos] = np.frombuffer(buf, dtype=meta.dtype).reshape(out.shape[1:])

    def _safe_header(self, meta: ArrayMeta, key: str) -> BloscHeader | object | None:
        try:
            return self._header(meta, key)
        except MissingChunkError:
            return _MISSING

    def _header(self, meta: ArrayMeta, key: str) -> BloscHeader | None:
        with self._lock:
            if key in self._headers:
                return self._headers[key]
        header: BloscHeader | None
        if meta.compressor is None or meta.compressor.get("id") != "blosc":
            header = None
        else:
            prefix = self.store.get(key, 0, _PREFIX_GUESS)
            try:
                need = header_prefix_size(prefix[:16])
                if need > len(prefix):
                    prefix = self.store.get(key, 0, need)
                header = parse_header(prefix)
            except UnsupportedFrameError:
                header = None
        with self._lock:
            self._headers[key] = header
        return header


_MISSING = object()
type ChunkIndex = tuple[int, ...]


@dataclass(frozen=True)
class _Task:
    chunk: ChunkIndex
    header: BloscHeader | None  # None: read and decode the whole chunk
    run: tuple[int, int] | None


def _plan_slabs(
    meta: ArrayMeta, lead: Sequence[Sequence[int]], r0: int, r1: int
) -> dict[ChunkIndex, list[_Slab]]:
    """Group the requested ``(lead, rows)`` slabs by chunk, with their byte spans."""
    nlead = len(meta.shape) - 2
    nlat, nlon = meta.shape[-2:]
    item = meta.dtype.itemsize
    by_chunk: dict[ChunkIndex, list[_Slab]] = {}
    for pos, raw in enumerate(lead):
        li = tuple(int(i) for i in raw)
        if len(li) != nlead:
            raise ValueError(f"{meta.name}: expected {nlead} leading indices, got {li}")
        cidx = (*(i // c for i, c in zip(li, meta.chunks[:nlead], strict=True)), 0, 0)
        within = tuple(i % c for i, c in zip(li, meta.chunks[:nlead], strict=True))
        flat = int(np.ravel_multi_index(within, meta.chunks[:nlead])) if within else 0
        base = flat * nlat * nlon
        slab = _Slab(pos, (base + r0 * nlon) * item, (base + r1 * nlon) * item)
        by_chunk.setdefault(cidx, []).append(slab)
    return by_chunk


@dataclass(frozen=True)
class _Slab:
    pos: int  # output position
    start: int  # uncompressed byte offset in the chunk
    stop: int


def _map[T, R](executor: Executor | None, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
    if executor is None or len(items) <= 1:
        return [fn(i) for i in items]
    return list(executor.map(fn, items))


def _block_runs(header: BloscHeader, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge the (decodable) block ranges of byte spans into disjoint runs.

    Each run fully contains every span it covers, so that a span is always
    decoded from a single run.
    """
    ranges = sorted(header.blocks_for(s, e) for s, e in spans)
    runs: list[list[int]] = []
    for first, last in ranges:
        if runs and first <= runs[-1][1] + 1:
            runs[-1][1] = max(runs[-1][1], last)
        else:
            runs.append([first, last])
    return [(f, la) for f, la in runs]
