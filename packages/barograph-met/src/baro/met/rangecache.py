"""On-disk cache of the byte ranges read from remote objects.

The unit of reuse is what the source itself stores: a run of Blosc blocks of
an ARCO-ERA5 chunk, a GRIB message of an ARPEGE file, a page of an Open-Meteo
``.om`` file. Whatever the shape of the requests (one flight, a fleet, many
calls in any order, several processes), a byte is downloaded once and served
from disk afterwards; only the missing sub-ranges of a request go over the
network.

Layout: ``root/<namespace>/<xx>/<digest>/`` per object, holding one file per
cached extent (``<start>-<stop>.bin``), a ``size`` file once the end of the
object is known, and ``key.txt`` (the object key, for humans). Extent files are
written atomically (temporary file + rename) and never modified, so readers
never see a partial extent.

Missing ranges are fetched under ``flock`` locks on the 1 MiB windows they
cover (lock files live in the object's folder), taken in order, across
threads and processes; once the locks are held the cache is checked again. A
range read by several readers at once is therefore downloaded once, while
disjoint reads of one object (e.g. the levels of an ERA5 chunk) stay parallel.

Only immutable objects may be cached: each source says which of its keys are
(e.g. ERA5 chunks once final, ARPEGE files of a published run, Open-Meteo
chunks older than the stitching window).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from baro.met.store import ByteStore, TransferStats

__all__ = ["CacheStats", "CachedStore", "RangeCache", "cover", "namespace_for"]

_EXTENT = re.compile(r"(\d+)-(\d+)\.bin")
_WINDOW = 1 << 20  # lock granularity for concurrent fetches

type Fetch = Callable[[int, int], bytes]


@dataclass
class CacheStats:
    """Thread-safe counters of what the cache served and stored."""

    hit_bytes: int = 0
    stored_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, hit: int = 0, stored: int = 0) -> None:
        """Account for bytes served from disk and bytes written to it."""
        with self._lock:
            self.hit_bytes += hit
            self.stored_bytes += stored

    def snapshot(self) -> dict[str, int]:
        """Current counters as a dict."""
        with self._lock:
            return {"hit_bytes": self.hit_bytes, "stored_bytes": self.stored_bytes}


class RangeCache:
    """Immutable byte-range cache rooted at ``root`` (see the module docstring)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.stats = CacheStats()

    # -- layout ----------------------------------------------------------------------------
    def object_dir(self, namespace: str, key: str) -> Path:
        """Folder of one object of one namespace."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / namespace / digest[:2] / digest[2:24]

    # -- reads -----------------------------------------------------------------------------
    def get(self, namespace: str, key: str, start: int, stop: int, fetch: Fetch) -> bytes:
        """Bytes ``[start, stop)`` of ``key``, fetching only the missing sub-ranges.

        ``fetch(a, b)`` downloads ``[a, b)``; a shorter answer marks the end of
        the object, and the result is then truncated there like a range read.
        """
        folder = self.object_dir(namespace, key)
        data = self._cached(folder, start, stop)
        if data is not None:
            return data
        windows = range(start // _WINDOW, (max(stop, start + 1) - 1) // _WINDOW + 1)
        with _locked(folder, [f"w{w}" for w in windows]):
            data = self._cached(folder, start, stop)  # fetched meanwhile by another reader
            if data is not None:
                return data
            return self._fill(folder, key, start, stop, fetch)

    def _cached(self, folder: Path, start: int, stop: int) -> bytes | None:
        """Bytes ``[start, stop)`` if they are all on disk, else ``None``."""
        size = _read_size(folder)
        if size is not None:
            stop = min(stop, size)
        if stop <= start:
            return b""
        pieces = cover(_extents(folder), start, stop)
        if any(path is None for _, _, path in pieces):
            return None
        self.stats.add(hit=stop - start)
        return b"".join(_read_slice(p, a, b) for a, b, p in pieces if p is not None)

    def _fill(self, folder: Path, key: str, start: int, stop: int, fetch: Fetch) -> bytes:
        """Serve ``[start, stop)`` from disk and ``fetch``, keeping what was fetched."""
        size = _read_size(folder)
        if size is not None:
            stop = min(stop, size)
        pieces: list[bytes] = []
        hit = 0
        for a, b, path in cover(_extents(folder), start, stop):
            if path is not None:
                pieces.append(_read_slice(path, a, b))
                hit += b - a
                continue
            data = fetch(a, b)
            self._store(folder, key, a, data)
            pieces.append(data)
            if len(data) < b - a:  # the object ends inside this gap
                _write_atomic(folder / "size", str(a + len(data)).encode())
                break
        self.stats.add(hit=hit)
        return b"".join(pieces)

    def lookup(self, namespace: str, key: str, start: int, stop: int) -> bytes | None:
        """Bytes ``[start, stop)`` of ``key`` if they are all on disk, else ``None``.

        For callers that deduplicate their own concurrent fetches (the
        Open-Meteo page reader); they keep what they fetched with :meth:`put`.
        """
        return self._cached(self.object_dir(namespace, key), start, stop)

    def put(self, namespace: str, key: str, start: int, data: bytes) -> None:
        """Keep ``data``, the bytes of ``key`` from ``start`` on."""
        self._store(self.object_dir(namespace, key), key, start, data)

    def get_full(self, namespace: str, key: str, fetch_all: Callable[[], bytes]) -> bytes:
        """The whole object ``key``, fetched once."""
        folder = self.object_dir(namespace, key)
        data = self._whole(folder)
        if data is not None:
            return data
        with _locked(folder, ["full"]):
            data = self._whole(folder)
            if data is not None:
                return data
            data = fetch_all()
            self._store(folder, key, 0, data)
            _write_atomic(folder / "size", str(len(data)).encode())
            return data

    def _whole(self, folder: Path) -> bytes | None:
        """The whole object if its size is known and all its bytes are on disk."""
        size = _read_size(folder)
        return None if size is None else self._cached(folder, 0, size)

    # -- writes ------------------------------------------------------------------------------
    def _store(self, folder: Path, key: str, start: int, data: bytes) -> None:
        if not data:
            return
        folder.mkdir(parents=True, exist_ok=True)
        label = folder / "key.txt"
        if not label.exists():
            _write_atomic(label, key.encode())
        _write_atomic(folder / f"{start}-{start + len(data)}.bin", data)
        self.stats.add(stored=len(data))


class CachedStore:
    """A :class:`~baro.met.store.ByteStore` whose immutable keys go through a :class:`RangeCache`.

    ``cacheable(key)`` says which keys never change; the others (store
    metadata, the growing time axis, objects still being rewritten) are always
    read from ``inner``. ``stats`` are the inner store's: they count only the
    bytes that actually went over the network.
    """

    def __init__(
        self,
        inner: ByteStore,
        cache: RangeCache,
        namespace: str,
        cacheable: Callable[[str], bool],
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.cacheable = cacheable
        self.stats: TransferStats = inner.stats
        self.location = inner.location

    def get(self, key: str, start: int | None = None, stop: int | None = None) -> bytes:
        """Object bytes, or the ``[start, stop)`` range of them."""
        if not self.cacheable(key) or (start is not None and stop is None):
            return self.inner.get(key, start, stop)
        if start is None:
            return self.cache.get_full(self.namespace, key, lambda: self.inner.get(key))
        return self.cache.get(
            self.namespace, key, start, stop or 0, lambda a, b: self.inner.get(key, a, b)
        )

    def size(self, key: str) -> int:
        """Object size in bytes (never cached: a cheap metadata request)."""
        return self.inner.size(key)


def namespace_for(location: str) -> str:
    """Stable, filesystem-safe namespace for a store location (URL or directory)."""
    location = location.rstrip("/")
    tail = re.sub(r"[^A-Za-z0-9._-]+", "_", location.rstrip("/").rsplit("/", 1)[-1])[:48]
    return f"{tail}-{hashlib.sha256(location.encode()).hexdigest()[:12]}"


# -- helpers --------------------------------------------------------------------------------
@contextmanager
def _locked(folder: Path, names: list[str]) -> Iterator[None]:
    """Hold ``flock`` locks on ``folder/.lock-<name>`` for every name, taken in order."""
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for name in sorted(names):
            fh = stack.enter_context(open(folder / f".lock-{name}", "a+b"))
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            stack.callback(fcntl.flock, fh.fileno(), fcntl.LOCK_UN)
        yield


def _extents(folder: Path) -> list[tuple[int, int, Path]]:
    """Cached extents of an object, sorted by start."""
    try:
        entries = list(os.scandir(folder))
    except FileNotFoundError:
        return []
    found = []
    for entry in entries:
        m = _EXTENT.fullmatch(entry.name)
        if m is not None:
            found.append((int(m.group(1)), int(m.group(2)), Path(entry.path)))
    return sorted(found)


def cover(
    extents: list[tuple[int, int, Path]], start: int, stop: int
) -> list[tuple[int, int, Path | None]]:
    """Split ``[start, stop)`` into cached pieces (with their file) and gaps (``None``)."""
    out: list[tuple[int, int, Path | None]] = []
    pos = start
    while pos < stop:
        best: tuple[int, int, Path] | None = None
        nxt = stop
        for a, b, path in extents:
            if a <= pos < b and (best is None or b > best[1]):
                best = (a, b, path)
            elif pos < a < nxt:
                nxt = a
        if best is not None:
            end = min(best[1], stop)
            out.append((pos, end, best[2]))
            pos = end
        else:
            out.append((pos, nxt, None))
            pos = nxt
    return out


def _read_slice(path: Path, a: int, b: int) -> bytes:
    """Bytes ``[a, b)`` of the object, from the extent file ``path``."""
    first = int(_EXTENT.fullmatch(path.name).group(1))  # type: ignore[union-attr]
    with path.open("rb") as fh:
        fh.seek(a - first)
        return fh.read(b - a)


def _read_size(folder: Path) -> int | None:
    try:
        return int((folder / "size").read_text())
    except (FileNotFoundError, ValueError):
        return None


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``path`` via a temporary file and a rename (readers never see a partial file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
