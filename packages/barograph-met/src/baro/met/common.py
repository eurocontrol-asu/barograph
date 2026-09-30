"""Small helpers shared by the readers: a bounded mapping and durable file writes."""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from pathlib import Path

__all__ = ["Lru", "write_atomic"]


class Lru[K, V]:
    """Mapping keeping the ``maxsize`` most recently used entries (not thread-safe)."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._items: OrderedDict[K, V] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: K) -> V:
        self._items.move_to_end(key)
        return self._items[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def get(self, key: K) -> V | None:
        """The value of ``key`` (refreshing it), or ``None``."""
        return self[key] if key in self._items else None


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``path`` durably: temporary file, fsync, rename, fsync of the directory.

    Readers never see a partial file, and after a crash the file is either
    absent or complete (a rename alone may leave an empty file on some
    filesystems after a power loss).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    folder = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(folder)
    finally:
        os.close(folder)
