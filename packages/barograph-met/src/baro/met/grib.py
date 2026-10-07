"""GRIB2 message index over range requests, and decoding of single messages.

A GRIB2 file is a plain concatenation of self-contained messages (one field:
variable x level x step), each starting with ``GRIB`` and announcing its own
length. When no inventory is published (ARPEGE open data), the index is built
by reading only the first KiB of every message. The walk is inherently
sequential (each length gives the next offset), so the file is split into
segments walked in parallel: each segment starts at the first ``GRIB``
signature found in a window at its nominal offset, and the chain of lengths
must land exactly on the next segment's start — otherwise the segment is
re-walked from the previous verified message, so a false signature inside
compressed data cannot corrupt the index.

Decoding needs ecCodes (``pip install 'barograph-met[grib]'``).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
import threading
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met.common import Lru, write_atomic
from baro.met.store import ByteStore

__all__ = [
    "GribFormatError",
    "GribIndex",
    "GribIndexStore",
    "GribMessage",
    "decode_message",
    "fetch_messages",
    "parse_header",
]

_STATISTICAL_TEMPLATE = 8
_STATISTICAL_SECTION_MIN = 58
_INDEX_VERSION = 2
_PREFIX = 512
_WINDOW = 256 << 10  # first window scanned for a message start; doubled until found
_MAX_MESSAGE = 1 << 30
_TIME_UNITS_H = {0: 1 / 60, 1: 1.0, 2: 24.0, 10: 3.0, 11: 6.0, 12: 12.0, 13: 1 / 3600}


class GribFormatError(ValueError):
    """Bytes that do not form a valid GRIB2 message header."""


@dataclass(frozen=True)
class GribMessage:
    """Where one GRIB2 message is and what it holds (variable, level, step)."""

    offset: int
    length: int
    discipline: int
    category: int
    number: int
    level_type: int
    level: float  # in the level type's unit (Pa for isobaric surfaces)
    step_h: float
    grid_template: int
    packing: int
    npoints: int
    start_step_h: float | None = None
    statistical_process: int | None = None

    @property
    def param(self) -> tuple[int, int, int]:
        """``(discipline, category, number)`` identifying the variable."""
        return (self.discipline, self.category, self.number)


# GRIB2 layout (WMO FM 92 GRIB edition 2)
_INDICATOR_LEN = 16  # section 0: "GRIB", reserved, discipline, edition, total length
_EDITION = 2
_SECTION_HEAD = 5  # every section starts with its length (4 bytes) and number (1 byte)
_LAST_SECTION = 8  # section 8 is the "7777" end marker
_GRID, _PRODUCT, _REPRESENTATION = 3, 4, 5
_MISSING_OCTET = 255


def _signed(raw: bytes) -> int:
    """GRIB2 signed integer: sign-and-magnitude (top bit is the sign), not two's complement."""
    value = int.from_bytes(raw, "big")
    top = 1 << (8 * len(raw) - 1)
    return -(value & (top - 1)) if value & top else value


def _grid_section(sec: bytes, info: dict[str, Any]) -> None:
    info["npoints"] = struct.unpack(">I", sec[6:10])[0]
    info["grid_template"] = struct.unpack(">H", sec[12:14])[0]


def _product_section(sec: bytes, info: dict[str, Any]) -> None:
    info["template4"] = struct.unpack(">H", sec[7:9])[0]
    info["category"], info["number"] = sec[9], sec[10]
    unit = sec[17]
    info["step_h"] = struct.unpack(">i", sec[18:22])[0] * _TIME_UNITS_H.get(unit, np.nan)
    if info["template4"] == _STATISTICAL_TEMPLATE:
        if len(sec) < _STATISTICAL_SECTION_MIN or sec[41] != 1:
            raise GribFormatError("only single-range statistical template 4.8 is supported")
        duration = struct.unpack(">I", sec[49:53])[0] * _TIME_UNITS_H.get(sec[48], np.nan)
        info["start_step_h"] = info["step_h"]
        info["step_h"] += duration
        info["statistical_process"] = sec[46]
    info["level_type"] = sec[22]
    scale = _signed(sec[23:24])
    value = _signed(sec[24:28])
    missing = sec[23] == _MISSING_OCTET
    info["level"] = float("nan") if missing else float(value) * 10.0 ** (-scale)


def _representation_section(sec: bytes, info: dict[str, Any]) -> None:
    info["packing"] = struct.unpack(">H", sec[9:11])[0]


# section number -> (minimum length read, parser)
_SECTIONS = {
    _GRID: (14, _grid_section),
    _PRODUCT: (28, _product_section),
    _REPRESENTATION: (11, _representation_section),
}


def _message_length(prefix: bytes, offset: int) -> int:
    """Total message length from section 0, after checking it is a GRIB2 message."""
    if len(prefix) < _INDICATOR_LEN or prefix[:4] != b"GRIB" or prefix[7] != _EDITION:
        raise GribFormatError(f"no GRIB2 message at offset {offset}")
    length = int(struct.unpack(">Q", prefix[8:_INDICATOR_LEN])[0])
    if not _INDICATOR_LEN < length < _MAX_MESSAGE:
        raise GribFormatError(f"implausible message length {length} at offset {offset}")
    return length


def _read_sections(prefix: bytes, offset: int, info: dict[str, Any]) -> None:
    """Fill ``info`` from sections 3 (grid), 4 (product) and 5 (representation)."""
    pos = _INDICATOR_LEN
    while pos + _SECTION_HEAD <= len(prefix):
        sec_len, num = struct.unpack(">IB", prefix[pos : pos + _SECTION_HEAD])
        if sec_len < _SECTION_HEAD or num > _LAST_SECTION:
            raise GribFormatError(f"corrupt section at offset {offset + pos}")
        sec = prefix[pos : pos + sec_len]
        if num in _SECTIONS and len(sec) >= _SECTIONS[num][0]:
            _SECTIONS[num][1](sec, info)
            if num == _REPRESENTATION:
                return
        pos += sec_len


def parse_header(prefix: bytes, offset: int) -> GribMessage:
    """Parse sections 0, 3, 4 and 5 of a GRIB2 message from its first bytes."""
    length = _message_length(prefix, offset)
    info: dict[str, Any] = {"offset": offset, "length": length, "discipline": prefix[6]}
    _read_sections(prefix, offset, info)
    missing = {"category", "level_type", "packing", "grid_template"} - set(info)
    if missing:
        raise GribFormatError(
            f"header at offset {offset} lacks {sorted(missing)} (prefix too short?)"
        )
    return GribMessage(
        offset=offset,
        length=length,
        discipline=info["discipline"],
        category=info["category"],
        number=info["number"],
        level_type=info["level_type"],
        level=info["level"],
        step_h=info["step_h"],
        grid_template=info["grid_template"],
        packing=info["packing"],
        npoints=info["npoints"],
        start_step_h=info.get("start_step_h"),
        statistical_process=info.get("statistical_process"),
    )


@dataclass
class GribIndex:
    """Messages of one GRIB2 file, looked up by (parameter, level, step)."""

    key: str
    size: int
    messages: list[GribMessage]

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        store: ByteStore,
        key: str,
        *,
        executor: Executor | None = None,
        segments: int = 32,
    ) -> GribIndex:
        """Walk the message headers of ``key`` (in parallel segments when an executor is given)."""
        size = store.size(key)
        if executor is None or segments <= 1 or size < 16 * _WINDOW:
            return cls(key, size, _walk(store, key, 0, size))
        bounds = [size * i // segments for i in range(1, segments)]
        try:
            starts = list(executor.map(lambda b: _find_start(store, key, b, size), bounds))
            # segment ends are the next segment's start; duplicates collapse
            edges = [*sorted({0, *starts}), size]
            spans = list(itertools.pairwise(edges))
            parts = list(executor.map(lambda ab: _walk(store, key, ab[0], ab[1]), spans))
        except GribFormatError:
            # a false signature broke the chaining: fall back to the exact sequential walk
            return cls(key, size, _walk(store, key, 0, size))
        return cls(key, size, [m for part in parts for m in part])

    def __post_init__(self) -> None:
        self._by_key = {(m.param, m.level_type, m.level, m.step_h): m for m in self.messages}

    # -- queries -----------------------------------------------------------------------------
    def find(
        self, param: tuple[int, int, int], level_type: int, level: float, step_h: float
    ) -> GribMessage | None:
        """The message of ``param`` at ``level`` and ``step_h``, if any."""
        return self._by_key.get((param, level_type, level, step_h))

    def levels(
        self, param: tuple[int, int, int], level_type: int, step_h: float | None = None
    ) -> list[float]:
        """Levels of ``param``, optionally only those published at ``step_h``."""
        return sorted(
            {
                m.level
                for m in self.messages
                if m.param == param
                and m.level_type == level_type
                and (step_h is None or m.step_h == step_h)
            }
        )

    def steps(self) -> list[float]:
        """Forecast steps (hours) present in the file."""
        return sorted({m.step_h for m in self.messages})

    # -- persistence ---------------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialise the index."""
        return json.dumps(
            {
                "version": _INDEX_VERSION,
                "key": self.key,
                "size": self.size,
                "messages": [asdict(m) for m in self.messages],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> GribIndex:
        """Rebuild an index serialised by :meth:`to_json`."""
        d = json.loads(text)
        if d.get("version") != _INDEX_VERSION:
            raise ValueError("obsolete GRIB index version")
        return cls(d["key"], d["size"], [GribMessage(**m) for m in d["messages"]])

    def save(self, path: Path) -> None:
        """Write the index atomically to ``path``."""
        write_atomic(path, self.to_json().encode())


def _walk(store: ByteStore, key: str, start: int, stop: int) -> list[GribMessage]:
    """Follow message lengths from ``start``; must land exactly on ``stop``."""
    out = []
    off = start
    while off < stop:
        prefix = store.get(key, off, min(off + _PREFIX, stop))
        msg = parse_header(prefix, off)
        out.append(msg)
        off += msg.length
    if off != stop:
        raise GribFormatError(f"message chain overshoots segment end ({off} != {stop})")
    return out


def _find_start(store: ByteStore, key: str, at: int, size: int) -> int:
    """Offset of the first verified message start at or after ``at``.

    The window starts small and doubles until it holds a verified start, so the
    cost stays close to one message size whatever the file layout.
    """
    window_size = _WINDOW
    while True:
        stop = min(at + window_size, size)
        window = store.get(key, at, stop)
        found = _verified_start(store, key, window, at, size)
        if found is not None:
            return found
        if stop >= size or window_size >= _MAX_MESSAGE:
            raise GribFormatError(f"no message start found after offset {at}")
        window_size *= 2


def _verified_start(store: ByteStore, key: str, window: bytes, at: int, size: int) -> int | None:
    pos = window.find(b"GRIB")
    while pos != -1:
        off = at + pos
        try:
            prefix = window[pos : pos + _PREFIX]
            if len(prefix) < _PREFIX and off + len(prefix) < size:
                prefix = store.get(key, off, min(off + _PREFIX, size))
            msg = parse_header(prefix, off)
            nxt = off + msg.length
            if nxt == size or (nxt < size and store.get(key, nxt, nxt + 4) == b"GRIB"):
                return off
        except GribFormatError:
            pass
        pos = window.find(b"GRIB", pos + 1)
    return None


def decode_message(
    buf: bytes,
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
    """Values ``(lat, lon)`` of a regular lat/lon message, lat descending, lon ascending.

    Longitudes are unwrapped (a 328E..42E grid gives 328..402).
    """
    try:
        import eccodes
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("decoding GRIB needs ecCodes: pip install 'barograph-met[grib]'") from exc
    h = eccodes.codes_new_from_message(buf)
    try:
        if eccodes.codes_get(h, "gridType") != "regular_ll":
            raise GribFormatError("only regular lat/lon grids are supported")
        g = {
            k: eccodes.codes_get(h, k)
            for k in (
                "Ni",
                "Nj",
                "latitudeOfFirstGridPointInDegrees",
                "longitudeOfFirstGridPointInDegrees",
                "iDirectionIncrementInDegrees",
                "jDirectionIncrementInDegrees",
                "iScansNegatively",
                "jScansPositively",
                "jPointsAreConsecutive",
                "bitmapPresent",
                "missingValue",
            )
        }
        values = np.asarray(eccodes.codes_get_values(h), dtype=np.float64)
    finally:
        eccodes.codes_release(h)
    if g["jPointsAreConsecutive"]:
        raise GribFormatError("column-major scanning is not supported")
    if g["bitmapPresent"]:
        values[values == g["missingValue"]] = np.nan
    ni, nj = int(g["Ni"]), int(g["Nj"])
    field = values.reshape(nj, ni)
    di, dj = float(g["iDirectionIncrementInDegrees"]), float(g["jDirectionIncrementInDegrees"])
    lat = float(g["latitudeOfFirstGridPointInDegrees"]) + np.arange(nj) * (
        dj if g["jScansPositively"] else -dj
    )
    lon = float(g["longitudeOfFirstGridPointInDegrees"]) + np.arange(ni) * (
        -di if g["iScansNegatively"] else di
    )
    if g["jScansPositively"]:
        field, lat = field[::-1], lat[::-1]
    if g["iScansNegatively"]:
        field, lon = field[:, ::-1], lon[::-1]
    return field.astype(np.float32), lat.astype(np.float64), lon.astype(np.float64)


def fetch_messages(
    store: ByteStore, key: str, messages: Sequence[GribMessage], executor: Executor | None
) -> list[bytes]:
    """Download the given messages (one range request each)."""

    def get(m: GribMessage) -> bytes:
        """Download one message."""
        return store.get(key, m.offset, m.offset + m.length)

    if executor is None or len(messages) <= 1:
        return [get(m) for m in messages]
    return list(executor.map(get, messages))


_MAX_INDEXES = 256  # message indexes kept in memory (the rest are re-read from disk)


class GribIndexStore:
    """Message indexes of the files of one store: memory, then ``directory``, else built.

    An index is built once per process even when several readers ask for it at
    the same time; a saved index that is damaged, stale (the file size changed)
    or for another key is rebuilt.
    """

    def __init__(self, store: ByteStore, directory: Path | None = None) -> None:
        self.store = store
        self.directory = directory
        self._indexes: Lru[str, GribIndex] = Lru(_MAX_INDEXES)
        self._building: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def get(self, key: str, executor: Executor | None = None) -> GribIndex:
        """Index of ``key``."""
        with self._lock:
            if key in self._indexes:
                return self._indexes[key]
            gate = self._building.setdefault(key, threading.Lock())
        with gate:
            with self._lock:
                if key in self._indexes:
                    return self._indexes[key]
            idx = self._saved(key)
            if idx is None:
                idx = GribIndex.build(self.store, key, executor=executor)
                if self.directory is not None:
                    idx.save(self._path(key))
            with self._lock:
                self._indexes[key] = idx
                self._building.pop(key, None)
        return idx

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def _saved(self, key: str) -> GribIndex | None:
        """The index saved for ``key``, if it is intact and still matches the file."""
        if self.directory is None:
            return None
        try:
            idx = GribIndex.from_json(self._path(key).read_text())
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError):  # damaged (e.g. by a crash): rebuild
            return None
        if idx.key != key or idx.size != self.store.size(key):
            return None  # stale or colliding entry
        return idx
