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

Decoding needs ecCodes (``pip install 'supertraj-met[grib]'``).
"""

from __future__ import annotations

import itertools
import json
import struct
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from supertraj_met.store import ByteStore

__all__ = [
    "GribFormatError",
    "GribIndex",
    "GribMessage",
    "decode_message",
    "fetch_messages",
    "parse_header",
]

_PREFIX = 512
_WINDOW = 256 << 10  # first window scanned for a message start; doubled until found
_MAX_MESSAGE = 1 << 30
_TIME_UNITS_H = {0: 1 / 60, 1: 1.0, 2: 24.0, 10: 3.0, 11: 6.0, 12: 12.0, 13: 1 / 3600}


class GribFormatError(ValueError):
    """Bytes that do not form a valid GRIB2 message header."""


@dataclass(frozen=True)
class GribMessage:
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

    @property
    def param(self) -> tuple[int, int, int]:
        return (self.discipline, self.category, self.number)


def parse_header(prefix: bytes, offset: int) -> GribMessage:
    """Parse sections 0, 3, 4 and 5 of a GRIB2 message from its first bytes."""
    if len(prefix) < 16 or prefix[:4] != b"GRIB" or prefix[7] != 2:
        raise GribFormatError(f"no GRIB2 message at offset {offset}")
    length = struct.unpack(">Q", prefix[8:16])[0]
    if not 16 < length < _MAX_MESSAGE:
        raise GribFormatError(f"implausible message length {length} at offset {offset}")
    info: dict[str, Any] = {"offset": offset, "length": length, "discipline": prefix[6]}
    pos = 16
    while pos + 5 <= len(prefix):
        sec_len, num = struct.unpack(">IB", prefix[pos : pos + 5])
        if sec_len < 5 or num > 8:
            raise GribFormatError(f"corrupt section at offset {offset + pos}")
        sec = prefix[pos : pos + sec_len]
        if num == 3 and len(sec) >= 14:
            info["npoints"] = struct.unpack(">I", sec[6:10])[0]
            info["grid_template"] = struct.unpack(">H", sec[12:14])[0]
        elif num == 4 and len(sec) >= 28:
            info["template4"] = struct.unpack(">H", sec[7:9])[0]
            info["category"], info["number"] = sec[9], sec[10]
            unit = sec[17]
            info["step_h"] = struct.unpack(">i", sec[18:22])[0] * _TIME_UNITS_H.get(unit, np.nan)
            info["level_type"] = sec[22]
            scale = struct.unpack(">b", sec[23:24])[0]
            value = struct.unpack(">i", sec[24:28])[0]
            info["level"] = float(value) * 10.0 ** (-scale) if sec[23] != 255 else float("nan")
        elif num == 5 and len(sec) >= 11:
            info["packing"] = struct.unpack(">H", sec[9:11])[0]
            break
        pos += sec_len
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
    )


@dataclass
class GribIndex:
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
        return sorted({m.step_h for m in self.messages})

    # -- persistence ---------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {"key": self.key, "size": self.size, "messages": [asdict(m) for m in self.messages]}
        )

    @classmethod
    def from_json(cls, text: str) -> GribIndex:
        d = json.loads(text)
        return cls(d["key"], d["size"], [GribMessage(**m) for m in d["messages"]])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)


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
        raise ImportError("decoding GRIB needs ecCodes: pip install 'supertraj-met[grib]'") from exc
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
        return store.get(key, m.offset, m.offset + m.length)

    if executor is None or len(messages) <= 1:
        return [get(m) for m in messages]
    return list(executor.map(get, messages))
