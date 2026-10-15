"""Persistent, bounded decoded-cube reuse for successive trajectory queries."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from numpy.typing import ArrayLike

from baro.met.base import CubeKey, Source
from baro.met.domain import Cube, Domain
from baro.met.enrich import Enricher, EnrichResult
from baro.met.kernel import SurfacePressureMode, VerticalMethod
from baro.met.vertical import VerticalCoordinate

__all__ = ["WeatherSession"]


@dataclass(frozen=True)
class _Request:
    key: CubeKey
    domain: Domain
    pressure: tuple[float, float]
    margin: int

    def contains(self, other: _Request) -> bool:
        a, b = self.domain, other.domain
        # Conservative longitude framing: a different wrap may miss the cache,
        # but must never serve a cube outside the original request.
        return (
            self.key == other.key
            and self.margin >= other.margin
            and a.lat_min <= b.lat_min <= b.lat_max <= a.lat_max
            and a.lon_min <= b.lon_min <= b.lon_max <= a.lon_max
            and self.pressure[0] <= other.pressure[0] <= other.pressure[1] <= self.pressure[1]
        )


def _owned_cube(cube: Cube) -> tuple[Cube, int]:
    """Own compact arrays: views must not retain an uncounted global backing array."""
    vertical = cube.vertical
    copy = Cube(
        cube.source,
        cube.hour,
        cube.lat.copy(),
        cube.lon.copy(),
        VerticalCoordinate(
            vertical.kind, vertical.ids.copy(), vertical.a.copy(), vertical.b.copy()
        ),
        {k: v.copy() for k, v in cube.fields.items()},
        None if cube.ps is None else cube.ps.copy(),
        cube.run,
    )
    size = copy.nbytes + sum(
        a.nbytes for a in (copy.lat, copy.lon, copy.vertical.ids, copy.vertical.a, copy.vertical.b)
    )
    return copy, size


class _DecodedSource:
    def __init__(self, source: Source, budget: int) -> None:
        self.source = source
        self.budget = budget
        self.entries: OrderedDict[_Request, tuple[Cube, int]] = OrderedDict()
        self.lock = threading.Lock()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        req = _Request(key, domain, p_range, level_margin)
        with self.lock:
            hit = next((k for k in reversed(self.entries) if k.contains(req)), None)
            if hit is not None:
                self.entries.move_to_end(hit)
                self.hits += 1
                return self.entries[hit][0]
            self.misses += 1
        cube = self.source.fetch(key, domain, p_range, level_margin=level_margin, executor=executor)
        # Avoid copying oversized fields. Coordinates may still put a small cube
        # over budget; the exact retained size is checked after making it compact.
        if not self.budget or cube.nbytes > self.budget:
            return cube
        owned, size = _owned_cube(cube)
        if size <= self.budget:
            with self.lock:
                previous = self.entries.pop(req, None)
                if previous is not None:
                    self.bytes -= previous[1]
                while self.entries and self.bytes + size > self.budget:
                    self.bytes -= self.entries.popitem(last=False)[1][1]
                    self.evictions += 1
                self.entries[req] = (owned, size)
                self.bytes += size
        return cube


class WeatherSession:
    """Reuse decoded weather across successive point queries, without extrapolation.

    ``source`` supplies a fixed set of fields. Optional ``fields`` selects a
    subset through its ``select_fields`` capability (currently ``GridSource``);
    unsupported sources must be configured with their own field selection.
    ``vertical`` is required: choose ``"linear"`` or ``"log"`` in pressure.

    The LRU budget bounds retained array payload, including coordinates, not
    Python overhead or temporary cubes held by the active enrichment call.
    Calls on one session are serialized; its normal source I/O stays parallel.
    Treat a session as a snapshot: call :meth:`clear` to refresh decoded mutable
    forecasts or ERA5T. The existing on-disk byte cache retains its own rules.
    Do not mutate the source configuration while using the session.
    """

    def __init__(  # noqa: PLR0913 - explicit public sampling configuration
        self,
        source: Source,
        *,
        vertical: VerticalMethod,
        fields: Sequence[str] | None = None,
        max_cache_bytes: int = 256 * 1024 * 1024,
        cache_dir: str | Path | None = None,
        domain: Domain | None = None,
        margin_deg: float = 0.5,
        level_margin: int = 1,
        io_workers: int = 16,
        prefetch: int = 2,
        surface_pressure: SurfacePressureMode = "actual",
        on_uncovered: Literal["warn", "raise", "flag"] = "raise",
    ) -> None:
        if not isinstance(max_cache_bytes, int) or max_cache_bytes < 0:
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        if vertical not in ("linear", "log"):
            raise ValueError("vertical must be 'linear' or 'log'")
        if fields is not None:
            select = getattr(source, "select_fields", None)
            if select is None:
                raise TypeError("source does not support select_fields; configure its fields first")
            source = select(fields)
        self._source = _DecodedSource(source, max_cache_bytes)
        self._lock = threading.Lock()
        self._enricher = Enricher(
            self._source,
            cache_dir=cache_dir,
            domain=domain,
            margin_deg=margin_deg,
            level_margin=level_margin,
            io_workers=io_workers,
            prefetch=prefetch,
            vertical=vertical,
            surface_pressure=surface_pressure,
            on_uncovered=on_uncovered,
        )

    def sample(
        self, time_utc: ArrayLike, lat: ArrayLike, lon: ArrayLike, *, pressure_pa: ArrayLike
    ) -> EnrichResult:
        """Sample 1-D points using nanosecond time and explicit pressure in Pa.

        Missing coverage retains Enricher flags; callers must check ``valid``
        before interpreting missing meteorology as a physical disappearance.
        Time interpolation divides integer timedelta differences, never floating
        epoch hours, preserving subsecond weights at modern dates.
        """
        with self._lock:
            before = self.cache_info()
            result = self._enricher.enrich(time_utc, lat, lon, pressure_pa=pressure_pa)
            after = self.cache_info()
            result.stats.update(
                decoded_cache_hits=after["hits"] - before["hits"],
                decoded_cache_misses=after["misses"] - before["misses"],
                decoded_cache_bytes=after["bytes"],
            )
            return result

    def cache_info(self) -> dict[str, int]:
        """Retained array bytes, entry count and lifetime LRU counters."""
        with self._source.lock:
            return {
                "bytes": self._source.bytes,
                "entries": len(self._source.entries),
                "hits": self._source.hits,
                "misses": self._source.misses,
                "evictions": self._source.evictions,
                "max_bytes": self._source.budget,
            }

    def clear(self) -> None:
        """Release decoded arrays after any active call; preserve lifetime counters."""
        with self._lock, self._source.lock:
            self._source.entries.clear()
            self._source.bytes = 0
