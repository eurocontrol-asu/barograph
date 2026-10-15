"""Demand-driven native weather windows with bounded decoded reuse."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from baro.met.arco_surface import ArcoRadiationSource
from baro.met.domain import FULL_TURN, Domain
from baro.met.materialize import WeatherRequest, materialize_radiation, materialize_weather
from baro.met.rangecache import RangeCache
from baro.met.sources import GridSource
from baro.met.weather import RadiationCube, WeatherCube

if TYPE_CHECKING:
    from baro.met.diffcocip import WeatherDemand

__all__ = ["WeatherWindowProvider"]

_EPOCH = int(np.datetime64("2000-01-01", "ns").astype(np.int64))
_NO_PREFETCH = np.timedelta64(0, "ns")
_DEFAULT_FIELDS = ("u", "v", "t", "q", "omega", "tau_cirrus")
type _Window = tuple[WeatherCube, RadiationCube]


def _contains(window: _Window, request: WeatherRequest) -> bool:
    d = request.domain
    for cube in window:
        lon, lat, times = np.asarray(cube.lon), np.asarray(cube.lat), np.asarray(cube.time_ns)
        if not (
            lon[0] <= d.lon_min <= d.lon_max <= lon[-1]
            and lat[0] <= d.lat_min <= d.lat_max <= lat[-1]
            and times[0] <= request.start.astype(np.int64)
            and request.end.astype(np.int64) <= times[-1]
        ):
            return False
    levels = np.asarray(window[0].level_hpa) * 100
    return bool(levels[0] <= request.pressure_pa[0] <= request.pressure_pa[1] <= levels[-1])


def _freeze(window: _Window) -> int:
    """Freeze owned arrays and count payload including integer time coordinates."""
    size = 0
    for cube in window:
        arrays = [cube.lon, cube.lat, cube.time_ns, *cube.fields.values()]
        if isinstance(cube, WeatherCube):
            arrays.append(cube.level_hpa)
        for value in arrays:
            array = np.asarray(value)
            array.flags.writeable = False
            size += array.nbytes
    return size


def _utc_ns(seconds: float, *, upper: bool) -> np.datetime64:
    if not math.isfinite(seconds):
        raise ValueError("demand time bounds must be finite")
    # Never add the Unix epoch as a float: preserve all precision available in
    # the engine's seconds since 2000. Round outward for bounding-box support.
    whole = math.floor(seconds)
    fraction = (seconds - whole) * 1_000_000_000
    value = (
        _EPOCH + whole * 1_000_000_000 + (math.ceil(fraction) if upper else math.floor(fraction))
    )
    if not np.iinfo(np.int64).min < value <= np.iinfo(np.int64).max:
        raise ValueError("demand time exceeds datetime64 nanosecond range")
    return np.datetime64(value, "ns")


def _native_fields(source: GridSource, fields: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """Validate the source coordinate and resolve native versus derived fields."""
    if source.vertical.kind != "pressure":
        raise ValueError("native weather windows require pressure levels")
    names = tuple(fields)
    if not names or len(set(names)) != len(names):
        raise ValueError("fields must be nonempty and unique")
    derive_tau = "tau_cirrus" in names and "tau_cirrus" not in source.fields
    native = tuple(n for n in names if n != "tau_cirrus" or not derive_tau)
    if not native:
        raise ValueError("derived tau requires at least one requested native weather field")
    return native, derive_tau


def _retrieval_options(
    budget: int, margin_deg: float, pressure_margin_pa: float, prefetch: np.timedelta64
) -> np.timedelta64:
    """Validate bounded retention and nonnegative speculative retrieval margins."""
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("max_cache_bytes must be a nonnegative integer")
    if (
        not np.isfinite([margin_deg, pressure_margin_pa]).all()
        or min(margin_deg, pressure_margin_pa) < 0
    ):
        raise ValueError("spatial and pressure margins must be finite and nonnegative")
    duration = np.timedelta64(prefetch, "ns")
    if np.isnat(duration) or duration < np.timedelta64(0, "ns"):
        raise ValueError("prefetch must be a finite nonnegative duration")
    return duration


class WeatherWindowProvider:
    """Fetch just the native weather windows requested by a forward model.

    Fields are fixed at construction. The default includes CoCiP's optical
    depth; when absent from the source it is computed from the complete upper
    cirrus column by :func:`materialize_weather`, requiring optional pycontrails.

    Returned containers are borrowed snapshots: their arrays are read-only;
    callers must not replace coordinates, fields or metadata. Engine exports
    make independent arrays. The byte budget counts retained array payload,
    excluding Python overhead, active downloads, exports and caller references.
    Calls are serialized. Use ``clear()`` to refresh decoded mutable data;
    the underlying immutable byte cache keeps its existing policy.
    """

    def __init__(  # noqa: PLR0913 - explicit provider retrieval configuration
        self,
        source: GridSource,
        radiation: ArcoRadiationSource,
        *,
        fields: Sequence[str] = _DEFAULT_FIELDS,
        max_cache_bytes: int = 256 * 1024 * 1024,
        cache_dir: str | Path | None = None,
        margin_deg: float = 0.0,
        pressure_margin_pa: float = 0.0,
        prefetch: np.timedelta64 = _NO_PREFETCH,
    ) -> None:
        self._fields, self._derive_tau = _native_fields(source, fields)
        duration = _retrieval_options(max_cache_bytes, margin_deg, pressure_margin_pa, prefetch)
        # Materialization takes its own field views, keeping dependencies such
        # as ciwc available only for the separate upper-column calculation.
        self.source, self.radiation = source, radiation
        self.max_cache_bytes = max_cache_bytes
        self.margin_deg, self.pressure_margin_pa = margin_deg, pressure_margin_pa
        self.prefetch = duration
        self._cache: OrderedDict[int, tuple[_Window, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = self._hits = self._misses = self._evictions = self._next = 0
        if cache_dir is not None:
            cache = RangeCache(cache_dir)
            source.use_cache(cache)
            radiation.use_cache(cache)

    def _expanded(self, request: WeatherRequest) -> WeatherRequest:
        d, m = request.domain, self.margin_deg
        width = d.lon_max - d.lon_min + 2 * m
        centre = (d.lon_min + d.lon_max) / 2
        lon_min, lon_max = (
            (d.lon_min - m, d.lon_max + m) if width < FULL_TURN else (centre - 180, centre + 180)
        )
        domain = Domain(max(-90, d.lat_min - m), min(90, d.lat_max + m), lon_min, lon_max)
        # Reject the actual demand before clipping speculative margins.
        native = self.source.vertical.a
        if not native[0] <= request.pressure_pa[0] <= request.pressure_pa[1] <= native[-1]:
            raise ValueError("weather demand exceeds source pressure coverage")
        lo = max(float(native[0]), request.pressure_pa[0] - self.pressure_margin_pa)
        hi = min(float(native[-1]), request.pressure_pa[1] + self.pressure_margin_pa)
        end_ns = int(request.end.astype(np.int64)) + int(self.prefetch.astype(np.int64))
        for bounds in (self.source.coverage(), self.radiation.coverage()):
            if bounds is not None:
                if not bounds[0] <= request.start <= request.end <= bounds[1]:
                    raise ValueError("weather demand exceeds source time coverage")
                end_ns = min(end_ns, int(np.datetime64(bounds[1], "ns").astype(np.int64)))
        if end_ns > np.iinfo(np.int64).max:
            raise ValueError("prefetch exceeds datetime64 nanosecond range")
        return WeatherRequest(domain, request.start, np.datetime64(end_ns, "ns"), (lo, hi))

    def get(self, request: WeatherRequest) -> _Window:
        """Return native met/radiation grids fully covering ``request``.

        Cache hits use actual coordinate coverage, including native bracketing
        neighbours. Misses use the usual targeted readers; no preadvection or
        whole archive retrieval is performed. Failures are never cached.
        """
        with self._lock:
            hit = next(
                (k for k in reversed(self._cache) if _contains(self._cache[k][0], request)), None
            )
            if hit is not None:
                self._cache.move_to_end(hit)
                self._hits += 1
                return self._cache[hit][0]
            self._misses += 1
            expanded = self._expanded(request)
            window = (
                materialize_weather(
                    self.source, expanded, fields=self._fields, tau_cirrus=self._derive_tau
                ),
                materialize_radiation(self.radiation, expanded),
            )
            if not _contains(window, request):
                raise ValueError("materialized weather does not cover the complete demand")
            size = _freeze(window)
            if size <= self.max_cache_bytes:
                while self._cache and self._bytes + size > self.max_cache_bytes:
                    self._bytes -= self._cache.popitem(last=False)[1][1]
                    self._evictions += 1
                self._cache[self._next] = (window, size)
                self._next += 1
                self._bytes += size
            return window

    def for_diffcocip(
        self, demand: WeatherDemand
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """Driver callback: lon/lat degrees, pressure hPa, seconds since 2000.

        Downloads run on the host outside JAX compilation/autodifferentiation.
        The driver remains responsible for complete evolution/lower-probe
        demands and for retrying a step when new support is required.
        """
        bounds = demand.bounds
        lon, lat, level, time = (bounds[k] for k in ("lon", "lat", "level", "time_s"))
        request = WeatherRequest(
            Domain(lat[0], lat[1], lon[0], lon[1]),
            _utc_ns(time[0], upper=False),
            _utc_ns(time[1], upper=True),
            (level[0] * 100, level[1] * 100),
        )
        weather, radiation = self.get(request)
        return weather.to_diffcocip(), radiation.to_diffcocip()

    def cache_info(self) -> dict[str, int]:
        """Retained payload, entries and lifetime hit/miss/eviction counters."""
        with self._lock:
            return {
                "bytes": self._bytes,
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "max_bytes": self.max_cache_bytes,
            }

    def clear(self) -> None:
        """Release decoded windows, retaining byte caches and lifetime counters."""
        with self._lock:
            self._cache.clear()
            self._bytes = 0
