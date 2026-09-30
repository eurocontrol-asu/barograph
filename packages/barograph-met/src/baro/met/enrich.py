"""Time-major enrichment of many trajectory points at once.

Remote files are organised by time (a global field per hour for ARCO-ERA5, one
message per variable/level/step for ARPEGE GRIB2), so the cost of a time slice
is paid once for every flight that crosses it. Points of *all* flights are
grouped by the interval ``[t0, t0 + step)`` of the source's native time step;
the source says which two cubes bracket each interval (and, for forecasts, from
which run). Each cube is fetched once (or read from the cache) over the domain
and pressure range of all the points that use it, with bounded look-ahead, and
released as soon as the last interval using it is done.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baro.met.base import CubeKey, DataUnavailableError, Source
from baro.met.cache import CubeCache, CubeQuery
from baro.met.domain import Cube, Domain
from baro.met.isa import pressure_from_altitude_ft
from baro.met.kernel import (
    Flag,
    SampleOptions,
    SurfacePressureMode,
    VerticalMethod,
    sample,
)

__all__ = ["EnrichResult", "Enricher", "UncoveredPointsWarning"]

_EPOCH = np.datetime64("1970-01-01T00", "ns")
_ALL_LEVELS = 1_000_000

type Mode = Literal["flight", "global"]


class UncoveredPointsWarning(UserWarning):
    """Some points fall outside the period the source covers."""


@dataclass
class EnrichResult:
    """Interpolated fields (NaN where ``flags != 0``) and run statistics."""

    fields: dict[str, NDArray[np.float64]]
    flags: NDArray[np.uint8]
    stats: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        return self.fields[name]

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Mask of the points with a value (no flag set)."""
        return np.asarray(self.flags == Flag.OK)


@dataclass(frozen=True)
class _CubeRequest:
    key: CubeKey
    domain: Domain
    p_range: tuple[float, float]


@dataclass(frozen=True)
class _Points:
    """Valid points of a run: interpolation weight in time, position, pressure."""

    w: NDArray[np.float64]
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    p: NDArray[np.float64]

    def take(self, sel: NDArray[np.int64] | NDArray[np.bool_]) -> _Points:
        """The subset of points selected by ``sel``."""
        return _Points(self.w[sel], self.lat[sel], self.lon[sel], self.p[sel])


@dataclass(frozen=True)
class _Interval:
    t0: np.datetime64
    points: NDArray[np.int64]  # indices into the valid points
    keys: tuple[CubeKey, CubeKey | None]  # second key only if some point needs it


class _Prefetcher:
    """Fetches cubes in order of first use, ``depth`` cubes ahead; frees them after last use."""

    def __init__(
        self,
        intervals: list[_Interval],
        fetch: Callable[[CubeKey], Future[Cube | None]],
        depth: int,
    ) -> None:
        used = [key for iv in intervals for key in iv.keys if key is not None]
        self._order = list(dict.fromkeys(used))
        self._position = {key: i for i, key in enumerate(self._order)}
        self._remaining = Counter(used)
        self._futures: dict[CubeKey, Future[Cube | None]] = {}
        self._submitted = 0
        self._fetch = fetch
        self._depth = depth

    def cubes(self, iv: _Interval) -> tuple[Cube | None, Cube | None]:
        """The two cubes of an interval (waiting for them), after scheduling the next ones."""
        needed = [k for k in iv.keys if k is not None]
        target = min(max(self._position[k] for k in needed) + self._depth, len(self._order))
        while self._submitted < target:
            key = self._order[self._submitted]
            self._futures[key] = self._fetch(key)
            self._submitted += 1
        k0, k1 = iv.keys
        return self._futures[k0].result(), None if k1 is None else self._futures[k1].result()

    def release(self, iv: _Interval) -> None:
        """Forget the cubes this interval was the last user of."""
        for key in iv.keys:
            if key is None:
                continue
            self._remaining[key] -= 1
            if self._remaining[key] == 0:
                self._futures.pop(key)


class Enricher:
    """Interpolate a gridded source along trajectories.

    Parameters
    ----------
    source:
        Gridded source, e.g. :func:`baro.met.arco_model_levels` or
        :class:`baro.met.arpege.ArpegeSource`.
    cache_dir:
        Optional immutable cube cache shared between runs and processes.
    mode:
        ``"flight"`` (default) reads only the region and levels the points
        need. ``"global"`` reads whole fields on every level (FastMeteo-like),
        so that one cache serves any later flight.
    domain:
        Fixed domain for every cube (recommended for campaigns: maximises
        cache reuse). By default each cube uses the box of its own points.
    margin_deg:
        Margin around the points when the domain is derived from them.
    level_margin:
        Extra levels kept above and below the bracketing levels.
    io_workers / prefetch:
        Concurrent range requests, and cubes fetched ahead of interpolation.
    vertical / surface_pressure:
        ``"log"`` or ``"linear"`` in pressure; ``"actual"`` surface pressure or
        ``"standard"`` (101325 Pa, reproduces a fixed altitude->level table).
    on_uncovered:
        What to do with points outside the period the source covers (e.g.
        before the start of an archive): ``"warn"`` (default) flags them
        ``OUT_OF_COVERAGE`` and emits a warning with the covered period,
        ``"raise"`` raises :class:`DataUnavailableError`, ``"flag"`` only flags.
    """

    def __init__(  # noqa: PLR0913 - keyword-only configuration of the public enricher
        self,
        source: Source,
        *,
        cache_dir: str | Path | None = None,
        mode: Mode = "flight",
        domain: Domain | None = None,
        margin_deg: float = 0.5,
        level_margin: int = 1,
        io_workers: int = 16,
        prefetch: int = 2,
        vertical: VerticalMethod = "log",
        surface_pressure: SurfacePressureMode = "actual",
        on_uncovered: Literal["warn", "raise", "flag"] = "warn",
    ) -> None:
        if on_uncovered not in ("warn", "raise", "flag"):
            raise ValueError(f"unknown on_uncovered {on_uncovered!r}")
        self.on_uncovered = on_uncovered
        if mode not in ("flight", "global"):
            raise ValueError(f"unknown mode {mode!r}")
        self.source = source
        self.cache = CubeCache(cache_dir) if cache_dir is not None else None
        self.mode = mode
        self.domain = domain if domain is not None or mode == "flight" else Domain.whole_globe()
        self.margin_deg = margin_deg
        self.level_margin = _ALL_LEVELS if mode == "global" else level_margin
        self.io_workers = io_workers
        self.prefetch = max(prefetch, 1)
        self.vertical = vertical
        self.surface_pressure = surface_pressure
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------------------
    def enrich(
        self,
        time_utc: ArrayLike,
        lat: ArrayLike,
        lon: ArrayLike,
        *,
        altitude_ft: ArrayLike | None = None,
        pressure_pa: ArrayLike | None = None,
    ) -> EnrichResult:
        """Fields at each point. Give barometric (pressure) altitude in ft or pressure in Pa."""
        if (altitude_ft is None) == (pressure_pa is None):
            raise ValueError("give exactly one of altitude_ft or pressure_pa")
        t = np.asarray(time_utc).astype("datetime64[ns]")
        la = np.asarray(lat, dtype=np.float64)
        lo = np.asarray(lon, dtype=np.float64)
        if pressure_pa is not None:
            p = np.asarray(pressure_pa, dtype=np.float64)
        else:
            p = pressure_from_altitude_ft(np.asarray(altitude_ft, dtype=np.float64))
        if not (t.shape == la.shape == lo.shape == p.shape) or t.ndim != 1:
            raise ValueError("inputs must be 1-D arrays of equal length")
        return self._run(t, la, lo, p)

    # -- internals -----------------------------------------------------------------------
    def _run(
        self,
        t: NDArray[np.datetime64],
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
        p: NDArray[np.float64],
    ) -> EnrichResult:
        started = time.perf_counter()
        before = [s.snapshot() for s in self.source.stats]
        n = t.size
        values = {name: np.full(n, np.nan) for name in self.source.fields}
        flags = np.full(n, Flag.INVALID_INPUT, dtype=np.uint8)
        ok = ~np.isnat(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(p) & (p > 0)
        uncovered = self._check_coverage(t, ok)
        flags[uncovered] = Flag.OUT_OF_COVERAGE
        ok &= ~uncovered
        idx = np.flatnonzero(ok)
        step = self.source.time_step
        t_ok = t[idx]
        t0 = t_ok - (t_ok - _EPOCH) % step
        w = (t_ok - t0) / step
        la, lo, pp = lat[idx], lon[idx], p[idx]
        points = _Points(w, la, lo, pp)
        intervals, requests = self._plan(t0, w, la, lo, pp)
        counters: dict[str, Any] = {
            "cubes": len(requests),
            "cache_hits": 0,
            "fetched": 0,
            "unavailable": 0,
            "cube_bytes": 0,
        }
        with (
            ThreadPoolExecutor(self.io_workers, thread_name_prefix="traj-io") as io,
            ThreadPoolExecutor(self.prefetch, thread_name_prefix="traj-cube") as cubes,
        ):
            for iv, cube0, cube1 in self._stream(intervals, requests, io, cubes, counters):
                sel = iv.points
                v, f = self._interpolate(cube0, cube1, points.take(sel))
                for name in values:
                    values[name][idx[sel]] = v[name]
                flags[idx[sel]] = f
        after = [s.snapshot() for s in self.source.stats]
        for name in ("bytes", "requests", "throttled"):
            counters[name] = int(sum(a[name] - b[name] for a, b in zip(after, before, strict=True)))
        counters["seconds"] = time.perf_counter() - started
        return EnrichResult(values, flags, counters)

    def _check_coverage(
        self, t: NDArray[np.datetime64], ok: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """Points whose bracketing interval falls outside the source's coverage."""
        coverage = self.source.coverage()
        uncovered = np.zeros(t.size, dtype=bool)
        if coverage is None or not ok.any():
            return uncovered
        start = np.datetime64(coverage[0], "ns")
        end = np.datetime64(coverage[1], "ns")
        uncovered[ok] = (t[ok] < start) | (t[ok] > end)
        n = int(uncovered.sum())
        if n == 0 or self.on_uncovered == "flag":
            return uncovered
        before = int((uncovered & (t < start)).sum())
        after = n - before
        span = f"{t[uncovered].min()} .. {t[uncovered].max()}"
        message = (
            f"{n} point(s) outside the period covered by {self.source.source_id} "
            f"({coverage[0]} .. {coverage[1]} UTC): {before} before, {after} after "
            f"(their times span {span}); they are flagged OUT_OF_COVERAGE"
        )
        if self.on_uncovered == "raise":
            raise DataUnavailableError(message)
        warnings.warn(message, UncoveredPointsWarning, stacklevel=4)
        return uncovered

    def _plan(
        self,
        t0: NDArray[np.datetime64],
        w: NDArray[np.float64],
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
        p: NDArray[np.float64],
    ) -> tuple[list[_Interval], dict[CubeKey, _CubeRequest]]:
        """Intervals in time order, and one request per cube (union of its users)."""
        intervals = []
        uses: dict[CubeKey, list[NDArray[np.int64]]] = {}
        for start, members in _group(t0).items():
            k0, k1 = self.source.cube_keys(start)
            later = members[w[members] > 0]
            uses.setdefault(k0, []).append(members)
            if later.size:
                uses.setdefault(k1, []).append(later)
            intervals.append(_Interval(start, members, (k0, k1 if later.size else None)))
        requests = {}
        for key, parts in uses.items():
            sel = np.concatenate(parts)
            domain = self.domain or Domain.from_points(lat[sel], lon[sel], self.margin_deg)
            requests[key] = _CubeRequest(key, domain, (float(p[sel].min()), float(p[sel].max())))
        return intervals, requests

    def _stream(
        self,
        intervals: list[_Interval],
        requests: dict[CubeKey, _CubeRequest],
        io: ThreadPoolExecutor,
        pool: ThreadPoolExecutor,
        counters: dict[str, Any],
    ) -> Iterator[tuple[_Interval, Cube | None, Cube | None]]:
        """Intervals with their cubes; cubes are fetched ahead and freed after last use."""

        def fetch(key: CubeKey) -> Future[Cube | None]:
            """Submit the download of one cube."""
            return pool.submit(self._get_cube, requests[key], io, counters)

        ahead = _Prefetcher(intervals, fetch, self.prefetch)
        for iv in intervals:
            cube0, cube1 = ahead.cubes(iv)
            yield iv, cube0, cube1
            ahead.release(iv)

    def _get_cube(
        self, req: _CubeRequest, io: ThreadPoolExecutor, counters: dict[str, Any]
    ) -> Cube | None:
        src = req.key.source or self.source.source_id
        if self.cache is None:
            return self._download(req, io, counters)
        hit = self._cached(src, req)
        if hit is not None:
            self._count(counters, "cache_hits")
            return hit
        with self.cache.claim(src, req.key):
            hit = self._cached(src, req)  # another process may have fetched it meanwhile
            if hit is not None:
                self._count(counters, "cache_hits")
                return hit
            cube = self._download(req, io, counters)
            if cube is not None:
                self.cache.put(cube)
            return cube

    def _cached(self, src: str, req: _CubeRequest) -> Cube | None:
        assert self.cache is not None
        query = CubeQuery(req.key, req.domain, req.p_range, self.level_margin)
        return self.cache.find(src, query, list(self.source.fields))

    def _download(
        self, req: _CubeRequest, io: ThreadPoolExecutor, counters: dict[str, Any]
    ) -> Cube | None:
        try:
            cube = self.source.fetch(
                req.key, req.domain, req.p_range, level_margin=self.level_margin, executor=io
            )
        except DataUnavailableError:
            self._count(counters, "unavailable")
            return None
        self._count(counters, "fetched")
        self._count(counters, "cube_bytes", cube.nbytes)
        return cube

    def _count(self, counters: dict[str, Any], name: str, amount: int = 1) -> None:
        with self._lock:
            counters[name] += amount

    def _interpolate(
        self,
        cube0: Cube | None,
        cube1: Cube | None,
        pts: _Points,
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.uint8]]:
        n = pts.w.size
        names = list(self.source.fields)
        if cube0 is None:
            return {k: np.full(n, np.nan) for k in names}, np.full(n, Flag.MISSING_DATA, np.uint8)
        opts = SampleOptions(self.vertical, self.surface_pressure)
        v0, flags = sample(cube0, pts.lat, pts.lon, pts.p, opts)
        later = pts.w > 0
        if not later.any():
            return v0, flags
        if cube1 is None:
            v1 = {k: np.full(int(later.sum()), np.nan) for k in names}
            f1 = np.full(int(later.sum()), Flag.MISSING_DATA, np.uint8)
        else:
            nxt = pts.take(later)
            v1, f1 = sample(cube1, nxt.lat, nxt.lon, nxt.p, opts)
        flags = flags.copy()
        flags[later] |= np.where(flags[later] == 0, f1, 0).astype(np.uint8)
        out = {}
        wl = pts.w[later]
        for name in names:
            blended = v0[name].copy()
            blended[later] = (1.0 - wl) * v0[name][later] + wl * v1[name]
            blended[flags != 0] = np.nan
            out[name] = blended
        return out, flags


def _group(t0: NDArray[np.datetime64]) -> dict[np.datetime64, NDArray[np.int64]]:
    """Indices of the points of each interval start, in time order."""
    if t0.size == 0:
        return {}
    order = np.argsort(t0, kind="stable")
    starts_t, starts = np.unique(t0[order], return_index=True)
    bounds = [*starts[1:], t0.size]
    return {
        np.datetime64(s, "s"): order[a:b] for s, a, b in zip(starts_t, starts, bounds, strict=True)
    }
