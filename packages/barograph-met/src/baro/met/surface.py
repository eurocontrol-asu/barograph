"""Surface fields with explicit heights, units and native interval semantics.

Instantaneous values are bilinearly interpolated in space and linearly in
 time. Period totals and maxima select the native interval ending at or after
 the query: a sample stamped ``t`` represents ``(t - step, t]``. They are
 never differenced or linearly interpolated in time. Spatially interpolated
 gust maxima describe the model field, not a guaranteed local extreme.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import LAT_LIMIT, Domain
from baro.met.kernel import Flag
from baro.met.rangecache import RangeCache
from baro.met.store import TransferStats
from baro.met.tiling import Tiling, partition

_GRID_EPS = 1e-9
_MIN_AXIS_POINTS = 2

__all__ = ["SurfaceCube", "SurfaceEnricher", "SurfaceResult", "SurfaceSource", "SurfaceVariable"]


@dataclass(frozen=True)
class SurfaceVariable:
    """Unit, height above ground and native temporal meaning of one field."""

    unit: str
    height_m: float
    temporal: Literal["instant", "sum", "max"] = "instant"


@dataclass(frozen=True)
class SurfaceCube:
    """Native valid-time surface fields on an ascending regular lat/lon grid."""

    key: CubeKey
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    fields: dict[str, NDArray[np.float32]]
    variables: dict[str, SurfaceVariable]

    def __post_init__(self) -> None:
        for axis in (self.lat, self.lon):
            if axis.ndim != 1 or len(axis) < _MIN_AXIS_POINTS or not np.all(np.isfinite(axis)):
                raise ValueError("surface axes must have at least two finite coordinates")
            increments = np.diff(axis)
            if increments[0] <= 0 or not np.allclose(increments, increments[0]):
                raise ValueError("surface axes must be ascending and regularly spaced")
        if set(self.fields) != set(self.variables):
            raise ValueError("surface field names and metadata must agree")
        if any(a.shape != (len(self.lat), len(self.lon)) for a in self.fields.values()):
            raise ValueError("surface fields must have shape (latitude, longitude)")

    def sample(
        self, lat: ArrayLike, lon: ArrayLike, *, fields: tuple[str, ...] | None = None
    ) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.uint8]]:
        """Bilinearly interpolate the grid without horizontal extrapolation."""
        la, lo = np.broadcast_arrays(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        la, lo = la.ravel(), lo.ravel()
        invalid = ~np.isfinite(la) | ~np.isfinite(lo)
        la = np.where(invalid, self.lat[0], la)
        lo = np.where(invalid, self.lon[0], lo)
        fi = (la - self.lat[0]) / (self.lat[1] - self.lat[0])
        centre = (self.lon[0] + self.lon[-1]) / 2
        framed = lo + 360.0 * np.round((centre - lo) / 360.0)
        fj = (framed - self.lon[0]) / (self.lon[1] - self.lon[0])
        outside = (
            (fi < -_GRID_EPS)
            | (fi > len(self.lat) - 1 + _GRID_EPS)
            | (fj < -_GRID_EPS)
            | (fj > len(self.lon) - 1 + _GRID_EPS)
        )
        fi = np.clip(np.nan_to_num(fi), 0, len(self.lat) - 1)
        fj = np.clip(np.nan_to_num(fj), 0, len(self.lon) - 1)
        i = np.minimum(fi.astype(np.int64), len(self.lat) - 2)
        j = np.minimum(fj.astype(np.int64), len(self.lon) - 2)
        wi, wj = fi - i, fj - j
        flags = np.zeros(len(la), dtype=np.uint8)
        flags[outside] = Flag.OUTSIDE_DOMAIN
        flags[invalid] = Flag.INVALID_INPUT
        values = {}
        for name in self.fields if fields is None else fields:
            grid = self.fields[name]
            value = np.zeros(len(la))
            for di, dj, weight in (
                (0, 0, (1 - wi) * (1 - wj)),
                (1, 0, wi * (1 - wj)),
                (0, 1, (1 - wi) * wj),
                (1, 1, wi * wj),
            ):
                corner = grid[i + di, j + dj]
                # Missing unused corners must not invalidate exact grid points.
                value += np.where(weight == 0, 0, weight * corner)
            flags[(~np.isfinite(value)) & (flags == 0)] = Flag.MISSING_DATA
            values[name] = value
        for value in values.values():
            value[flags != 0] = np.nan
        return values, flags


@dataclass
class SurfaceResult:
    """Surface query values and native support/provenance for every output.

    Period fields retain full native interval totals/maxima even at fractional
    query times. ``interval_start`` and ``interval_end`` are per-field arrays;
    instant fields have both equal to query time. ``run`` is NaT and ``lead``
    NaT where a stitched archive has discarded the original run identity.
    """

    fields: dict[str, NDArray[np.float64]]
    flags: NDArray[np.uint8]
    variables: dict[str, SurfaceVariable]
    interval_start: dict[str, NDArray[np.datetime64]]
    interval_end: dict[str, NDArray[np.datetime64]]
    run: NDArray[np.datetime64]
    lead: NDArray[np.timedelta64]
    stats: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        return self.fields[name]

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Points with every requested field available."""
        return np.asarray(self.flags == Flag.OK)


class SurfaceSource(Protocol):
    """Native gridded surface-source contract, separate from pressure fields."""

    time_step: np.timedelta64
    variables: dict[str, SurfaceVariable]

    @property
    def stats(self) -> list[TransferStats]: ...
    def coverage(self) -> tuple[np.datetime64, np.datetime64]: ...
    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]: ...
    def tiling(self, key: CubeKey) -> Tiling | None: ...
    def use_cache(self, cache: RangeCache) -> None: ...
    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube: ...


class SurfaceEnricher:
    """Retrieve surface weather at fixed sites or trajectories in one batch.

    Points sharing a native interval are grouped by occupied geographic tiles;
    only requested fields and their interpolation neighbours are read. Supply
    ``cache_dir`` for persistent byte-range caching. Inputs are UTC datetime64
    (Unix nanoseconds are accepted), latitude and longitude arrays of equal
    length. No altitude or pressure argument is needed.
    """

    def __init__(
        self,
        source: SurfaceSource,
        *,
        cache_dir: str | Path | None = None,
        wind_components: tuple[str, str] | None = ("u10", "v10"),
    ) -> None:
        self.source = source
        self.wind_components = wind_components
        if cache_dir is not None:
            source.use_cache(RangeCache(cache_dir))

    def enrich(self, time: ArrayLike, lat: ArrayLike, lon: ArrayLike) -> SurfaceResult:
        """Interpolate instants; preserve the enclosing native totals and maxima."""
        try:
            times, la, lo = np.broadcast_arrays(
                np.asarray(time, dtype="datetime64[ns]"),
                np.asarray(lat, dtype=float),
                np.asarray(lon, dtype=float),
            )
        except ValueError as exc:
            raise ValueError(
                "time, latitude and longitude must broadcast to the same shape"
            ) from exc
        times, la, lo = times.ravel(), la.ravel(), lo.ravel()
        result = self._empty(times)
        valid = ~np.isnat(times) & np.isfinite(la) & np.isfinite(lo) & (abs(la) <= LAT_LIMIT)
        result.flags[~valid] = Flag.INVALID_INPUT
        if times.size == 0:
            self._wind(result, times)
            result.stats = {"bytes": 0}
            return result
        start, end = self.source.coverage()
        covered = (times >= start) & (times <= end)
        result.flags[valid & ~covered] = Flag.OUT_OF_COVERAGE
        points = np.flatnonzero(valid & covered)
        step_ns = int(self.source.time_step / np.timedelta64(1, "ns"))
        groups = times[points].astype(np.int64) // step_ns
        before = sum(s.bytes for s in self.source.stats)
        for group in np.unique(groups):
            selected = points[groups == group]
            t0 = np.datetime64(int(group) * step_ns, "ns")
            try:
                keys = self.source.cube_keys(t0)
                tiling = self.source.tiling(keys[0])
            except DataUnavailableError:
                result.flags[selected] = Flag.MISSING_DATA
                continue
            if keys[0].run != keys[1].run:
                raise ValueError("surface interpolation requires both endpoints from the same run")
            labels = partition(la[selected], lo[selected], 0.0, tiling)
            for label in np.unique(labels):
                sel = selected[labels == label]
                exact = times[sel] == keys[0].valid
                for mask in (exact, ~exact):
                    if mask.any():
                        self._group(result, times, la, lo, sel[mask], keys)
        self._wind(result, times)
        result.stats = {"bytes": sum(s.bytes for s in self.source.stats) - before}
        return result

    def _wind(self, result: SurfaceResult, times: NDArray[np.datetime64]) -> None:
        if self.wind_components is None or not all(
            n in result.fields for n in self.wind_components
        ):
            return
        u, v = (result.fields[n] for n in self.wind_components)
        speed = np.hypot(u, v)
        direction = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
        direction[speed == 0] = np.nan  # wind-from direction is undefined in calm air
        for name, value, unit in (
            ("wind_speed", speed, "m/s"),
            ("wind_direction", direction, "degree"),
        ):
            if name in result.fields:
                raise ValueError(f"derived field {name} conflicts with a source field")
            result.fields[name] = value
            result.variables[name] = SurfaceVariable(
                unit, result.variables[self.wind_components[0]].height_m
            )
            result.interval_start[name] = times.copy()
            result.interval_end[name] = times.copy()

    def _empty(self, times: NDArray[np.datetime64]) -> SurfaceResult:
        variables = dict(self.source.variables)
        n = len(times)
        return SurfaceResult(
            {name: np.full(n, np.nan) for name in variables},
            np.zeros(n, dtype=np.uint8),
            variables,
            {name: np.full(n, np.datetime64("NaT", "ns")) for name in variables},
            {name: np.full(n, np.datetime64("NaT", "ns")) for name in variables},
            np.full(n, np.datetime64("NaT", "ns")),
            np.full(n, np.timedelta64("NaT", "ns")),
        )

    def _group(  # noqa: PLR0913, PLR0917 - private point-group interpolation
        self,
        result: SurfaceResult,
        times: NDArray[np.datetime64],
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
        sel: NDArray[np.int64],
        keys: tuple[CubeKey, CubeKey],
    ) -> None:
        domain = Domain.from_points(lat[sel], lon[sel], margin=0.0)
        w = np.asarray((times[sel] - keys[0].valid) / (keys[1].valid - keys[0].valid), dtype=float)
        instant = any(v.temporal == "instant" for v in result.variables.values())
        # Exact native timestamps (including the final archive sample) need no future cube.
        needed = ([0] if instant or np.any(w == 0) else []) + ([1] if np.any(w != 0) else [])
        sampled = {}
        try:
            for which in needed:
                names = tuple(
                    n
                    for n, v in result.variables.items()
                    if which != 0 or np.all(w == 0) or v.temporal == "instant"
                )
                cube = self.source.fetch(keys[which], domain, fields=names)
                sampled[which] = cube.sample(lat[sel], lon[sel], fields=names)
        except DataUnavailableError:
            result.flags[sel] |= np.uint8(Flag.MISSING_DATA)
            return
        for name, variable in result.variables.items():
            if variable.temporal == "instant":
                value, flags = sampled[0][0][name].copy(), sampled[0][1].copy()
                if 1 in sampled:
                    active = w != 0
                    value[active] = (1 - w[active]) * value[active] + w[active] * sampled[1][0][
                        name
                    ][active]
                    flags[active] |= sampled[1][1][active]
                begin = finish = times[sel]
            else:
                value, flags = np.full(len(sel), np.nan), np.zeros(len(sel), dtype=np.uint8)
                finish = np.where(w == 0, keys[0].valid, keys[1].valid).astype("datetime64[ns]")
                begin = finish - self.source.time_step
                for which, (values, why) in sampled.items():
                    active = (w == 0) if which == 0 else (w != 0)
                    if active.any():
                        value[active], flags[active] = values[name][active], why[active]
            result.fields[name][sel] = value
            result.flags[sel] |= flags
            result.interval_start[name][sel], result.interval_end[name][sel] = begin, finish
        key = keys[0]
        if key.run is not None:
            result.run[sel] = key.run
            result.lead[sel] = times[sel] - key.run
        for value in result.fields.values():
            value[sel[result.flags[sel] != 0]] = np.nan
