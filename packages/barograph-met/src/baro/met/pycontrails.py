"""Optional lazy native grids for unmodified pycontrails consumers.

Only array slices actually evaluated by the consumer are materialized. This
module performs no aircraft or contrail physics and installs no engine hooks.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from functools import partial
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met._lazy_backend import Axes, lazy_variable
from baro.met.arco_surface import ArcoRadiationSource
from baro.met.base import DataUnavailableError
from baro.met.domain import FULL_TURN, Domain
from baro.met.materialize import WeatherRequest, _times, materialize_radiation, materialize_weather
from baro.met.rangecache import RangeCache
from baro.met.sources import GridSource
from baro.met.weather import MET_FIELDS, RAD_FIELDS

__all__ = ["PycontrailsWeather"]

_DEFAULT_FIELDS = ("u", "v", "t", "q", "omega", "tau_cirrus")
_MIN_AXIS = 2


def _longitude(native: NDArray[Any]) -> NDArray[np.float64]:
    native = np.sort(np.asarray(native, dtype=np.float64))
    lon = np.sort((native + 180) % FULL_TURN - 180)
    if len(lon) < _MIN_AXIS or np.any(np.diff(lon) <= 0):
        raise ValueError("native longitude needs at least two unique coordinates")
    delta = np.diff(native)
    if np.allclose(delta, delta[0]) and np.isclose(len(native) * delta[0], FULL_TURN):
        # Native pycontrails interpolation must bracket the seam. The appended
        # endpoint reads the same source column as the first longitude.
        lon = np.append(lon, lon[0] + FULL_TURN)
    elif np.any(np.diff(lon) > 180):  # noqa: PLR2004
        raise ValueError(
            "regional grids crossing the antimeridian need an explicit regional adapter"
        )
    return lon


def _attributes(source: Any, supplied: Mapping[str, str] | None) -> dict[str, str]:
    attrs = getattr(getattr(source, "store", None), "attrs", {})
    attrs = {**attrs, **getattr(source, "attrs", {})}
    out = {key: str(attrs.get(key, "unknown")) for key in ("provider", "dataset", "product")}
    out["source"] = str(getattr(source, "source_id", "unknown"))
    out.update(supplied or {})
    return out


def _select(native: Sequence[NDArray[Any]], target: Axes, values: Any) -> NDArray[np.float64]:
    selections = []
    for index, (axis, wanted) in enumerate(zip(native, target, strict=True)):
        axis, wanted = np.asarray(axis), np.asarray(wanted)
        if index == 0:
            wanted = wanted + FULL_TURN * np.round((axis.mean() - wanted) / FULL_TURN)
        pos = np.searchsorted(axis, wanted)
        bounded = np.clip(pos, 0, len(axis) - 1)
        if not np.array_equal(axis[bounded], wanted):
            raise DataUnavailableError(
                "materialized coordinates do not cover exact lazy grid request"
            )
        selections.append(pos)
    return np.array(np.asarray(values)[np.ix_(*selections)], dtype=np.float64, copy=True)


def _fields(source: GridSource, fields: Sequence[str]) -> tuple[str, ...]:
    if source.vertical.kind != "pressure":
        raise ValueError("native lazy weather requires pressure levels")
    names = tuple(fields)
    if not names or len(set(names)) != len(names) or set(names) - MET_FIELDS.keys():
        raise ValueError("select unique known pressure weather fields")
    dependencies = (
        {"t", "ciwc"} if "tau_cirrus" in names and "tau_cirrus" not in source.fields else set()
    )
    missing = (set(names) - {"tau_cirrus"} | dependencies) - source.fields.keys()
    if missing:
        raise ValueError(f"source does not supply {sorted(missing)}")
    return names


class PycontrailsWeather:
    """Supply lazy native pressure grids and optional resolved radiation grids.

    ``datasets`` creates pycontrails MetDataset objects with no field reads.
    Source constructors may already have read coordinate metadata. Fields are
    read only when the consumer evaluates an indexed array. The retained-cache
    budget excludes source buffers, tau-column temporaries, xarray/consumer
    caches and caller-held arrays; it is not a process memory limit.

    Metadata comes from source attributes or explicit ``provenance``; unknown
    metadata stays unknown. Radiation times are already resolved by its source.
    Use native CoCiP with ``interpolation_bounds_error=True`` and
    ``compute_tau_cirrus_in_model_init=False`` to preserve strict coverage and
    lazy cirrus reads. Missing/nonfinite weather raises, never means extinction.
    Only final source time chunks qualify for bridge retention; recent ERA5T
    is always materialized anew. Do not mutate source configuration in use.
    """

    def __init__(  # noqa: PLR0913 - public optional retrieval configuration
        self,
        source: GridSource,
        radiation: ArcoRadiationSource | None = None,
        *,
        fields: Sequence[str] = _DEFAULT_FIELDS,
        max_cache_bytes: int = 256 * 1024**2,
        cache_dir: str | Path | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> None:
        names = _fields(source, fields)
        if (
            not isinstance(max_cache_bytes, int)
            or isinstance(max_cache_bytes, bool)
            or max_cache_bytes < 0
        ):
            raise ValueError("max_cache_bytes must be a nonnegative integer")
        self.source, self.radiation, self.fields = source, radiation, names
        self.max_cache_bytes = max_cache_bytes
        self.provenance = dict(provenance or {})
        self._cache: OrderedDict[Any, NDArray[np.float64]] = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = self._hits = self._misses = self._evictions = 0
        self._seconds = 0.0
        self._field_reads: dict[str, int] = {}
        if cache_dir is not None:
            cache = RangeCache(cache_dir)
            source.use_cache(cache)
            if radiation is not None:
                radiation.use_cache(cache)

    def _retention_allowed(self, field: str, axes: Axes, radiation: bool) -> bool:
        source = self.radiation.grid if radiation and self.radiation is not None else self.source
        # Reserve a complete time chunk beyond the selected interval. This is
        # deliberately conservative for midpoint radiation and tau dependencies.
        names = ("t", "ciwc") if field == "tau_cirrus" and field not in source.fields else (field,)
        arrays = tuple(source.fields[name] for name in names)
        hours = max(source.store.array(name).chunks[0] for name in arrays)
        latest = axes[-1].max() + np.timedelta64(hours + 1, "h")
        return bool(latest < np.datetime64("now", "ns") - source.final_after)

    def _read(self, field: str, axes: Axes, radiation: bool = False) -> NDArray[np.float64]:
        key = (field, radiation, tuple((a.dtype.str, a.tobytes()) for a in axes))
        with self._lock:
            eligible = self.max_cache_bytes > 0 and self._retention_allowed(field, axes, radiation)
            if eligible and key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            started = perf_counter()
            out = self._materialize(field, axes, radiation)
            if not np.isfinite(out).all():
                raise DataUnavailableError("Nonfinite materialized meteorology")
            out.setflags(write=False)
            self._seconds += perf_counter() - started
            self._field_reads[field] = self._field_reads.get(field, 0) + 1
            if eligible and out.nbytes <= self.max_cache_bytes:
                while self._cache and self._bytes + out.nbytes > self.max_cache_bytes:
                    self._bytes -= self._cache.popitem(last=False)[1].nbytes
                    self._evictions += 1
                self._cache[key] = out
                self._bytes += out.nbytes
            return out

    def _materialize(self, field: str, axes: Axes, radiation: bool) -> NDArray[np.float64]:
        lon, lat, level, times = axes
        request = WeatherRequest(
            Domain(float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())),
            times.min(),
            times.max(),
            (1.0, 1.0) if radiation else (float(level.min() * 100), float(level.max() * 100)),
        )
        if radiation:
            if self.radiation is None:
                raise ValueError("radiation source is not configured")
            cube = materialize_radiation(self.radiation, request, fields=(field,))
            native: Axes = (
                np.asarray(cube.lon),
                np.asarray(cube.lat),
                np.asarray(cube.time_ns).astype("datetime64[ns]"),
            )
            return np.array(
                _select(native, (lon, lat, times), cube.fields[field])[:, :, None, :], copy=True
            )
        tau = field == "tau_cirrus" and field not in self.source.fields
        weather = materialize_weather(
            self.source, request, fields=("t",) if tau else (field,), tau_cirrus=tau
        )
        native = (
            np.asarray(weather.lon),
            np.asarray(weather.lat),
            np.asarray(weather.level_hpa),
            np.asarray(weather.time_ns).astype("datetime64[ns]"),
        )
        return _select(native, axes, weather.fields[field])

    def _dataset(self, start: np.datetime64, end: np.datetime64, radiation: bool) -> Any:
        source: Any = self.radiation if radiation else self.source
        grid = source.grid if radiation else source
        request = WeatherRequest(Domain.whole_globe(), start, end, (1.0, 1.0))
        origin = getattr(source, "time_origin", np.datetime64("1970-01-01"))
        times = _times(request, origin, source.time_step, source.coverage())
        coords = {
            "longitude": _longitude(grid.lon),
            "latitude": np.sort(np.asarray(grid.lat, dtype=np.float64)),
            "level": np.array([-1.0])
            if radiation
            else np.asarray(self.source.vertical.ids, dtype=np.float64),
            "time": times,
        }
        specs = (
            {name: RAD_FIELDS[name] for name in source.variables}
            if radiation
            else {name: MET_FIELDS[name] for name in self.fields}
        )
        variables = {
            spec[0]: lazy_variable(coords, partial(self._read, name, radiation=radiation), spec[2])
            for name, spec in specs.items()
        }
        xr = import_module("xarray")
        ds = xr.Dataset(variables, coords=coords, attrs=_attributes(grid, self.provenance))
        if radiation:
            ds.time.attrs["shift_radiation_time"] = "already resolved by source"
        return import_module("pycontrails").MetDataset(ds, wrap_longitude=False)

    def datasets(self, start: np.datetime64, end: np.datetime64) -> tuple[Any, Any | None]:
        """Construct metadata-only lazy datasets covering the requested UTC interval.

        Times are bracketed with each source's own cadence and origin. Spatial
        and pressure axes remain native; a global longitude axis is closed at
        its periodic seam. An omitted radiation source returns ``None``.
        """
        return self._dataset(start, end, False), self._dataset(
            start, end, True
        ) if self.radiation is not None else None

    def cache_info(self) -> dict[str, Any]:
        """Bounded retention and cumulative successful field materialization counters."""
        with self._lock:
            return {
                "bytes": self._bytes,
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "max_bytes": self.max_cache_bytes,
                "materialization_seconds": self._seconds,
                "field_reads": dict(self._field_reads),
            }

    def clear(self) -> None:
        """Drop bridge-owned arrays; consumer/xarray caches may retain their copies."""
        with self._lock:
            self._cache.clear()
            self._bytes = 0
