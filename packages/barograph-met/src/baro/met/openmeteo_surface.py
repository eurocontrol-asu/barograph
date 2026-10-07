"""Surface weather from the selectively read Open-Meteo stitched archive.

The archive preserves valid times, not original run identities or issue times.
Use a run-aware source to compare forecasts available before a cutoff. Native
wind components are instantaneous model values at 10 m, not hourly means.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain, GridIndex
from baro.met.openmeteo import (
    OPENMETEO_BASE_URL,
    ArchiveCatalog,
    TileReader,
    _ArchiveSource,
    gather_tiles,
)
from baro.met.store import TransferStats
from baro.met.surface import SurfaceCube, SurfaceEnricher, SurfaceResult, SurfaceVariable

if TYPE_CHECKING:
    from baro.met.rangecache import RangeCache

__all__ = ["OPENMETEO_SURFACE_VARIABLES", "OpenMeteoSurfaceSource"]

# Public variable -> (native archive folders, offset, metadata).
# Rain is derived from already-deaccumulated precipitation and snow water equivalent.
OPENMETEO_SURFACE_VARIABLES: dict[str, tuple[tuple[str, ...], float, SurfaceVariable]] = {
    "u10": (("wind_u_component_10m",), 0.0, SurfaceVariable("m/s", 10.0)),
    "v10": (("wind_v_component_10m",), 0.0, SurfaceVariable("m/s", 10.0)),
    "t2m": (("temperature_2m",), 273.15, SurfaceVariable("K", 2.0)),
    "precipitation": (("precipitation",), 0.0, SurfaceVariable("mm", 0.0, "sum")),
    "snowfall_water_equivalent": (
        ("snowfall_water_equivalent",),
        0.0,
        SurfaceVariable("mm", 0.0, "sum"),
    ),
    "rain": (
        ("precipitation", "snowfall_water_equivalent"),
        0.0,
        SurfaceVariable("mm", 0.0, "sum"),
    ),
    "gusts": (("wind_gusts_10m",), 0.0, SurfaceVariable("m/s", 10.0, "max")),
}
_DEFAULT_FIELDS = ("u10", "v10", "t2m", "precipitation", "rain", "gusts")


class OpenMeteoSurfaceSource(_ArchiveSource):
    """Native surface fields of an Open-Meteo archived model.

    Parameters
    ----------
    model:
        Native folder, default ``meteofrance_arome_france0025`` (regional 2.5 km).
    fields:
        Output name -> key in OPENMETEO_SURFACE_VARIABLES. Default u10/v10/t2m,
        precipitation, rain and gusts. Only these variables/dependencies are read.
    base_url:
        Public bucket or local mirror. Archive forecasts extend to metadata coverage.
    tile_deg:
        Geographic tile edge in degrees (same caching mechanism as pressure fields).
    cache_mb:
        Tile LRU memory budget in megabytes.

    Notes
    -----
    Rain is ``max(precipitation - snowfall_water_equivalent, 0)`` at each native
    grid point. Both archived amounts are already totals over the preceding
    native period; never difference them again. Gusts are the preceding-period
    maximum. Fetch returns native fields; SurfaceEnricher preserves these periods.
    Heights are above model terrain; there is no terrain/bias correction.
    Run/lead times are unavailable for this stitched archive.
    """

    def __init__(
        self,
        model: str = "meteofrance_arome_france0025",
        *,
        fields: dict[str, str] | None = None,
        base_url: str = OPENMETEO_BASE_URL,
        tile_deg: float = 5.0,
        cache_mb: float = 1024.0,
    ) -> None:
        self.model = model
        self.fields = dict(fields if fields is not None else {n: n for n in _DEFAULT_FIELDS})
        if not self.fields:
            raise ValueError("at least one surface field is required")
        unknown = set(self.fields.values()) - set(OPENMETEO_SURFACE_VARIABLES)
        if unknown:
            raise ValueError(f"unknown surface variables {sorted(unknown)}")
        if not np.isfinite(tile_deg) or tile_deg <= 0 or not np.isfinite(cache_mb) or cache_mb <= 0:
            raise ValueError("tile_deg and cache_mb must be positive and finite")
        self.variables = {
            out: OPENMETEO_SURFACE_VARIABLES[name][2] for out, name in self.fields.items()
        }
        self._native = sorted(
            {v for name in self.fields.values() for v in OPENMETEO_SURFACE_VARIABLES[name][0]}
        )
        self.source_id = f"openmeteo-surface-{model}"
        self._stats = TransferStats()
        self.catalog = ArchiveCatalog(base_url, model, self._native, self._stats)
        # Folder availability, not pressure levels, defines surface support.
        available = self.catalog.available_variables
        missing = set(self._native) - set(available)
        if missing:
            raise ValueError(f"{model}: surface variables not archived: {sorted(missing)}")
        self.time_step = self.catalog.time_step
        self.reader = TileReader(self.catalog, self._stats, tile_deg, cache_mb)
        self.reader.final = self._final_path
        self._probe = (self._native[0], 0)  # level zero selects the surface folder

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        """Common native endpoint coverage (oldest chunks may be partly empty)."""
        first = max(self.catalog.locate_first_chunk(v, 0) for v in self._native)
        return np.datetime64(first * self.catalog.chunk_seconds, "s"), self.catalog.end

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        """Read one native timestamp over the requested domain, without pressure."""
        selected = self.fields if fields is None else {n: self.fields[n] for n in fields}
        if not selected:
            raise ValueError("at least one surface field is required")
        native_names = sorted(
            {v for name in selected.values() for v in OPENMETEO_SURFACE_VARIABLES[name][0]}
        )
        start, end = self.coverage()
        if not start <= key.valid <= end:
            raise DataUnavailableError(f"{self.source_id}: {key.valid} outside {start}..{end}")
        index = self.catalog.time_index(key.valid)
        if key.valid != np.datetime64(index * self.catalog.step_seconds, "s"):
            raise ValueError(f"{key.valid} is not aligned to {self.time_step}")
        if key.run is not None:
            raise ValueError("stitched archives do not retain forecast runs")
        chunk, t_in = divmod(index, self.catalog.chunk_steps)
        nlat, _ = self.reader.ensure_shape(*self._probe)
        lat, lon = self.reader.grid()
        grid = GridIndex.locate(domain, lat, lon)
        rows = nlat - 1 - np.arange(grid.rows.start, grid.rows.stop)
        blocks = self.reader.blocks([(v, 0) for v in native_names], chunk, rows, grid.cols)
        tp = self.reader.tile_points()
        native = {v: gather_tiles(blocks[(v, 0)], rows, grid.cols, tp, t_in) for v in native_names}
        values: dict[str, NDArray[np.float32]] = {}
        for out, name in selected.items():
            dependencies, offset, _ = OPENMETEO_SURFACE_VARIABLES[name]
            if name == "rain":
                value = np.maximum(native[dependencies[0]] - native[dependencies[1]], 0)
            else:
                value = native[dependencies[0]] + offset
            values[out] = np.asarray(value, dtype=np.float32)
        return SurfaceCube(
            key,
            grid.lat[::-1].copy(),
            grid.lon,
            {n: a[::-1].copy() for n, a in values.items()},
            {n: self.variables[n] for n in selected},
        )

    def enrich(
        self, time: ArrayLike, lat: ArrayLike, lon: ArrayLike, *, cache: RangeCache | None = None
    ) -> SurfaceResult:
        """Convenience query; use SurfaceEnricher for a persistent cache directory."""
        if cache is not None:
            self.use_cache(cache)
        return SurfaceEnricher(self).enrich(time, lat, lon)
