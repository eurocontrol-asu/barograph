"""Targeted ERA5 top-of-atmosphere fluxes from the ARCO pressure-level store."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain, GridIndex
from baro.met.rangecache import RangeCache
from baro.met.sources import ARCO_BASE_URL, GridSource, arco_pressure_levels
from baro.met.store import TransferStats
from baro.met.surface import SurfaceCube, SurfaceVariable
from baro.met.tiling import Tiling

__all__ = ["ArcoRadiationSource"]

_FIELDS = ("top_net_solar_radiation", "top_net_thermal_radiation")
_HALF_HOUR = np.timedelta64(30, "m")


def _flux_scale(source: GridSource, name: str) -> float:
    meta = source.store.array(name)
    if meta.attrs.get("_ARRAY_DIMENSIONS") != ["time", "latitude", "longitude"]:
        raise ValueError(f"{name}: expected time, latitude, longitude dimensions")
    unit = meta.attrs.get("units")
    if unit in {"J m**-2", "J m-2"}:
        return 1.0 / 3600.0
    if unit in {"W m**-2", "W m-2"}:
        return 1.0
    raise ValueError(f"{name}: unsupported radiation unit {unit!r}")


class ArcoRadiationSource:
    """Hourly ERA5 flux averages, represented at interval midpoints in W/m².

    ARCO stores one-hour energies (J/m²) at interval ends. This source divides
    them by 3600 and moves the interpolation coordinate back 30 minutes,
    exactly once. It preserves ERA5's signed net-flux convention. These are
    top-of-atmosphere fields, not surface irradiance. Only requested fields
    and occupied latitude blocks are read through the existing range reader.

    ``grid_source`` permits local mirrors/tests; it must use the ERA5 ARCO
    time and spatial axes. The default opens the public pressure-level store.
    """

    time_step = np.timedelta64(1, "h")
    time_origin = np.datetime64("1970-01-01T00:30", "ns")

    def __init__(
        self,
        *,
        fields: Sequence[str] = _FIELDS,
        base_url: str = ARCO_BASE_URL,
        grid_source: GridSource | None = None,
    ) -> None:
        names = tuple(fields)
        if not names or len(set(names)) != len(names) or set(names) - set(_FIELDS):
            raise ValueError("select unique ERA5 top net solar/thermal radiation fields")
        self.grid = grid_source or arco_pressure_levels(
            base_url=base_url, fields={name: name for name in names}
        )
        if any(self.grid.fields.get(name) != name for name in names):
            raise ValueError("grid_source must be configured with the requested radiation arrays")
        self.grid = self.grid.select_fields(names)
        # TOA is a radiative boundary, not a finite height above ground.
        self.variables = {name: SurfaceVariable("W m-2", float("nan")) for name in names}
        self._scale = {name: _flux_scale(self.grid, name) for name in names}

    @property
    def stats(self) -> list[TransferStats]:
        """Underlying transfer counters, including byte-cache hits."""
        return self.grid.stats

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        """Effective interpolation coverage after the ERA5 midpoint shift."""
        bounds = self.grid.coverage()
        if bounds is None:
            raise DataUnavailableError("ERA5 radiation coverage is unavailable")
        return bounds[0] - _HALF_HOUR, bounds[1] - _HALF_HOUR

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Midpoint samples bracketing the actual query time."""
        end = np.datetime64(t0 + _HALF_HOUR, "h")
        return CubeKey(end - _HALF_HOUR), CubeKey(end + _HALF_HOUR)

    def tiling(self, key: CubeKey) -> Tiling | None:
        """Block-sized latitude bands shared with the range reader."""
        return self.grid.tiling(CubeKey(key.valid + _HALF_HOUR))

    def use_cache(self, cache: RangeCache) -> None:
        """Cache only immutable ERA5 archive ranges."""
        self.grid.use_cache(cache)

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        """Read a midpoint field without downloading unrelated pressure variables."""
        names = tuple(self.variables if fields is None else fields)
        if set(names) - self.variables.keys():
            raise ValueError("requested radiation field is not configured")
        end = key.valid + _HALF_HOUR
        if end != np.datetime64(end, "h") or key.run is not None:
            raise ValueError("ERA5 radiation keys must be hourly interval midpoints, without run")
        try:
            index = self.grid.time_index(end)
        except KeyError as exc:
            raise DataUnavailableError(f"ERA5 radiation: {end} outside archive") from exc
        grid = GridIndex.locate(domain, self.grid.lat, self.grid.lon)
        values = {
            name: (
                self.grid.store.read_rows(name, [(index,)], grid.rows)[0][:, grid.cols][::-1]
                * self._scale[name]
            ).astype(np.float32)
            for name in names
        }
        return SurfaceCube(
            key, grid.lat[::-1], grid.lon, values, {n: self.variables[n] for n in names}
        )
