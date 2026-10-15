"""Validated weather grids and optional contrail-engine exports.

These containers do not download, extrapolate, derive cirrus optical depth or
convert accumulated radiation. Coordinates are effective interpolation times
in UTC nanoseconds; callers resolve source-specific time conventions first.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["RadiationCube", "WeatherCube"]

Array = npt.NDArray[np.float64]
MET_FIELDS = {
    "u": ("eastward_wind", "m s-1", "m s**-1"),
    "v": ("northward_wind", "m s-1", "m s**-1"),
    "t": ("air_temperature", "K", "K"),
    "q": ("specific_humidity", "kg kg-1", "kg kg**-1"),
    "omega": ("lagrangian_tendency_of_air_pressure", "Pa s-1", "Pa s**-1"),
    "tau_cirrus": ("tau_cirrus", "1", "1"),
    "ciwc": ("specific_cloud_ice_water_content", "kg kg-1", "kg kg**-1"),
    "geopotential": ("geopotential", "m2 s-2", "m**2 s**-2"),
}
RAD_FIELDS = {
    name: (name, "W m-2", "W m**-2")
    for name in ("top_net_solar_radiation", "top_net_thermal_radiation")
}
COCIP_FIELDS = frozenset(("u", "v", "t", "q", "omega", "tau_cirrus"))
_EPOCH_NS = int(np.datetime64("2000-01-01", "ns").astype(np.int64))
_DAY_NS = 86_400_000_000_000
_MAX_LATITUDE = 90
_MIN_INTERPOLATION_POINTS = 2


def _axis(values: npt.ArrayLike, name: str) -> Array:
    a = np.array(values, dtype=np.float64, copy=True)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional coordinate")
    if not np.all(a[1:] > a[:-1]):
        raise ValueError(f"{name} must be strictly ascending")
    return a


def _times(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    a = np.asarray(values)
    if a.dtype.kind != "i" or a.ndim != 1 or not a.size:
        raise ValueError("time_ns must contain one-dimensional signed UTC Unix nanoseconds")
    a = np.array(a, dtype=np.int64, copy=True)
    if np.any(a == np.iinfo(np.int64).min) or not np.all(a[1:] > a[:-1]):
        raise ValueError("time_ns must be strictly ascending and must not contain NaT")
    return a


def _fields(
    fields: Mapping[str, npt.ArrayLike],
    units: Mapping[str, str],
    shape: tuple[int, ...],
    specs: Mapping[str, tuple[str, str, str]],
) -> dict[str, Array]:
    out = {}
    if not fields or set(fields) != set(units):
        raise ValueError("fields and units must have identical nonempty keys")
    for name, values in fields.items():
        if name not in specs or units[name] not in specs[name][1:]:
            raise ValueError(f"Unsupported field or units: {name} ({units[name]})")
        a = np.array(values, dtype=np.float64, copy=True)
        if a.shape != shape or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must have shape {shape} and only finite values")
        out[name] = a
    return out


@dataclass(kw_only=True)
class RadiationCube:
    """TOA flux grid with explicitly resolved timestamps.

    Arrays have dimensions ``(lon, lat, time)``. ``time_semantics`` must be
    ``valid`` (instantaneous flux) or ``interval_midpoint`` (mean flux assigned
    to its averaging-interval midpoint). Interval-end or accumulated radiation
    must be converted upstream. No engine export shifts these timestamps.
    """

    lon: npt.ArrayLike
    lat: npt.ArrayLike
    time_ns: npt.ArrayLike
    fields: Mapping[str, npt.ArrayLike]
    units: Mapping[str, str]
    provenance: Mapping[str, str]
    time_semantics: str

    def __post_init__(self) -> None:
        """Validate axes, shape, units, and timestamp semantics."""
        self.lon = _axis(self.lon, "longitude")
        self.lat = _axis(self.lat, "latitude")
        if np.any(np.abs(self.lat) > _MAX_LATITUDE):
            raise ValueError("latitude must be in [-90, 90]")
        self.time_ns = _times(self.time_ns)
        if self.time_semantics not in {"valid", "interval_midpoint"}:
            raise ValueError("Resolve radiation times to valid or interval_midpoint first")
        self.fields = _fields(self.fields, self.units, self.shape, self.specs)
        self.units = {k: self.specs[k][1] for k in self.fields}
        self.provenance = dict(self.provenance)

    @property
    def specs(self) -> Mapping[str, tuple[str, str, str]]:
        """Recognized variables, canonical and engine units."""
        return RAD_FIELDS

    @property
    def shape(self) -> tuple[int, ...]:
        """Required array shape in export axis order."""
        return (np.size(self.lon), np.size(self.lat), np.size(self.time_ns))

    @property
    def coordinates(self) -> dict[str, npt.NDArray[Any]]:
        """Coordinates in native engine dimension order."""
        return {
            "longitude": np.asarray(self.lon),
            "latitude": np.asarray(self.lat),
            "time": np.asarray(self.time_ns).astype("datetime64[ns]"),
        }

    @property
    def coverage(self) -> dict[str, tuple[Any, Any]]:
        """Inclusive coordinate bounds, without implying extrapolation support."""
        return {k: (v[0], v[-1]) for k, v in self.coordinates.items()}

    def _require_complete(self) -> None:
        missing = set(RAD_FIELDS) - self.fields.keys()
        if missing:
            raise ValueError(f"Missing required radiation fields: {sorted(missing)}")

    def to_diffcocip(self) -> dict[str, Array]:
        """Export the NumPy dict accepted by diffcocip, without importing JAX.

        Every interpolation axis needs at least two points. The engine's time
        bridge uses seconds since 2000; grids before this epoch are rejected.
        Metadata stays available on this container, not in the engine dict.
        """
        self._require_complete()
        if min(self.shape) < _MIN_INTERPOLATION_POINTS:
            raise ValueError("diffcocip requires at least two points on every axis")
        time_ns = np.asarray(self.time_ns)
        if np.any(time_ns < _EPOCH_NS):
            raise ValueError("diffcocip requires timestamps on or after 2000-01-01")
        relative = time_ns - _EPOCH_NS
        out = {
            "lon": np.asarray(self.lon).copy(),
            "lat": np.asarray(self.lat).copy(),
            "time_s": relative / 1e9,
            "days_since_2000": relative / _DAY_NS,
            "hours_of_day": np.remainder(time_ns, _DAY_NS) / 3_600_000_000_000,
        }
        if "level" in self.coordinates:
            out["level"] = self.coordinates["level"].copy()
        out.update({self.specs[k][0]: np.asarray(v).copy() for k, v in self.fields.items()})
        return out

    def to_pycontrails(self, *, require_cocip: bool = True) -> Any:
        """Export an optional pycontrails MetDataset with resolved time metadata.

        Import pycontrails and xarray only on this call. Radiation's time
        coordinate is marked as resolved, preventing a second engine shift.
        Set ``require_cocip=False`` to export partial fields for preprocessing.
        """
        if require_cocip:
            self._require_complete()
        try:
            xr = importlib.import_module("xarray")
            pc = importlib.import_module("pycontrails")
        except ImportError as exc:
            raise ImportError("Install pycontrails and xarray to export MetDataset") from exc
        coords = self.coordinates
        dims = tuple(coords)
        data = {
            self.specs[k][0]: (dims, np.asarray(v).copy(), {"units": self.specs[k][2]})
            for k, v in self.fields.items()
        }
        ds = xr.Dataset(data, coords=coords, attrs=dict(self.provenance))
        if "level" not in coords:
            ds = ds.expand_dims(level=[-1])
            ds["time"].attrs["shift_radiation_time"] = "already resolved by caller"
        return pc.MetDataset(ds)


@dataclass(kw_only=True)
class WeatherCube(RadiationCube):
    """Pressure-level grid with dimensions ``(lon, lat, level_hpa, time)``.

    A cube may contain wind only; CoCiP exports require ``u, v, t, q, omega,
    tau_cirrus``. Optical depth must be supplied after a correct full-column
    calculation; it cannot safely be derived from a cropped pressure slab.
    """

    level_hpa: npt.ArrayLike
    time_semantics: str = "valid"

    def __post_init__(self) -> None:
        """Validate the pressure axis and inherited grid contract."""
        self.level_hpa = _axis(self.level_hpa, "level_hpa")
        if np.any(np.asarray(self.level_hpa) <= 0):
            raise ValueError("level_hpa must be positive")
        if self.time_semantics != "valid":
            raise ValueError("Pressure-level weather must use valid timestamps")
        super().__post_init__()

    @property
    def specs(self) -> Mapping[str, tuple[str, str, str]]:
        """Pressure-level fields and units."""
        return MET_FIELDS

    @property
    def shape(self) -> tuple[int, ...]:
        """Required array shape in export axis order."""
        return (
            np.size(self.lon),
            np.size(self.lat),
            np.size(self.level_hpa),
            np.size(self.time_ns),
        )

    @property
    def coordinates(self) -> dict[str, npt.NDArray[Any]]:
        """Engine coordinates, including pressure in hPa."""
        return {
            "longitude": np.asarray(self.lon),
            "latitude": np.asarray(self.lat),
            "level": np.asarray(self.level_hpa),
            "time": np.asarray(self.time_ns).astype("datetime64[ns]"),
        }

    def _require_complete(self) -> None:
        missing = COCIP_FIELDS - self.fields.keys()
        if missing:
            raise ValueError(f"Missing required CoCiP fields: {sorted(missing)}")
