"""Geographic domains and regional cubes of gridded fields."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from supertraj_met.base import CubeKey
from supertraj_met.vertical import VerticalCoordinate

__all__ = ["FULL_TURN", "Cube", "Domain", "GridIndex"]

FULL_TURN = 360.0  # degrees of longitude
LAT_LIMIT = 90.0
_PERIODIC_TOLERANCE = 1e-6
_MIN_GRID_POINTS = 2


@dataclass(frozen=True)
class Domain:
    """Latitude/longitude box; longitudes may be expressed beyond 180 (no wrap inside)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-LAT_LIMIT <= self.lat_min <= self.lat_max <= LAT_LIMIT):
            raise ValueError(f"invalid latitude range {self.lat_min}..{self.lat_max}")
        if not (0.0 <= self.lon_max - self.lon_min <= FULL_TURN):
            raise ValueError(f"invalid longitude range {self.lon_min}..{self.lon_max}")

    @classmethod
    def from_points(cls, lat: ArrayLike, lon: ArrayLike, margin: float = 0.5) -> Domain:
        """Smallest box around the points (shortest longitude arc), plus ``margin`` degrees."""
        la = np.asarray(lat, dtype=np.float64)
        lo = np.asarray(lon, dtype=np.float64)
        if la.size == 0:
            raise ValueError("no points")
        best: tuple[float, float] | None = None
        for base in (-180.0, 0.0):  # [-180, 180) and [0, 360) framings
            framed = base + np.mod(lo - base, 360.0)
            span = (float(framed.min()), float(framed.max()))
            if best is None or span[1] - span[0] < best[1] - best[0]:
                best = span
        assert best is not None
        lon_min, lon_max = best[0] - margin, best[1] + margin
        if lon_max - lon_min > FULL_TURN:
            lon_min, lon_max = -180.0, 180.0
        return cls(
            max(float(la.min()) - margin, -90.0),
            min(float(la.max()) + margin, 90.0),
            lon_min,
            lon_max,
        )

    def to_json(self) -> dict[str, float]:
        """Bounds as a JSON-friendly dict."""
        return {k: getattr(self, k) for k in ("lat_min", "lat_max", "lon_min", "lon_max")}

    @classmethod
    def whole_globe(cls) -> Domain:
        """The domain covering the entire globe."""
        return cls(-90.0, 90.0, 0.0, 360.0)

    def encloses(self, other: Domain) -> bool:
        """True if ``other`` lies inside this box (longitudes compared modulo 360)."""
        if not (self.lat_min <= other.lat_min and other.lat_max <= self.lat_max):
            return False
        if self.lon_max - self.lon_min >= FULL_TURN:
            return True
        start = self.lon_min + (other.lon_min - self.lon_min) % 360.0
        return start + (other.lon_max - other.lon_min) <= self.lon_max


@dataclass(frozen=True)
class GridIndex:
    """Where a domain sits in a regular grid (latitude descending, longitude ascending).

    A grid spanning 360 degrees of longitude is periodic (columns wrap around);
    a regional grid is clipped: the part of the domain outside it is dropped and
    points there are later flagged ``OUTSIDE_DOMAIN``.
    """

    rows: slice
    cols: NDArray[np.int64]  # indices into the global longitude axis (may wrap)
    lat: NDArray[np.float64]
    lon: NDArray[np.float64]  # unwrapped, strictly increasing

    @classmethod
    def locate(
        cls, domain: Domain, grid_lat: NDArray[np.float64], grid_lon: NDArray[np.float64]
    ) -> GridIndex:
        """Rows, columns and coordinates of ``domain`` in a regular grid."""
        dlat = float(grid_lat[1] - grid_lat[0])
        dlon = float(grid_lon[1] - grid_lon[0])
        if dlat >= 0 or dlon <= 0:
            raise ValueError("expected descending latitudes and ascending longitudes")
        nlat, nlon = len(grid_lat), len(grid_lon)
        i0 = int(np.floor((domain.lat_max - grid_lat[0]) / dlat + 1e-9))
        i1 = int(np.ceil((domain.lat_min - grid_lat[0]) / dlat - 1e-9))
        i0, i1 = max(i0, 0), min(max(i1, i0 + 1), nlat - 1)
        i0 = min(i0, i1 - 1)
        periodic = abs(nlon * dlon - FULL_TURN) < _PERIODIC_TOLERANCE
        lon0 = float(grid_lon[0])
        # periodic: keep the caller's framing (columns wrap); regional: express the
        # domain start in the grid frame [lon0, lon0 + 360)
        start = domain.lon_min if periodic else lon0 + (domain.lon_min - lon0) % 360.0
        stop = start + (domain.lon_max - domain.lon_min)
        j0 = int(np.floor((start - lon0) / dlon + 1e-9))
        j1 = int(np.ceil((stop - lon0) / dlon - 1e-9))
        if periodic:
            j1 = min(max(j1, j0 + 1), j0 + nlon)
        else:
            j0, j1 = max(j0, 0), min(j1, nlon - 1)
            if j1 - j0 < 1:  # domain outside the grid: keep a 2-column sliver (all flagged)
                j0, j1 = (nlon - 2, nlon - 1) if j0 >= nlon - 1 else (0, 1)
        j = np.arange(j0, j1 + 1, dtype=np.int64)
        return cls(
            rows=slice(i0, i1 + 1),
            cols=np.mod(j, nlon),
            lat=np.asarray(grid_lat[i0 : i1 + 1], dtype=np.float64),
            lon=(float(grid_lon[0]) + j * dlon).astype(np.float64),
        )


@dataclass
class Cube:
    """One hour of gridded fields on a regional grid, levels top to bottom."""

    source: str
    hour: np.datetime64
    lat: NDArray[np.float64]  # descending
    lon: NDArray[np.float64]  # unwrapped, ascending
    vertical: VerticalCoordinate
    fields: dict[str, NDArray[np.float32]]  # (level, lat, lon)
    ps: NDArray[np.float32] | None = None  # surface pressure (Pa), (lat, lon)
    # request the levels were selected for: pressures (Pa) and spare levels per side
    p_range: tuple[float, float] = (0.0, float("inf"))
    level_margin: int = 0
    run: np.datetime64 | None = None  # forecast base time (None for analyses)
    request_domain: Domain | None = None  # domain asked for (may exceed a regional grid)

    @property
    def key(self) -> CubeKey:
        """Cache key (validity time and run) of this cube."""
        return CubeKey(np.datetime64(self.hour, "h"), self.run)

    def __post_init__(self) -> None:
        shape = (len(self.vertical), len(self.lat), len(self.lon))
        for name, arr in self.fields.items():
            if arr.shape != shape:
                raise ValueError(f"field {name} has shape {arr.shape}, expected {shape}")
        if self.ps is not None and self.ps.shape != shape[1:]:
            raise ValueError("surface pressure grid does not match the fields")
        if len(self.lat) < _MIN_GRID_POINTS or len(self.lon) < _MIN_GRID_POINTS:
            raise ValueError("a cube needs at least 2x2 grid points")

    @property
    def nbytes(self) -> int:
        """Memory held by the fields and surface pressure."""
        total = sum(a.nbytes for a in self.fields.values())
        return total + (0 if self.ps is None else self.ps.nbytes)

    def contains(self, domain: Domain) -> bool:
        """True if the cube grid encloses ``domain``."""
        lat_ok = self.lat[-1] <= domain.lat_min and domain.lat_max <= self.lat[0]
        lon0 = float(self.lon[0])
        start = lon0 + (domain.lon_min - lon0) % 360.0
        return bool(lat_ok and start + (domain.lon_max - domain.lon_min) <= self.lon[-1])

    def surface_pressure_range(self) -> tuple[float, float] | None:
        """Minimum and maximum surface pressure (Pa), if known."""
        if self.ps is None or not np.isfinite(self.ps).any():
            return None
        return float(np.nanmin(self.ps)), float(np.nanmax(self.ps))

    # -- persistence -----------------------------------------------------------------------
    def manifest(self) -> dict[str, Any]:
        """JSON description written next to the cached arrays."""
        return {
            "source": self.source,
            "hour": str(self.hour),
            "lat": [float(self.lat[0]), float(self.lat[-1]), len(self.lat)],
            "lon": [float(self.lon[0]), float(self.lon[-1]), len(self.lon)],
            "vertical": self.vertical.kind,
            "levels": [int(i) for i in self.vertical.ids],
            "fields": sorted(self.fields),
            "has_ps": self.ps is not None,
            "p_range": list(self.p_range),
            "level_margin": self.level_margin,
            "run": None if self.run is None else str(self.run),
            "domain": None if self.request_domain is None else self.request_domain.to_json(),
        }

    def save(self, path: Path) -> None:
        """Atomically write ``path`` (.npz) then its ``.json`` manifest (commit marker)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, NDArray[Any]] = {f"field_{k}": v for k, v in self.fields.items()}
        arrays |= {
            "lat": self.lat,
            "lon": self.lon,
            "level_ids": self.vertical.ids,
            "level_a": self.vertical.a,
            "level_b": self.vertical.b,
        }
        if self.ps is not None:
            arrays["ps"] = self.ps
        _atomic_write(path, lambda fh: np.savez(fh, **arrays))  # type: ignore[arg-type]
        text = json.dumps(self.manifest(), indent=1).encode()
        _atomic_write(path.with_suffix(".json"), lambda fh: fh.write(text))

    @classmethod
    def load(cls, path: Path) -> Cube:
        """Read a cube written by :meth:`save`."""
        manifest = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as z:
            fields = {k.removeprefix("field_"): z[k] for k in z.files if k.startswith("field_")}
            vertical = VerticalCoordinate(
                manifest["vertical"], z["level_ids"], z["level_a"], z["level_b"]
            )
            ps = z["ps"] if "ps" in z.files else None
            return cls(
                source=manifest["source"],
                hour=np.datetime64(manifest["hour"], "h"),
                lat=z["lat"],
                lon=z["lon"],
                vertical=vertical,
                fields=fields,
                ps=ps,
                p_range=(float(manifest["p_range"][0]), float(manifest["p_range"][1])),
                level_margin=int(manifest["level_margin"]),
                run=None if manifest.get("run") is None else np.datetime64(manifest["run"], "h"),
                request_domain=None
                if manifest.get("domain") is None
                else Domain(**manifest["domain"]),
            )

    def serves(self, domain: Domain, p_range: tuple[float, float], level_margin: int) -> bool:
        """True if this cube answers a request for ``domain`` and ``p_range``.

        Levels are selected deterministically from the request pressure range
        and the cube's surface-pressure range, which can only shrink on a
        sub-domain: a wider original request therefore implies enough levels.
        """
        region_ok = (
            self.request_domain.encloses(domain)
            if self.request_domain is not None
            else self.contains(domain)
        )
        return (
            region_ok
            and self.p_range[0] <= p_range[0]
            and p_range[1] <= self.p_range[1]
            and self.level_margin >= level_margin
        )


def _atomic_write(path: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
