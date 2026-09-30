"""Geographic domains and regional cubes of gridded fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baro.met.vertical import VerticalCoordinate

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


def _regional_start(domain: Domain, lon0: float, dlon: float, nlon: int) -> float:
    """Start of ``domain`` in the frame of a regional grid ``[lon0, lon0 + (nlon - 1) dlon]``.

    A longitude box can be written with its start shifted by any multiple of
    360 degrees; the framing kept is the one covering most of the grid, so a
    box straddling the western edge of the grid is clipped, not sent a full
    turn east of it.
    """
    width = domain.lon_max - domain.lon_min
    if width >= FULL_TURN:  # every longitude: the whole grid
        return lon0
    last = lon0 + (nlon - 1) * dlon
    base = lon0 + (domain.lon_min - lon0) % FULL_TURN  # in [lon0, lon0 + 360)

    def overlap(start: float) -> float:
        """Length of grid covered by the box starting at ``start``."""
        return min(start + width, last) - max(start, lon0)

    return max((base, base - FULL_TURN), key=overlap)


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
        # periodic: keep the caller's framing (columns wrap); regional: the framing
        # of the domain that overlaps the grid most
        start = domain.lon_min if periodic else _regional_start(domain, lon0, dlon, nlon)
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
    run: np.datetime64 | None = None  # forecast base time (None for analyses)

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
