"""Interpolation of a regional cube at aircraft positions.

For each of the 4 horizontal corners around a point, the full-level pressures
of that grid column are computed from *its own* surface pressure
(``p_k = A_k + B_k * ps``), the two levels bracketing the aircraft pressure are
found, and the field is interpolated linearly in ``ln p`` (default) or ``p``.
The 4 column values are then combined bilinearly. Time interpolation between
two hourly cubes is done by the caller. Nothing is ever extrapolated: points
outside the domain or outside a column get NaN and a flag.
"""

from __future__ import annotations

from enum import IntFlag
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from supertraj_met.domain import Cube
from supertraj_met.vertical import STANDARD_SURFACE_PRESSURE

__all__ = ["Flag", "SampleResult", "sample", "track_components"]

type VerticalMethod = Literal["log", "linear"]
type SurfacePressureMode = Literal["actual", "standard"]


class Flag(IntFlag):
    """Bit flags describing why a value is missing."""

    OK = 0
    INVALID_INPUT = 1  # non-finite time/position/pressure
    OUTSIDE_DOMAIN = 2  # outside the cube's horizontal grid
    ABOVE_TOP = 4  # above the highest retrieved level
    BELOW_BOTTOM = 8  # below the lowest retrieved level (near/below the surface)
    MISSING_DATA = 16  # NaN in the source data or cube not published
    OUT_OF_COVERAGE = 32  # time outside the period the source covers


type SampleResult = tuple[dict[str, NDArray[np.float64]], NDArray[np.uint8]]

_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))


def sample(
    cube: Cube,
    lat: ArrayLike,
    lon: ArrayLike,
    pressure: ArrayLike,
    *,
    vertical: VerticalMethod = "log",
    surface_pressure: SurfacePressureMode = "actual",
    batch: int = 32768,
) -> SampleResult:
    """Interpolate every cube field at the given positions (pressure in Pa)."""
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    n = la.size
    values = {name: np.full(n, np.nan) for name in cube.fields}
    flags = np.zeros(n, dtype=np.uint8)
    if n == 0:
        return values, flags
    if len(cube.vertical) < 2:
        flags[:] = Flag.ABOVE_TOP | Flag.BELOW_BOTTOM
        return values, flags
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        v, f = _sample_batch(cube, la[sl], lo[sl], p[sl], vertical, surface_pressure)
        for name in values:
            values[name][sl] = v[name]
        flags[sl] = f
    return values, flags


def _horizontal(
    cube: Cube, lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    nlat, nlon = len(cube.lat), len(cube.lon)
    dlat = cube.lat[1] - cube.lat[0]
    dlon = cube.lon[1] - cube.lon[0]
    lon_framed = cube.lon[0] + np.mod(lon - cube.lon[0], 360.0)
    fi = (lat - cube.lat[0]) / dlat
    fj = (lon_framed - cube.lon[0]) / dlon
    eps = 1e-9
    outside = ~((fi >= -eps) & (fi <= nlat - 1 + eps) & (fj >= -eps) & (fj <= nlon - 1 + eps))
    fi = np.clip(np.nan_to_num(fi), 0.0, nlat - 1)
    fj = np.clip(np.nan_to_num(fj), 0.0, nlon - 1)
    i0 = np.minimum(np.floor(fi).astype(np.int64), nlat - 2)
    j0 = np.minimum(np.floor(fj).astype(np.int64), nlon - 2)
    return i0, j0, fi - i0, fj - j0, outside


def _sample_batch(
    cube: Cube,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    p: NDArray[np.float64],
    vertical: VerticalMethod,
    surface_pressure: SurfacePressureMode,
) -> SampleResult:
    n = lat.size
    invalid = ~(np.isfinite(lat) & np.isfinite(lon) & np.isfinite(p) & (p > 0))
    i0, j0, wi, wj, outside = _horizontal(cube, lat, lon)
    transform = np.log if vertical == "log" else (lambda x: x)
    safe_p = np.where(invalid, STANDARD_SURFACE_PRESSURE, p)
    tp = transform(safe_p)
    a, b = cube.vertical.a, cube.vertical.b
    nlev = len(a)
    rows = np.arange(n)
    acc = {name: np.zeros(n) for name in cube.fields}
    above = np.zeros(n, dtype=bool)
    below = np.zeros(n, dtype=bool)
    missing = np.zeros(n, dtype=bool)
    use_ps = surface_pressure == "actual" and cube.ps is not None
    for di, dj in _CORNERS:
        ii, jj = i0 + di, j0 + dj
        weight = (wi if di else 1.0 - wi) * (wj if dj else 1.0 - wj)
        ps = cube.ps[ii, jj].astype(np.float64) if use_ps else np.full(n, STANDARD_SURFACE_PRESSURE)  # type: ignore[index]
        missing |= ~np.isfinite(ps)
        pcol = a[None, :] + b[None, :] * np.nan_to_num(ps, nan=STANDARD_SURFACE_PRESSURE)[:, None]
        k = np.count_nonzero(pcol < safe_p[:, None], axis=1)
        k = np.where((k == 0) & (safe_p == pcol[:, 0]), 1, k)
        above |= k == 0
        below |= k == nlev
        lo_k = np.clip(k - 1, 0, nlev - 2)
        hi_k = lo_k + 1
        t_lo = transform(pcol[rows, lo_k])
        t_hi = transform(pcol[rows, hi_k])
        wv = (tp - t_lo) / (t_hi - t_lo)
        for name, field in cube.fields.items():
            f_lo = field[lo_k, ii, jj].astype(np.float64)
            f_hi = field[hi_k, ii, jj].astype(np.float64)
            acc[name] += weight * ((1.0 - wv) * f_lo + wv * f_hi)
    nan_values = np.zeros(n, dtype=bool)
    for name in acc:
        nan_values |= ~np.isfinite(acc[name])
    flags = _compose_flags(invalid, outside, above, below, missing | nan_values)
    for name in acc:
        acc[name][flags != 0] = np.nan
    return acc, flags


def _compose_flags(
    invalid: NDArray[np.bool_],
    outside: NDArray[np.bool_],
    above: NDArray[np.bool_],
    below: NDArray[np.bool_],
    missing: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Most fundamental cause first: an invalid or outside point gets no other flag."""
    flags = np.where(invalid, int(Flag.INVALID_INPUT), 0)
    flags = np.where(~invalid & outside, int(Flag.OUTSIDE_DOMAIN), flags)
    located = ~(invalid | outside)
    flags |= np.where(located & above, int(Flag.ABOVE_TOP), 0)
    flags |= np.where(located & below, int(Flag.BELOW_BOTTOM), 0)
    flags = np.where((flags == 0) & missing, int(Flag.MISSING_DATA), flags)
    return flags.astype(np.uint8)


def track_components(
    u: ArrayLike, v: ArrayLike, track_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Along-track (tailwind > 0) and cross-track (from left to right > 0) wind."""
    chi = np.deg2rad(np.asarray(track_deg, dtype=np.float64))
    ua = np.asarray(u, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    return ua * np.sin(chi) + va * np.cos(chi), ua * np.cos(chi) - va * np.sin(chi)
