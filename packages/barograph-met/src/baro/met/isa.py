"""ICAO standard atmosphere: pressure altitude <-> static pressure.

ADS-B/Mode S barometric altitude is a *pressure altitude* (standard setting,
1013.25 hPa), so converting it with the ISA is exact by definition — no QNH, no
surface pressure and no temperature assumption is involved (ICAO Doc 7488).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["FT", "altitude_from_pressure", "pressure_from_altitude", "pressure_from_altitude_ft"]

FT = 0.3048  # m
_G0 = 9.80665
_R = 287.05287  # J kg-1 K-1 (ICAO)
_P0 = 101325.0
_T0 = 288.15

# (base geopotential altitude m, lapse rate K/m, top altitude m)
_LAYERS: tuple[tuple[float, float, float], ...] = (
    (0.0, -0.0065, 11000.0),
    (11000.0, 0.0, 20000.0),
    (20000.0, 0.0010, 32000.0),
)


def _layer_bases() -> list[tuple[float, float, float, float]]:
    bases = []
    t, p = _T0, _P0
    for h0, lapse, h1 in _LAYERS:
        bases.append((h0, t, lapse, p))
        dh = h1 - h0
        if lapse == 0.0:
            p = p * np.exp(-_G0 * dh / (_R * t))
        else:
            t1 = t + lapse * dh
            p = p * (t1 / t) ** (-_G0 / (lapse * _R))
            t = t1
    return bases


_BASES = _layer_bases()


def pressure_from_altitude(h_m: ArrayLike) -> NDArray[np.float64]:
    """Static pressure (Pa) for a pressure altitude in metres (valid up to 32 km)."""
    h = np.asarray(h_m, dtype=np.float64)
    p = np.full_like(h, np.nan)
    for i, (h0, t0, lapse, p0) in enumerate(_BASES):
        upper = _LAYERS[i][2]
        lower = -np.inf if i == 0 else h0
        m = (h >= lower) & (h < upper) if i < len(_BASES) - 1 else (h >= lower) & (h <= upper)
        dh = h[m] - h0
        if lapse == 0.0:
            p[m] = p0 * np.exp(-_G0 * dh / (_R * t0))
        else:
            p[m] = p0 * ((t0 + lapse * dh) / t0) ** (-_G0 / (lapse * _R))
    return p


def pressure_from_altitude_ft(h_ft: ArrayLike) -> NDArray[np.float64]:
    """Static pressure (Pa) for a pressure altitude in feet."""
    return pressure_from_altitude(np.asarray(h_ft, dtype=np.float64) * FT)


def altitude_from_pressure(p_pa: ArrayLike) -> NDArray[np.float64]:
    """Pressure altitude (m) for a static pressure in Pa (inverse of the above)."""
    p = np.asarray(p_pa, dtype=np.float64)
    h = np.full_like(p, np.nan)
    for i, (h0, t0, lapse, p0) in enumerate(_BASES):
        p_top = _BASES[i + 1][3] if i + 1 < len(_BASES) else 0.0
        m = (p <= p0) & (p > p_top) if i > 0 else (p > p_top)
        if lapse == 0.0:
            h[m] = h0 - _R * t0 / _G0 * np.log(p[m] / p0)
        else:
            h[m] = h0 + t0 / lapse * ((p[m] / p0) ** (-lapse * _R / _G0) - 1.0)
    return h
