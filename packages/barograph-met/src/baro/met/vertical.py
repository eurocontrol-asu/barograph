"""Vertical coordinates expressed uniformly as ``p_k = A_k + B_k * ps``.

* ERA5 model levels (L137, hybrid sigma-pressure, Simmons & Burridge 1981):
  half levels ``p_{k+1/2} = a_{k+1/2} + b_{k+1/2} * ps`` and full levels are the
  mean of the two adjacent half levels (IFS convention), hence
  ``A_k = (a_{k-1/2} + a_{k+1/2}) / 2`` and ``B_k`` likewise.
* Pressure levels: ``A_k = p_k``, ``B_k = 0``.

Levels are always ordered top (low pressure) to bottom (high pressure).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MIN_LEVELS",
    "N_MODEL_LEVELS",
    "STANDARD_SURFACE_PRESSURE",
    "VerticalCoordinate",
    "model_levels",
    "pressure_levels",
]

STANDARD_SURFACE_PRESSURE = 101325.0
N_MODEL_LEVELS = 137  # ERA5 / IFS L137
MIN_LEVELS = 2  # vertical interpolation needs a bracketing pair


@dataclass(frozen=True)
class VerticalCoordinate:
    """Full-level pressures ``A + B * ps`` for a set of native level identifiers."""

    kind: str  # "model" or "pressure"
    ids: NDArray[np.int64]  # native identifiers (hybrid index 1..137 or hPa)
    a: NDArray[np.float64]  # Pa
    b: NDArray[np.float64]  # dimensionless

    def __post_init__(self) -> None:
        if not (len(self.ids) == len(self.a) == len(self.b)):
            raise ValueError("ids, a and b must have the same length")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def needs_surface_pressure(self) -> bool:
        """True for hybrid levels (pressures depend on the surface pressure)."""
        return bool(np.any(self.b != 0.0))

    def pressure(self, ps: ArrayLike) -> NDArray[np.float64]:
        """Full-level pressure (Pa); ``ps`` broadcasts on a trailing level axis."""
        ps_arr = np.asarray(ps, dtype=np.float64)[..., None]
        return self.a + self.b * ps_arr

    def subset(self, positions: ArrayLike) -> VerticalCoordinate:
        """The levels at ``positions``."""
        pos = np.asarray(positions, dtype=np.int64)
        return VerticalCoordinate(self.kind, self.ids[pos], self.a[pos], self.b[pos])

    def select_bracketing(
        self, p_min: float, p_max: float, ps_min: float, ps_max: float, margin: int = 1
    ) -> NDArray[np.int64]:
        """Positions of the levels needed to bracket pressures in ``[p_min, p_max]``.

        The level pressures move with ``ps``: a level is kept if, for some
        ``ps`` in ``[ps_min, ps_max]``, it can be the level just above or just
        below a target pressure. ``margin`` extra levels are added on each side.
        """
        lo = self.pressure(ps_min)  # smallest pressure of each level (B >= 0)
        hi = self.pressure(ps_max)
        # first level that can lie at/above p_min, last that can lie at/below p_max
        top = int(np.searchsorted(hi, p_min, side="right")) - 1
        bottom = int(np.searchsorted(lo, p_max, side="left"))
        top = max(top - margin, 0)
        bottom = min(bottom + margin, len(self) - 1)
        return np.arange(top, bottom + 1, dtype=np.int64)


@cache
def _l137() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    text = resources.files("baro.met").joinpath("data/l137.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    a_half = np.array([float(r["a_pa"]) for r in rows])
    b_half = np.array([float(r["b"]) for r in rows])
    if a_half.shape != (138,):
        raise RuntimeError("corrupted L137 table")
    return a_half, b_half


def model_levels(ids: ArrayLike | None = None) -> VerticalCoordinate:
    """ERA5 L137 full levels (default: all 137, hybrid index 1 = model top)."""
    a_half, b_half = _l137()
    all_ids = np.arange(1, 138, dtype=np.int64)
    a_full = 0.5 * (a_half[:-1] + a_half[1:])
    b_full = 0.5 * (b_half[:-1] + b_half[1:])
    coord = VerticalCoordinate("model", all_ids, a_full, b_full)
    if ids is None:
        return coord
    sel = np.asarray(ids, dtype=np.int64)
    if np.any((sel < 1) | (sel > N_MODEL_LEVELS)) or np.any(np.diff(sel) <= 0):
        raise ValueError("model level ids must be strictly increasing within 1..137")
    return coord.subset(sel - 1)


def pressure_levels(hpa: ArrayLike) -> VerticalCoordinate:
    """Isobaric levels given in hPa (must be strictly increasing)."""
    lev = np.asarray(hpa, dtype=np.int64)
    if np.any(np.diff(lev) <= 0):
        raise ValueError("pressure levels must be strictly increasing (top to bottom)")
    return VerticalCoordinate("pressure", lev, lev.astype(np.float64) * 100.0, np.zeros(len(lev)))
