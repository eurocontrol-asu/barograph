"""Split the points of one time slice along the source's own horizontal tiling.

A source stores its fields in tiles (latitude bands of Blosc blocks for
ARCO-ERA5, square tiles for Open-Meteo). The points of one hour, possibly from
many flights, occupy some of them. Occupied tiles that touch form one
*component*; each component becomes its own cube, so the empty tiles between
two distant groups of flights are never read, and no threshold has to be
tuned: the source's tiling decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Tiling", "partition"]


@dataclass(frozen=True)
class Tiling:
    """Regular tiling of a grid: tile ``(i, j)`` starts at ``lat0 + i * lat_step``.

    ``lon_step=None`` means tiles span every longitude (ARCO-ERA5 bands).
    ``merge_adjacent=False`` groups points by their own tile, keeping long
    connected trajectories from requesting large mostly empty rectangles.
    """

    lat0: float
    lat_step: float
    lon0: float = -180.0
    lon_step: float | None = None
    merge_adjacent: bool = True

    def __post_init__(self) -> None:
        if self.lat_step <= 0 or (self.lon_step is not None and self.lon_step <= 0):
            raise ValueError("tile steps must be positive")

    @property
    def n_cols(self) -> int:
        """Number of tile columns around the globe."""
        return 1 if self.lon_step is None else int(np.ceil(360.0 / self.lon_step - 1e-9))


def partition(
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    margin: float,
    tiling: Tiling | None,
) -> NDArray[np.int64]:
    """Component label (0, 1, ...) of every point, from the tiles within ``margin`` of it.

    Components are ordered by their first tile (south to north, then west to
    east); a point near a tile edge marks both tiles, so the neighbours its
    interpolation needs are always in its own component.
    With ``merge_adjacent=False``, labels instead identify each point's own
    tile. The caller must retain its normal domain margin and grid neighbours
    when fetching each group.
    """
    if tiling is None or lat.size == 0:
        return np.zeros(lat.size, dtype=np.int64)
    if not tiling.merge_adjacent:
        # The caller still expands each group's domain by its requested margin.
        # Assigning only the point's own tile avoids merging overlapping halos.
        cells = _tile_ranges(lat, lon, 0.0, tiling)[:, [0, 2]]
        _, labels = np.unique(cells, axis=0, return_inverse=True)
        return np.asarray(labels, dtype=np.int64)
    rects, inverse = np.unique(_tile_ranges(lat, lon, margin, tiling), axis=0, return_inverse=True)
    sets = _UnionFind()
    ncols = tiling.n_cols
    firsts = [sets.add_rect(r0, r1, c0, c1, ncols) for r0, r1, c0, c1 in rects.tolist()]
    sets.link_neighbours(ncols)
    label = sets.labels()
    per_rect = np.array([label[sets.find(cell)] for cell in firsts], dtype=np.int64)
    return per_rect[inverse.reshape(-1)]


def _tile_ranges(
    lat: NDArray[np.float64], lon: NDArray[np.float64], margin: float, tiling: Tiling
) -> NDArray[np.int64]:
    """Per point, the tile rows and columns ``(r0, r1, c0, c1)`` within ``margin`` of it.

    Rows are shifted to start at 0; ``c1`` may exceed the column count when the
    range wraps around the dateline.
    """
    r_lo = np.floor((lat - margin - tiling.lat0) / tiling.lat_step).astype(np.int64)
    r_hi = np.floor((lat + margin - tiling.lat0) / tiling.lat_step).astype(np.int64)
    shift = int(r_lo.min())
    if tiling.lon_step is None:
        c_lo = c_hi = np.zeros(lat.size, dtype=np.int64)
    else:
        ncols = tiling.n_cols
        rel = lon - tiling.lon0
        c_lo = np.floor((rel - margin) / tiling.lon_step).astype(np.int64)
        c_hi = np.floor((rel + margin) / tiling.lon_step).astype(np.int64)
        width = np.minimum(c_hi - c_lo, ncols - 1)
        c_lo = np.mod(c_lo, ncols)
        c_hi = c_lo + width
    return np.stack([r_lo - shift, r_hi - shift, c_lo, c_hi], axis=1)


class _UnionFind:
    """Disjoint sets of occupied tiles, remembering each set's smallest tile."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}
        self._first: dict[tuple[int, int], tuple[int, int]] = {}

    def __contains__(self, cell: tuple[int, int]) -> bool:
        return cell in self._parent

    def add(self, cell: tuple[int, int]) -> None:
        """Register an occupied tile."""
        if cell not in self._parent:
            self._parent[cell] = cell
            self._first[cell] = cell

    def add_rect(self, r0: int, r1: int, c0: int, c1: int, ncols: int) -> tuple[int, int]:
        """Register a rectangle of tiles as one set; returns its first tile."""
        cells = [(r, c % ncols) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        for cell in cells:
            self.add(cell)
            self.union(cells[0], cell)
        return cells[0]

    def link_neighbours(self, ncols: int) -> None:
        """Merge the sets of tiles that touch (north, and east across the dateline)."""
        for r, c in self.cells():
            for nb in ((r + 1, c), (r, (c + 1) % ncols)):
                if nb in self:
                    self.union((r, c), nb)

    def labels(self) -> dict[tuple[int, int], int]:
        """Label of each set, ordered by its smallest tile."""
        roots = sorted({self.find(cell) for cell in self.cells()}, key=self.first_cell)
        return {root: i for i, root in enumerate(roots)}

    def cells(self) -> list[tuple[int, int]]:
        """Every occupied tile."""
        return list(self._parent)

    def find(self, cell: tuple[int, int]) -> tuple[int, int]:
        """Representative of the set of ``cell`` (with path halving)."""
        while self._parent[cell] != cell:
            self._parent[cell] = self._parent[self._parent[cell]]
            cell = self._parent[cell]
        return cell

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        """Merge the sets of ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
            self._first[ra] = min(self._first[ra], self._first[rb])

    def first_cell(self, root: tuple[int, int]) -> tuple[int, int]:
        """Smallest tile of a set (stable ordering of components)."""
        return self._first[root]
