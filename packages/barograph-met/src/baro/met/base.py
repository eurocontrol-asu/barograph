"""Source contract shared by reanalyses (ERA5) and forecasts (ARPEGE, ...).

A *cube* is one gridded time slice of a source over a region. It is identified
by its validity time and, for forecasts, the run (base time) it comes from.
Sources declare their native time step; the enricher brackets every point
between two consecutive valid times and asks the source which cubes (and, for
forecasts, which run) to use for that interval.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from baro.met.domain import Cube, Domain
    from baro.met.rangecache import RangeCache
    from baro.met.store import TransferStats
    from baro.met.tiling import Tiling

__all__ = ["CubeKey", "DataUnavailableError", "Source"]


class DataUnavailableError(LookupError):
    """The source has no data for this key (not published, expired, out of range)."""


@dataclass(frozen=True, order=True)
class CubeKey:
    """Validity time of a cube and, for forecasts, the run it belongs to."""

    valid: np.datetime64
    run: np.datetime64 | None = None
    source: str | None = None  # set by composite sources to route the fetch

    @property
    def lead(self) -> np.timedelta64 | None:
        """Forecast lead time, or ``None`` for analyses and stitched forecast archives."""
        return None if self.run is None else self.valid - self.run

    def __str__(self) -> str:
        return str(self.valid) if self.run is None else f"{self.valid} (run {self.run})"


class Source(Protocol):
    """What the enricher needs from a gridded source."""

    source_id: str
    fields: dict[str, str]  # output name -> source-specific variable identifier
    time_step: np.timedelta64

    @property
    def stats(self) -> list[TransferStats]:
        """Transfer counters of the source's stores."""
        ...

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Cubes bracketing the interval ``[t0, t0 + time_step)``."""
        ...

    def coverage(self) -> tuple[np.datetime64, np.datetime64] | None:
        """First and last valid times the source can serve (``None``: unknown)."""
        ...

    def tiling(self, key: CubeKey) -> Tiling | None:
        """Horizontal tiling of the cube ``key`` in storage (``None``: read as one piece)."""
        ...

    def use_cache(self, cache: RangeCache) -> None:
        """Serve the immutable bytes this source reads through ``cache`` from now on."""
        ...

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        """Read one cube; raise :class:`DataUnavailableError` if it does not exist."""
        ...
