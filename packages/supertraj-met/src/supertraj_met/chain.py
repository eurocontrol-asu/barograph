"""Chain several sources of the same fields: the first one that serves an interval wins.

Typical use: the exact Météo-France files while they are published (rolling
14 days, full runs), and Open-Meteo's archive of the same model before that::

    arpege = SourceChain(
        ArpegeSource(resolution="01"), OpenMeteoSource("meteofrance_arpege_europe")
    )

Routing is decided per interval ``[t0, t0 + step)``: both cubes bracketing a
point always come from the same source (never a mix within one
interpolation). Both sources above serve "the most recent run available at
each hour", so the hand-over is seamless up to the archive's quantisation.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import replace

import numpy as np

from supertraj_met.base import CubeKey, DataUnavailableError, Source
from supertraj_met.domain import Cube, Domain
from supertraj_met.store import TransferStats

__all__ = ["SourceChain"]


class SourceChain:
    def __init__(self, *sources: Source) -> None:
        if not sources:
            raise ValueError("a chain needs at least one source")
        first = sources[0]
        for src in sources[1:]:
            if list(src.fields) != list(first.fields):
                raise ValueError("chained sources must provide the same output fields")
            if src.time_step != first.time_step:
                raise ValueError("chained sources must share their time step")
        ids = [s.source_id for s in sources]
        if len(set(ids)) != len(ids):
            raise ValueError("chained sources must have distinct ids")
        self.sources = sources
        self.fields = dict(first.fields)
        self.time_step = first.time_step
        self.source_id = "+".join(ids)
        self._by_id = {s.source_id: s for s in sources}

    @property
    def stats(self) -> list[TransferStats]:
        return [st for s in self.sources for st in s.stats]

    def coverage(self) -> tuple[np.datetime64, np.datetime64] | None:
        spans = [c for s in self.sources if (c := s.coverage()) is not None]
        if not spans:
            return None
        return min(c[0] for c in spans), max(c[1] for c in spans)

    def _serves(self, source: Source, t0: np.datetime64) -> bool:
        cov = source.coverage()
        t1 = t0 + self.time_step
        if cov is not None and not (cov[0] <= t0 and t1 <= cov[1]):
            return False
        check = getattr(source, "serves", None)
        return bool(check(t0)) if callable(check) else True

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        t0h = np.datetime64(t0, "h")
        chosen = next((s for s in self.sources if self._serves(s, t0h)), self.sources[-1])
        k0, k1 = chosen.cube_keys(t0h)
        return replace(k0, source=chosen.source_id), replace(k1, source=chosen.source_id)

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        if key.source is None or key.source not in self._by_id:
            raise DataUnavailableError(f"cube key {key} is not routed to a chained source")
        return self._by_id[key.source].fetch(
            key, domain, p_range, level_margin=level_margin, executor=executor
        )
