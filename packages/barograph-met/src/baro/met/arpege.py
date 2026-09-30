"""Météo-France ARPEGE forecasts (open data GRIB2) as a trajectory source.

Files: one GRIB2 per (run, grid, package, lead-time range), published without
authentication. Every message is one variable x isobaric level x step, so a
cube only downloads the messages of the requested variables, of the levels
bracketing the points and of one step: typically a few percent of the file.
The per-file index (offsets of the messages) is built once with range
requests and cached on disk, since a published file never changes.

Grids: ``"01"`` = 0.1 deg over Europe (20N-72N, 32W-42E), 24 levels 100-1000 hPa;
``"025"`` = 0.25 deg global, 34 levels 1-1000 hPa. Steps are hourly up to 48 h
(3-hourly beyond, and for the 1-70 hPa levels). Only isobaric levels are used.
Retention is 14 days (no open archive); runs at 00/06/12/18 UTC. Licence
Ouverte / Etalab 2.0.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import numpy as np

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.common import Lru
from baro.met.domain import Cube, Domain, GridIndex
from baro.met.grib import GribIndex, GribIndexStore, decode_message, fetch_messages
from baro.met.rangecache import CachedStore, RangeCache, namespace_for
from baro.met.store import (
    ByteStore,
    FileStore,
    HttpStore,
    MissingChunkError,
    TransferStats,
    list_s3,
)
from baro.met.tiling import Tiling
from baro.met.vertical import (
    MIN_LEVELS,
    STANDARD_SURFACE_PRESSURE,
    VerticalCoordinate,
    pressure_levels,
)

__all__ = ["ARPEGE_BASE_URL", "ARPEGE_LISTING_URL", "ARPEGE_PARAMS", "ArpegeSource"]

# Official entry point (data.gouv.fr "Données PNT - rétention 14 jours"); it redirects to
# Météo-France's object storage, and the store learns that redirect after one request.
ARPEGE_BASE_URL = "https://files.data.gouv.fr/meteofrance-pnt/pnt"
# S3-style listing of the same files (the data.gouv.fr entry point only serves HTML)
ARPEGE_LISTING_URL = "https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net"
_RECHECK_S = 300.0  # how long "file not published yet" is trusted
_MAX_LEAD_H = 102
_RUN_PREFIX = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}):00:00Z")

# FastMeteo / Météo-France short names -> GRIB2 (discipline, category, number)
ARPEGE_PARAMS: dict[str, tuple[int, int, int]] = {
    "u": (0, 2, 2),
    "v": (0, 2, 3),
    "t": (0, 0, 0),
    "r": (0, 1, 1),
    "z": (0, 3, 4),
}
_ISOBARIC = 100  # GRIB2 level type, value in Pa

_LEAD_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "01": (
        (0, 12), (13, 24), (25, 36), (37, 48), (49, 60), (61, 72), (73, 84), (85, 96), (97, 102),
    ),
    "025": ((0, 24), (25, 48), (49, 72), (73, 102)),
}  # fmt: skip
_HOUR = np.timedelta64(1, "h")

type Resolution = Literal["01", "025"]
type RunPolicy = Literal["latest", "daily"]


def _checked_fields(fields: dict[str, str] | None) -> dict[str, str]:
    """Output name -> ARPEGE short name, defaulting to u, v, t."""
    checked = dict(fields or {"u": "u", "v": "v", "t": "t"})
    unknown = set(checked.values()) - set(ARPEGE_PARAMS)
    if unknown:
        raise ValueError(f"unknown ARPEGE variables {sorted(unknown)}")
    return checked


class ArpegeSource:
    """ARPEGE isobaric fields.

    Parameters
    ----------
    resolution:
        ``"01"`` (Europe) or ``"025"`` (global).
    fields:
        Output name -> ARPEGE short name (default ``u``, ``v``, ``t``).
    levels:
        Isobaric levels (hPa) allowed; default: all published levels.
    run:
        ``"latest"``: most recent run (00/06/12/18 UTC) not after the interval,
        falling back to older runs if a file is not published; ``"daily"``:
        the 00 UTC run of the same day (FastMeteo's behaviour); or a fixed run
        time. Both cubes bracketing an interval always come from the same run.
    base_url / store:
        Where files live (HTTP base URL, local mirror directory, or a store).
    index_dir:
        Directory caching the per-file message indexes (strongly recommended;
        defaults to the enricher's cache when it has one).

    The files of a published run never change: with a cache (:meth:`use_cache`)
    every GRIB message read is kept on disk and never downloaded again.
    """

    def __init__(  # noqa: PLR0913 - keyword-only configuration of a public source
        self,
        *,
        resolution: Resolution = "01",
        package: str = "IP1",
        fields: dict[str, str] | None = None,
        levels: Sequence[int] | None = None,
        run: RunPolicy | np.datetime64 | datetime = "latest",
        run_hours: Sequence[int] = (0, 6, 12, 18),
        max_run_fallback: int = 4,
        base_url: str = ARPEGE_BASE_URL,
        store: ByteStore | None = None,
        index_dir: str | Path | None = None,
        listing_url: str | None = ARPEGE_LISTING_URL,
    ) -> None:
        if resolution not in _LEAD_RANGES:
            raise ValueError(f"unknown resolution {resolution!r}")
        self.resolution = resolution
        self.package = package
        self.fields = _checked_fields(fields)
        self.levels = None if levels is None else sorted(int(x) for x in levels)
        self.run = run if isinstance(run, str) else np.datetime64(run, "h")
        self.run_hours = sorted(run_hours)
        self.max_run_fallback = max_run_fallback
        self.time_step = _HOUR  # hourly for isobaric levels >= 100 hPa on both grids
        is_local = "://" not in base_url
        self.store = store or (FileStore(base_url) if is_local else HttpStore(base_url))
        self._messages: ByteStore = self.store  # message reads (through the cache, if any)
        self.source_id = f"arpege-{resolution}-{package}"
        self.indexes = GribIndexStore(
            self.store,
            None if index_dir is None else Path(index_dir).expanduser() / self.source_id,
        )
        self.listing_url = listing_url
        self._local_root = Path(base_url) if is_local else None
        self._coverage: tuple[float, tuple[np.datetime64, np.datetime64] | None] | None = None
        self._exists: Lru[str, tuple[bool, float]] = Lru(4096)  # published?, checked at
        self._lock = threading.Lock()

    @property
    def stats(self) -> list[TransferStats]:
        """Transfer counters of the underlying store."""
        return [self.store.stats]

    def tiling(self, key: CubeKey) -> Tiling | None:
        """``None``: a GRIB message holds a whole field, read in one piece."""
        return None

    def use_cache(self, cache: RangeCache) -> None:
        """Keep every message read in ``cache``; message indexes go there too by default."""
        self._messages = CachedStore(
            self.store, cache, namespace_for(self.store.location), lambda _key: True
        )
        if self.indexes.directory is None:
            self.indexes.directory = cache.root / "grib-index" / self.source_id

    # -- coverage ------------------------------------------------------------------------------
    def published_runs(self) -> list[np.datetime64]:
        """Runs currently published (rolling ~14-day window), oldest first."""
        if self._local_root is not None:
            names = [p.name for p in self._local_root.iterdir() if p.is_dir()]
        elif self.listing_url is not None:
            listed = list_s3(self.listing_url, "pnt/", dirs=True, stats=self.store.stats)
            names = [name.removeprefix("pnt/").rstrip("/") for name in listed]
        else:
            return []
        runs = sorted({np.datetime64(m[1], "h") for n in names if (m := _RUN_PREFIX.fullmatch(n))})
        return [r for r in runs if int(str(r)[11:13]) in self.run_hours]

    def coverage(self, max_age_s: float = 600.0) -> tuple[np.datetime64, np.datetime64] | None:
        """From the oldest published run to the last lead time of the newest one."""
        now = time.monotonic()
        if self._coverage is None or now - self._coverage[0] > max_age_s:
            try:
                runs = self.published_runs()
            except httpx.HTTPError:
                runs = []
            cov = (runs[0], runs[-1] + np.timedelta64(_MAX_LEAD_H, "h")) if runs else None
            self._coverage = (now, cov)
        return self._coverage[1]

    def serves(self, t0: np.datetime64) -> bool:
        """True if both ends of the interval starting at ``t0`` are published."""
        t0h = np.datetime64(t0, "h")
        for run in self._candidate_runs(t0h):
            leads = [int((t - run) / _HOUR) for t in (t0h, t0h + self.time_step)]
            try:
                if all(self._published(self.file_key(run, lead)) for lead in leads):
                    return True
            except DataUnavailableError:
                continue
        return False

    # -- files and runs ------------------------------------------------------------------------
    def file_key(self, run: np.datetime64, lead_h: int) -> str:
        """Path of the GRIB2 file holding ``lead_h`` hours of ``run``."""
        for a, b in _LEAD_RANGES[self.resolution]:
            if a <= lead_h <= b:
                break
        else:
            raise DataUnavailableError(f"lead time {lead_h} h beyond the published range")
        stamp = str(np.datetime64(run, "h")) + ":00:00Z"
        res, pkg = self.resolution, self.package
        name = f"arpege__{res}__{pkg}__{a:03d}H{b:03d}H__{stamp}.grib2"
        return f"{stamp}/arpege/{res}/{pkg}/{name}"

    def _published(self, key: str) -> bool:
        """Whether ``key`` exists; a "not yet" is re-checked after a while (runs keep coming)."""
        with self._lock:
            known = self._exists.get(key)
        if known is not None and (known[0] or time.monotonic() - known[1] < _RECHECK_S):
            return known[0]
        try:
            self.store.size(key)
            ok = True
        except MissingChunkError:
            ok = False
        with self._lock:
            self._exists[key] = (ok, time.monotonic())
        return ok

    def _candidate_runs(self, t0: np.datetime64) -> list[np.datetime64]:
        if not isinstance(self.run, str):
            return [self.run]
        day = np.datetime64(t0, "D")
        if self.run == "daily":
            return [np.datetime64(day, "h")]
        runs = []
        for d in (day, day - np.timedelta64(1, "D")):
            runs += [
                np.datetime64(d, "h") + np.timedelta64(h, "h") for h in reversed(self.run_hours)
            ]
        runs = [r for r in runs if r <= t0]
        return runs[: self.max_run_fallback + 1]

    def run_for(self, t0: np.datetime64) -> np.datetime64:
        """Run whose files hold both ends of the interval starting at ``t0``."""
        t0h = np.datetime64(t0, "h")
        candidates = self._candidate_runs(t0h)
        for run in candidates:
            leads = [int((t - run) / _HOUR) for t in (t0h, t0h + self.time_step)]
            try:
                if all(self._published(self.file_key(run, lead)) for lead in leads):
                    return run
            except DataUnavailableError:
                continue
        return candidates[0]  # nothing published: fetch will report the data as unavailable

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Cubes bracketing ``[t0, t0 + 1 h)``, both from the run chosen by :meth:`run_for`."""
        t0h = np.datetime64(t0, "h")
        run = self.run_for(t0h)
        return CubeKey(t0h, run), CubeKey(t0h + self.time_step, run)

    # -- cubes ---------------------------------------------------------------------------------
    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        """One validity time of one run over ``domain``, levels bracketing ``p_range`` (Pa)."""
        if key.run is None:
            raise ValueError("ARPEGE cubes need a run")
        lead_h = int((key.valid - key.run) / _HOUR)
        file = self.file_key(key.run, lead_h)
        try:
            idx = self.indexes.get(file, executor)
        except MissingChunkError as exc:
            raise DataUnavailableError(f"{self.source_id}: {file} not published") from exc
        vertical = self._vertical(idx, float(lead_h))
        positions = vertical.select_bracketing(
            p_range[0],
            p_range[1],
            STANDARD_SURFACE_PRESSURE,
            STANDARD_SURFACE_PRESSURE,
            level_margin,
        )
        wanted = []
        for name in self.fields.values():
            for hpa in vertical.ids[positions]:
                msg = idx.find(ARPEGE_PARAMS[name], _ISOBARIC, float(hpa) * 100.0, float(lead_h))
                if msg is None:
                    raise DataUnavailableError(f"{file}: no {name} at {hpa} hPa, step {lead_h} h")
                wanted.append(msg)
        buffers = fetch_messages(self._messages, file, wanted, executor)
        decoded = [decode_message(b) for b in buffers]
        _, lat, lon = decoded[0]
        grid = GridIndex.locate(domain, lat, lon)
        nlev = len(positions)
        fields = {}
        for i, out_name in enumerate(self.fields):
            layers = [decoded[i * nlev + k][0][grid.rows][:, grid.cols] for k in range(nlev)]
            fields[out_name] = np.stack(layers).astype(np.float32)
        return Cube(
            source=self.source_id,
            hour=key.valid,
            lat=grid.lat,
            lon=grid.lon,
            vertical=vertical.subset(positions),
            fields=fields,
            run=key.run,
        )

    def _vertical(self, idx: GribIndex, step_h: float) -> VerticalCoordinate:
        """Isobaric levels published for every field at this step.

        Stratospheric levels (1-70 hPa) of the 0.25 deg grid, and every level
        beyond 48 h, are only 3-hourly: they are simply absent at other steps.
        """
        common: set[int] | None = None
        for name in self.fields.values():
            lv = {round(x / 100.0) for x in idx.levels(ARPEGE_PARAMS[name], _ISOBARIC, step_h)}
            common = lv if common is None else common & lv
        levels = sorted(common or set())
        if self.levels is not None:
            levels = [x for x in levels if x in set(self.levels)]
        if len(levels) < MIN_LEVELS:
            raise DataUnavailableError(f"{idx.key}: fewer than 2 common isobaric levels")
        return pressure_levels(levels)
