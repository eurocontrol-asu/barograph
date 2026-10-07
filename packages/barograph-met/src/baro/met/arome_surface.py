"""Run-preserving AROME surface forecasts from Météo-France public GRIB files.

The public server retains approximately fourteen days. A fixed initialization
identifies the forecast, not its historical publication time: this source does
not reconstruct which run was available at a past cutoff.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met.arpege import ArpegeSource
from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain, GridIndex
from baro.met.grib import GribIndex, GribMessage, decode_message, fetch_messages
from baro.met.rangecache import RangeCache
from baro.met.store import MissingChunkError, TransferStats
from baro.met.surface import SurfaceCube, SurfaceVariable
from baro.met.tiling import Tiling

_MAXIMUM = 2
_MAX_LEAD = 51
_NEGATIVE_TOLERANCE = -0.001

_PARAMS = {
    "u10": ((0, 2, 2), 103, 10),
    "v10": ((0, 2, 3), 103, 10),
    "t2m": ((0, 0, 0), 103, 2),
    "precipitation": ((0, 1, 52), 1, 0),
    "gusts": ((0, 2, 22), 103, 10),
}
_VARIABLES = {
    "u10": SurfaceVariable("m/s", 10),
    "v10": SurfaceVariable("m/s", 10),
    "t2m": SurfaceVariable("K", 2),
    "precipitation": SurfaceVariable("mm", 0, "sum"),
    "gusts": SurfaceVariable("m/s", 10, "max"),
}
type DecodedField = tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]


_RANGES = ((0, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 36), (37, 42), (43, 48), (49, 51))


class AromeSurfaceSource:
    """Hourly surface fields from one explicit AROME initialization (0..51 h).

    ``fields`` maps output names to u10, v10, t2m, precipitation or gusts.
    Precipitation is a preceding-hour amount, derived from cumulative totals
    within the same run. Gusts retain the reported preceding-hour maximum.
    Only the requested GRIB messages are downloaded; their immutable ranges
    and per-file indexes use the shared range cache. No altitude is required.
    ``published_runs`` describes files visible now, never historical availability.
    """

    def __init__(
        self,
        *,
        run: np.datetime64 | datetime,
        fields: dict[str, str] | None = None,
        base_url: str = "https://files.data.gouv.fr/meteofrance-pnt/pnt",
        index_dir: str | Path | None = None,
    ) -> None:
        stamp = np.datetime64(run, "ns")
        hour = stamp.astype("datetime64[h]")
        if np.isnat(stamp) or stamp != hour or int(str(hour)[11:13]) % 3:
            raise ValueError("AROME run must be an exact UTC three-hour initialization")
        self._forecast = ArpegeSource(
            run=hour, run_hours=range(0, 24, 3), base_url=base_url, index_dir=index_dir
        )
        self.run = hour
        self.observed_at: np.datetime64 | None = None
        self.time_step = np.timedelta64(1, "h")
        self.indexes = self._forecast.indexes
        self.fields = _checked_fields(fields)
        self.variables = {out: _VARIABLES[name] for out, name in self.fields.items()}
        self.source_id = "arome-0025-surface"
        self._forecast.source_id = self.source_id
        if index_dir is not None:
            self.indexes.directory = Path(index_dir).expanduser() / self.source_id

    def __enter__(self) -> AromeSurfaceSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def latest(cls, valid_time: np.datetime64, **kwargs: Any) -> AromeSurfaceSource:
        """Newest run whose required files are available now for ``valid_time``.

        ``observed_at`` records when publication was checked. It is not a
        historical publication timestamp and cannot establish a past cutoff.
        """
        if np.isnat(valid_time):
            raise ValueError("valid time cannot be NaT")
        probe = cls(run=np.datetime64("2000-01-01T00"), **kwargs)
        try:
            run = probe._latest_available_run(valid_time)
            result = cls(run=run, **kwargs)
            result.observed_at = np.datetime64("now", "s")
            return result
        finally:
            probe.close()

    def _latest_available_run(self, valid_time: np.datetime64) -> np.datetime64:
        """Check the required native files of candidate initializations."""
        for run in reversed(self.published_runs()):
            leads = _required_leads(valid_time, run, self.fields)
            if leads and all(self._forecast._published(self.file_key(run, lead)) for lead in leads):
                return run
        raise DataUnavailableError("no currently published AROME run covers this valid time")

    @property
    def stats(self) -> list[TransferStats]:
        return self._forecast.stats

    def close(self) -> None:
        close = getattr(self._forecast.store, "close", None)
        if close is not None:
            close()

    def tiling(self, key: CubeKey) -> Tiling | None:
        return None

    def use_cache(self, cache: RangeCache) -> None:
        self._forecast.use_cache(cache)

    def published_runs(self) -> list[np.datetime64]:
        return self._forecast.published_runs()

    def coverage(self, max_age_s: float = 600) -> tuple[np.datetime64, np.datetime64]:
        return self.run, self.run + np.timedelta64(51, "h")

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        hour = np.datetime64(t0, "h")
        return CubeKey(hour, self.run), CubeKey(hour + self.time_step, self.run)

    def file_key(self, run: np.datetime64, lead_h: int) -> str:
        for start, end in _RANGES:
            if start <= lead_h <= end:
                stamp = str(np.datetime64(run, "h")) + ":00:00Z"
                name = f"arome__0025__SP1__{start:02d}H{end:02d}H__{stamp}.grib2"
                return f"{stamp}/arome/0025/SP1/{name}"
        raise DataUnavailableError(f"AROME lead {lead_h} h outside 0..51")

    def _index(self, key: str) -> GribIndex:
        """Translate a missing forecast package to the source availability contract."""
        try:
            return self.indexes.get(key)
        except MissingChunkError as exc:
            raise DataUnavailableError(f"{key}: forecast file not published") from exc

    def _message(self, key: str, name: str, lead: int, *, cumulative: bool = False) -> GribMessage:
        """Select a unique native message, including its statistical meaning."""
        param, level_type, level = _PARAMS[name]
        candidates = [
            m
            for m in self._index(key).messages
            if (m.param, m.level_type, m.level, m.step_h) == (param, level_type, level, lead)
            if m.statistical_process == {"gusts": _MAXIMUM}.get(name, m.statistical_process)
        ]
        if cumulative:
            candidates = [
                m for m in candidates if (m.start_step_h, m.statistical_process) == (0, 1)
            ]
        if len(candidates) != 1:
            raise DataUnavailableError(f"{key}: missing or ambiguous {name} at lead {lead}")
        return candidates[0]

    def _decode(self, key: str, message: GribMessage) -> DecodedField:
        """Read one immutable message through the shared byte cache."""
        return decode_message(fetch_messages(self._forecast._messages, key, [message], None)[0])

    def _field(self, name: str, lead: int) -> DecodedField:
        """Read a native field and normalize its supported interval semantics."""
        key = self.file_key(self.run, lead)
        message = self._message(key, name, lead)
        if name == "gusts" and message.start_step_h != lead - 1:
            raise DataUnavailableError("gust does not describe the preceding hour")
        decoded = self._decode(key, message)
        if name == "precipitation":
            return _hourly_precipitation(decoded, message, lead, self._cumulative)
        return decoded

    def _cumulative(self, name: str, lead: int) -> DecodedField:
        """Read the preceding same-run cumulative counter without differencing it."""
        key = self.file_key(self.run, lead)
        return self._decode(key, self._message(key, name, lead, cumulative=True))

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        lead = _forecast_lead(key, self.run)
        self.file_key(self.run, lead)
        selected = self.fields if fields is None else {out: self.fields[out] for out in fields}
        if not selected:
            raise ValueError("at least one surface field is required")
        arrays = {name: self._field(name, lead) for name in set(selected.values())}
        _, lat, lon = next(iter(arrays.values()))
        for _, field_lat, field_lon in arrays.values():
            _require_same_grid((lat, lon), (field_lat, field_lon))
        # The shared GRIB decoder emits descending latitude.
        grid = GridIndex.locate(domain, lat, lon)
        cube_fields = {
            out: arrays[name][0][grid.rows][:, grid.cols][::-1].copy()
            for out, name in selected.items()
        }
        return SurfaceCube(
            key,
            grid.lat[::-1].copy(),
            grid.lon,
            cube_fields,
            {out: self.variables[out] for out in selected},
        )


def _require_same_grid(
    expected: tuple[NDArray[np.float64], NDArray[np.float64]],
    actual: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Reject unrelated grids before combining fields or cumulative counters."""
    if not all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True)):
        raise DataUnavailableError("surface field grids differ")


def _checked_fields(fields: dict[str, str] | None) -> dict[str, str]:
    """Resolve output aliases and validate the selected native variable names."""
    selected = dict(fields if fields is not None else {n: n for n in _VARIABLES})
    if not selected or set(selected.values()) - set(_VARIABLES):
        raise ValueError("unknown or empty AROME surface fields")
    return selected


def _required_leads(
    valid_time: np.datetime64, run: np.datetime64, fields: dict[str, str]
) -> list[int]:
    """Native endpoints and preceding counters needed for one query."""
    hour = np.datetime64(valid_time, "h")
    leads = [int((hour - run) / np.timedelta64(1, "h"))]
    if np.datetime64(valid_time, "ns") != hour:
        leads.append(leads[0] + 1)
    endpoint = leads[-1]
    period = any(_VARIABLES[name].temporal != "instant" for name in fields.values())
    if min(leads) < 0 or max(leads) > _MAX_LEAD or (period and endpoint == 0):
        return []
    if "precipitation" in fields.values() and endpoint > 1:
        leads.append(endpoint - 1)
    return leads


def _hourly_precipitation(
    decoded: DecodedField,
    message: GribMessage,
    lead: int,
    cumulative: Callable[[str, int], DecodedField],
) -> DecodedField:
    """Convert a native hourly or run-total counter to the preceding-hour amount."""
    if message.statistical_process != 1 or message.start_step_h is None:
        raise DataUnavailableError("precipitation has no supported accumulation interval")
    field, lat, lon = decoded
    if message.start_step_h != lead - 1:
        if message.start_step_h != 0 or lead <= 1:
            raise DataUnavailableError("unsupported precipitation accumulation interval")
        previous, plat, plon = cumulative("precipitation", lead - 1)
        _require_same_grid((lat, lon), (plat, plon))
        field = field - previous
    field = np.where(field < _NEGATIVE_TOLERANCE, np.nan, np.maximum(field, 0)).astype(np.float32)
    return field, lat, lon


def _forecast_lead(key: CubeKey, run: np.datetime64) -> int:
    """Validate run identity, hourly alignment and the published forecast horizon."""
    if key.run != run:
        raise ValueError("cube initialization differs from this source run")
    delta = (key.valid - run) / np.timedelta64(1, "h")
    if not np.isfinite(delta) or delta != int(delta):
        raise ValueError("valid time must be aligned to an hourly forecast step")
    lead = int(delta)
    return lead
