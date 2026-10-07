"""Run-preserving AROME surface forecasts from Météo-France public GRIB files.

The public server retains approximately fourteen days. A fixed initialization
identifies the forecast, not its historical publication time: this source does
not reconstruct which run was available at a past cutoff.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met.arpege import ArpegeSource
from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain, GridIndex
from baro.met.grib import GribMessage, decode_message, fetch_messages
from baro.met.rangecache import RangeCache
from baro.met.store import MissingChunkError, TransferStats
from baro.met.surface import SurfaceCube, SurfaceVariable
from baro.met.tiling import Tiling

_MAX_LEAD = 51
_HEIGHT_LEVEL = 103
_WIND_HEIGHT = 10
_MAXIMUM = 2
_NEGATIVE_TOLERANCE = -0.001

_PARAMS = {
    "u10": ((0, 2, 2), 103, 10),
    "v10": ((0, 2, 3), 103, 10),
    "t2m": ((0, 0, 0), 103, 2),
    "precipitation": ((0, 1, 52), 1, 0),
}
_VARIABLES = {
    "u10": SurfaceVariable("m/s", 10),
    "v10": SurfaceVariable("m/s", 10),
    "t2m": SurfaceVariable("K", 2),
    "precipitation": SurfaceVariable("mm", 0, "sum"),
    "gusts": SurfaceVariable("m/s", 10, "max"),
}
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
        self.fields = dict(fields if fields is not None else {n: n for n in _VARIABLES})
        if not self.fields or set(self.fields.values()) - set(_VARIABLES):
            raise ValueError("unknown or empty AROME surface fields")
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
            runs = probe.published_runs()
            for run in reversed(runs):
                hours = [np.datetime64(valid_time, "h")]
                if np.datetime64(valid_time, "ns") != hours[0]:
                    hours.append(hours[0] + probe.time_step)
                leads = [int((hour - run) / probe.time_step) for hour in hours]
                period = any(v.temporal != "instant" for v in probe.variables.values())
                endpoint = leads[-1]
                if min(leads) < 0 or max(leads) > _MAX_LEAD or (period and endpoint == 0):
                    continue
                if "precipitation" in probe.fields.values() and endpoint > 1:
                    leads.append(endpoint - 1)
                if all(probe._forecast._published(probe.file_key(run, lead)) for lead in leads):
                    result = cls(run=run, **kwargs)
                    result.observed_at = np.datetime64("now", "s")
                    return result
            raise DataUnavailableError("no currently published AROME run covers this valid time")
        finally:
            probe.close()

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

    def _field(  # noqa: C901 - explicit validation of native accumulation intervals
        self, name: str, lead: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
        key = self.file_key(self.run, lead)
        try:
            idx = self.indexes.get(key)
        except MissingChunkError as exc:
            raise DataUnavailableError(f"{key}: forecast file not published") from exc
        if name == "gusts":
            candidates = [
                m
                for m in idx.messages
                if m.param == (0, 2, 22)
                and m.level_type == _HEIGHT_LEVEL
                and m.level == _WIND_HEIGHT
                and m.step_h == lead
                and m.statistical_process == _MAXIMUM
            ]
        else:
            param, level_type, level = _PARAMS[name]
            candidates = [
                m
                for m in idx.messages
                if m.param == param
                and m.level_type == level_type
                and m.level == level
                and m.step_h == lead
            ]
        if len(candidates) != 1:
            raise DataUnavailableError(f"{key}: missing or ambiguous {name} at lead {lead}")
        msg: GribMessage = candidates[0]
        if name == "gusts" and msg.start_step_h != lead - 1:
            raise DataUnavailableError("gust does not describe the preceding hour")
        field, lat, lon = decode_message(
            fetch_messages(self._forecast._messages, key, [msg], None)[0]
        )
        if name == "precipitation":
            if msg.statistical_process != 1 or msg.start_step_h is None:
                raise DataUnavailableError("precipitation has no supported accumulation interval")
            if msg.start_step_h == lead - 1:
                pass
            elif msg.start_step_h == 0 and lead > 1:
                previous, plat, plon = self._cumulative(name, lead - 1)
                if not np.array_equal(lat, plat) or not np.array_equal(lon, plon):
                    raise DataUnavailableError("accumulation grids differ")
                field = field - previous
            elif not (lead == 1 and msg.start_step_h == 0):
                raise DataUnavailableError("unsupported precipitation accumulation interval")
            field = np.where(field < _NEGATIVE_TOLERANCE, np.nan, np.maximum(field, 0)).astype(
                np.float32
            )
        return field, lat, lon

    def _cumulative(
        self, name: str, lead: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]:
        key = self.file_key(self.run, lead)
        try:
            idx = self.indexes.get(key)
        except MissingChunkError as exc:
            raise DataUnavailableError(f"{key}: forecast file not published") from exc
        messages = [
            m
            for m in idx.messages
            if m.param == _PARAMS[name][0]
            and m.level_type == 1
            and m.level == 0
            and m.step_h == lead
            and m.start_step_h == 0
            and m.statistical_process == 1
        ]
        if len(messages) != 1:
            raise DataUnavailableError(f"{key}: previous same-run accumulation unavailable")
        return decode_message(fetch_messages(self._forecast._messages, key, messages, None)[0])

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        if key.run != self.run:
            raise ValueError("cube initialization differs from this source run")
        delta = (key.valid - self.run) / self.time_step
        if not np.isfinite(delta) or delta != int(delta):
            raise ValueError("valid time must be aligned to an hourly forecast step")
        lead = int(delta)
        self.file_key(self.run, lead)
        selected = self.fields if fields is None else {out: self.fields[out] for out in fields}
        if not selected:
            raise ValueError("at least one surface field is required")
        arrays = {name: self._field(name, lead) for name in set(selected.values())}
        _, lat, lon = next(iter(arrays.values()))
        if any(
            not np.array_equal(lat, a[1]) or not np.array_equal(lon, a[2]) for a in arrays.values()
        ):
            raise DataUnavailableError("surface field grids differ")
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
