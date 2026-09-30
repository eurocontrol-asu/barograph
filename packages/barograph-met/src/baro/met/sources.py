"""Gridded sources: which arrays, which vertical coordinate, how to build a cube."""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Cube, Domain, GridIndex
from baro.met.store import FileStore, HttpStore, TransferStats, ZarrStore
from baro.met.vertical import (
    STANDARD_SURFACE_PRESSURE,
    VerticalCoordinate,
    model_levels,
    pressure_levels,
)

__all__ = [
    "ARCO_BASE_URL",
    "ARCO_MODEL_LEVEL",
    "ARCO_PRESSURE_LEVEL",
    "FASTMETEO_LEVELS_22",
    "GridSource",
    "arco_model_levels",
    "arco_pressure_levels",
]

ARCO_BASE_URL = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar"
ARCO_MODEL_LEVEL = "model-level-1h-0p25deg.zarr-v1"
ARCO_PRESSURE_LEVEL = "full_37-1h-0p25deg-chunk-1.zarr-v3"

DEFAULT_FIELDS = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}

# FastMeteo 1.2.0 default pressure-level subset (fastmeteo/source/arco_era5.py)
FASTMETEO_LEVELS_22 = (
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
)  # fmt: skip

_UNITS = re.compile(r"hours since (\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?")


@dataclass(frozen=True)
class SurfacePressure:
    """Where a hybrid-level source reads its surface pressure."""

    store: ZarrStore
    array: str = "surface_pressure"


class GridSource:
    """An hourly, regular lat/lon zarr store with a level dimension.

    ``vertical`` lists all levels of the store's level axis, in the store's
    order (top to bottom). Cubes carry only the levels needed to bracket the
    requested pressures, read with block-level range requests.
    """

    def __init__(  # noqa: PLR0913 - keyword-only configuration of a public source
        self,
        source_id: str,
        store: ZarrStore,
        vertical: VerticalCoordinate,
        *,
        fields: dict[str, str] | None = None,
        level_array: str,
        surface_pressure: SurfacePressure | None = None,
    ) -> None:
        self.source_id = source_id
        self.store = store
        self.vertical = vertical
        self.fields = dict(fields or DEFAULT_FIELDS)
        self.surface_pressure = surface_pressure
        if vertical.needs_surface_pressure and surface_pressure is None:
            raise ValueError("hybrid levels need a surface pressure array")
        store_levels = np.asarray(store.read_1d(level_array), dtype=np.int64)
        missing = np.setdiff1d(vertical.ids, store_levels)
        if missing.size:
            raise ValueError(f"{source_id}: levels {missing.tolist()} not in the store")
        # position on the store's level axis of each level of ``vertical``
        self._store_index = np.searchsorted(store_levels, vertical.ids)
        if not np.array_equal(store_levels[self._store_index], vertical.ids):
            raise ValueError(f"{source_id}: store level axis must be sorted top to bottom")
        self.lat = store.read_1d("latitude").astype(np.float64)
        self.lon = store.read_1d("longitude").astype(np.float64)
        self._epoch = _hour_epoch(store.array("time").attrs.get("units", ""))
        self._t0 = int(store.read_1d("time", 0, 1)[0])
        self._checked_time_chunks: set[int] = set()
        self._lock = threading.Lock()

    @property
    def stats(self) -> list[TransferStats]:
        """Transfer counters of every underlying byte store."""
        stores = [self.store.store]
        if self.surface_pressure is not None and self.surface_pressure.store is not self.store:
            stores.append(self.surface_pressure.store.store)
        return [s.stats for s in stores]

    def time_index(self, hour: np.datetime64) -> int:
        """Index of ``hour`` on the store's time axis, verified against the time array.

        The axis is assumed hourly and contiguous; each time chunk is checked
        once (a few KiB) before its indices are trusted.
        """
        hours = int((np.datetime64(hour, "h") - self._epoch) / np.timedelta64(1, "h"))
        idx = hours - self._t0
        meta = self.store.array("time")
        if not 0 <= idx < meta.shape[0]:
            raise KeyError(f"{hour} outside the store's time axis")
        size = meta.chunks[0]
        chunk = idx // size
        with self._lock:
            known = chunk in self._checked_time_chunks
        if not known:
            start, stop = chunk * size, min((chunk + 1) * size, meta.shape[0])
            values = self.store.read_1d("time", start, stop)
            if not np.array_equal(values, np.arange(start, stop) + self._t0):
                raise RuntimeError(f"{self.source_id}: time axis is not hourly-contiguous")
            with self._lock:
                self._checked_time_chunks.add(chunk)
        return idx

    time_step = np.timedelta64(1, "h")

    def coverage(self) -> tuple[np.datetime64, np.datetime64] | None:
        """Published period, from the store's ``valid_time_start``/``valid_time_stop``.

        For ARCO-ERA5 the model-level store lags ERA5T (about three months).
        """
        attrs = self.store.attrs
        start, stop = attrs.get("valid_time_start"), attrs.get("valid_time_stop")
        if not start or not stop:
            return None
        begin = np.datetime64(start, "h")
        end = (
            np.datetime64(stop, "D") + np.timedelta64(23, "h")
            if len(stop) <= len("YYYY-MM-DD")
            else np.datetime64(stop, "h")
        )
        return begin, np.datetime64(end, "h")

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Analyses at ``t0`` and one hour later (no run: reanalysis)."""
        h = np.datetime64(t0, "h")
        return CubeKey(h), CubeKey(h + self.time_step)

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        """Read one hour over ``domain`` with the levels bracketing ``p_range`` (Pa)."""
        hour = key.valid
        try:
            t = self.time_index(hour)
        except KeyError as exc:
            raise DataUnavailableError(f"{self.source_id}: {hour} not in the store") from exc
        grid = GridIndex.locate(domain, self.lat, self.lon)
        ps: NDArray[np.float32] | None = None
        ps_lo = ps_hi = STANDARD_SURFACE_PRESSURE
        if self.surface_pressure is not None:
            sp = self.surface_pressure
            ps = sp.store.read_rows(sp.array, [(t,)], grid.rows, executor=executor)[0][:, grid.cols]
            if np.isfinite(ps).any():
                ps_lo, ps_hi = float(np.nanmin(ps)), float(np.nanmax(ps))
        positions = self.vertical.select_bracketing(
            p_range[0], p_range[1], ps_lo, ps_hi, level_margin
        )
        lead = [(t, int(k)) for k in self._store_index[positions]]
        fields = {
            name: self.store.read_rows(array, lead, grid.rows, executor=executor)[:, :, grid.cols]
            for name, array in self.fields.items()
        }
        return Cube(
            source=self.source_id,
            hour=np.datetime64(hour, "h"),
            lat=grid.lat,
            lon=grid.lon,
            vertical=self.vertical.subset(positions),
            fields=fields,
            ps=ps,
            p_range=(float(p_range[0]), float(p_range[1])),
            level_margin=level_margin,
            request_domain=domain,
        )


def _hour_epoch(units: str) -> np.datetime64:
    m = _UNITS.fullmatch(units.strip())
    if m is None:
        raise ValueError(f"unsupported time units {units!r}; expected 'hours since ...'")
    stamp = datetime.fromisoformat(f"{m.group(1)}T{m.group(2) or '00:00'}")
    return np.datetime64(stamp, "h")


def _open(url: str) -> ZarrStore:
    """HTTP(S) store, or a local mirror directory when ``url`` has no scheme."""
    return ZarrStore(HttpStore(url) if "://" in url else FileStore(url))


def _source_id(prefix: str, all_ids: NDArray[np.int64], ids: NDArray[np.int64]) -> str:
    """Stable identifier; a level subset gets its own id so caches never mix them."""
    if np.array_equal(all_ids, ids):
        return f"{prefix}{len(ids)}"
    digest = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()[:8]
    return f"{prefix}{len(ids)}-{digest}"


def arco_model_levels(
    *,
    levels: ArrayLike | None = None,
    base_url: str = ARCO_BASE_URL,
    fields: dict[str, str] | None = None,
) -> GridSource:
    """ARCO-ERA5 L137 model levels (0.25 deg, hourly) + surface pressure from the PL store.

    ``levels`` restricts the hybrid levels used (default: all 137).
    """
    ml = _open(f"{base_url}/{ARCO_MODEL_LEVEL}")
    pl = _open(f"{base_url}/{ARCO_PRESSURE_LEVEL}")
    vertical = model_levels(levels)
    return GridSource(
        _source_id("arco-era5-ml", model_levels().ids, vertical.ids),
        ml,
        vertical,
        fields=fields,
        level_array="hybrid",
        surface_pressure=SurfacePressure(pl),
    )


def arco_pressure_levels(
    *,
    levels: ArrayLike | None = None,
    base_url: str = ARCO_BASE_URL,
    fields: dict[str, str] | None = None,
) -> GridSource:
    """ARCO-ERA5 pressure levels (0.25 deg, hourly); ``levels`` in hPa (default: all 37)."""
    pl = _open(f"{base_url}/{ARCO_PRESSURE_LEVEL}")
    all_ids = np.asarray(pl.read_1d("level"), dtype=np.int64)
    vertical = pressure_levels(all_ids if levels is None else levels)
    return GridSource(
        _source_id("arco-era5-pl", all_ids, vertical.ids),
        pl,
        vertical,
        fields=fields,
        level_array="level",
    )
