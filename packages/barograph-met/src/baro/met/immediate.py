"""Targeted, run-preserving AROME-PI and PIAF surface forecasts over WCS."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Self

import httpx
import numpy as np

from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain, GridIndex
from baro.met.immediate_grib import (
    ImmediateField,
    decode_surface_response,
    normalize_precipitation,
)
from baro.met.rangecache import RangeCache
from baro.met.store import TransferStats
from baro.met.surface import SurfaceCube, SurfaceVariable
from baro.met.tiling import Tiling
from baro.met.wcs import WCSClient, coverage_run
from baro.met.wcs_catalog import CoverageGrid, parse_description, select_coverage

_VARIABLES = {
    "u10": SurfaceVariable("m/s", 10),
    "v10": SurfaceVariable("m/s", 10),
    "t2m": SurfaceVariable("K", 2),
    "precipitation": SurfaceVariable("mm", 0, "sum"),
    "rain": SurfaceVariable("mm", 0, "sum"),
    "gusts": SurfaceVariable("m/s", 10, "max"),
}
_AMOUNTS = {"precipitation", "rain"}
_AROME_STEP_SECONDS = 900
_AROME_URL = "https://public-api.meteofrance.fr/public/aromepi/1.0/wcs/"
_PIAF_URL = "https://api.meteofrance.fr/pro/piaf/1.0/wcs/MF-NWP-HIGHRES-PIAF-001-FRANCE-WCS"


@dataclass(frozen=True)
class _Model:
    endpoint: str
    step_seconds: int
    run_seconds: int
    horizon_seconds: int
    defaults: tuple[str, ...]


def _stamp(run: np.datetime64 | datetime, cadence: int) -> np.datetime64:
    stamp = np.datetime64(run, "ns")
    if np.isnat(stamp) or int(stamp.astype(np.int64)) % (cadence * 1_000_000_000):
        raise ValueError("run must be an exact UTC model initialization")
    return stamp


class _CoverageReader:
    """Discover metadata once and cache only validated fixed-run field requests."""

    def __init__(self, client: WCSClient, step_seconds: int) -> None:
        self.client = client
        self.step_seconds = step_seconds
        self._ids: tuple[str, ...] | None = None
        self._grids: dict[str, CoverageGrid] = {}

    def catalog(self, *, refresh: bool = False) -> tuple[str, ...]:
        if self._ids is None or refresh:
            self._ids = self.client.capabilities()
        return self._ids

    def metadata(self, field: str, run: np.datetime64) -> tuple[str, CoverageGrid]:
        try:
            identifier = select_coverage(self.catalog(), field, run, self.step_seconds)
        except ValueError as exc:
            raise DataUnavailableError(str(exc)) from exc
        if identifier not in self._grids:
            self._grids[identifier] = parse_description(self.client.describe(identifier))
        return identifier, self._grids[identifier]

    def supports(self, name: str, run: np.datetime64, endpoints: list[np.datetime64]) -> bool:
        """Check the advertised level and all counters needed for one field."""
        identifier, grid = self.metadata(name, run)
        height = _VARIABLES[name].height_m
        if height and height not in grid.heights:
            return False
        leads = {int((t - run) / np.timedelta64(1, "s")) for t in endpoints}
        if name in _AMOUNTS and identifier.endswith("Z"):
            leads.update(
                lead - self.step_seconds for lead in tuple(leads) if lead > self.step_seconds
            )
        return leads.issubset(grid.lead_seconds)

    def field(self, name: str, key: CubeKey, domain: Domain) -> ImmediateField:
        current = self._read(name, key, domain)
        if name not in _AMOUNTS:
            return current
        previous = None
        start = key.valid - np.timedelta64(self.step_seconds, "s")
        if current.interval_start != start:
            previous = self._read(name, CubeKey(start, key.run), domain)
        return normalize_precipitation(
            current, interval_seconds=self.step_seconds, previous=previous
        )

    def _read(self, name: str, key: CubeKey, domain: Domain) -> ImmediateField:
        assert key.run is not None
        run = key.run
        identifier, grid = self.metadata(name, key.run)
        lead = int((key.valid - key.run) / np.timedelta64(1, "s"))
        if lead not in grid.lead_seconds:
            raise DataUnavailableError(f"{name}: forecast lead {lead} s is not published")
        subsets = _subsets(grid, domain, lead, _VARIABLES[name].height_m)
        decoded: list[ImmediateField] = []

        def validate(data: bytes) -> None:
            value = decode_surface_response(
                data,
                field=name,
                run=run,
                valid_time=key.valid,
                interval_seconds=self.step_seconds,
            )
            _check_region(value, grid, domain)
            decoded.append(value)

        self.client.coverage(identifier, subsets=subsets, validate=validate)
        return decoded[0]


def _check_region(value: ImmediateField, grid: CoverageGrid, domain: Domain) -> None:
    """Require the requested native grid, allowing equivalent longitude framing."""
    expected = GridIndex.locate(domain, grid.lat, grid.lon)
    longitude = value.lon + 360 * round((expected.lon[0] - value.lon[0]) / 360)
    for actual, wanted in ((value.lat, expected.lat[::-1]), (longitude, expected.lon)):
        if actual.shape != wanted.shape or not np.allclose(actual, wanted, rtol=0, atol=1e-6):
            raise ValueError("WCS response grid differs from requested native subset")


def _subsets(grid: CoverageGrid, domain: Domain, lead: int, height: float) -> list[str]:
    """Snap to just the grid centres required for bilinear interpolation."""
    region = GridIndex.locate(domain, grid.lat, grid.lon)
    subsets = [
        f"time({lead})",
        f"lat({region.lat[-1]:.8f},{region.lat[0]:.8f})",
        f"long({region.lon[0]:.8f},{region.lon[-1]:.8f})",
    ]
    if grid.heights:
        if height not in grid.heights:
            raise DataUnavailableError(f"height {height} m is not published")
        subsets.append(f"height({height:g})")
    elif height != 0:
        raise DataUnavailableError("height field has no height axis")
    return subsets


def _endpoints(
    variable: SurfaceVariable, valid: np.datetime64, keys: tuple[CubeKey, CubeKey]
) -> list[np.datetime64]:
    """Native timestamps contributing to one requested variable."""
    if valid == keys[0].valid:
        return [keys[0].valid]
    if variable.temporal == "instant":
        return [keys[0].valid, keys[1].valid]
    return [keys[1].valid]


def _surface_cube(
    key: CubeKey, selected: dict[str, str], arrays: dict[str, ImmediateField]
) -> SurfaceCube:
    """Assemble output aliases only after checking that all grids coincide."""
    first = next(iter(arrays.values()))
    for array in arrays.values():
        if not np.array_equal(first.lat, array.lat) or not np.array_equal(first.lon, array.lon):
            raise ValueError("requested surface fields have different grids")
    return SurfaceCube(
        key,
        first.lat,
        first.lon,
        {out: arrays[name].values for out, name in selected.items()},
        {out: _VARIABLES[name] for out, name in selected.items()},
    )


class _ImmediateSource:
    """Shared surface-source contract; public subclasses select the model."""

    def __init__(
        self,
        model: _Model,
        run: np.datetime64 | datetime,
        client: WCSClient,
        *,
        fields: dict[str, str] | None,
        tile_deg: float,
    ) -> None:
        self.run = _stamp(run, model.run_seconds)
        self._model = model
        self.time_step = np.timedelta64(model.step_seconds, "s")
        self.fields = dict(fields if fields is not None else {n: n for n in model.defaults})
        allowed = (
            set(_VARIABLES) if model.step_seconds == _AROME_STEP_SECONDS else {"precipitation"}
        )
        if not self.fields or set(self.fields.values()) - allowed:
            raise ValueError("unknown or empty immediate-forecast surface fields")
        if not np.isfinite(tile_deg) or tile_deg <= 0:
            raise ValueError("tile_deg must be positive and finite")
        self.variables = {out: _VARIABLES[name] for out, name in self.fields.items()}
        self.observed_at: np.datetime64 | None = None
        self._tiling = Tiling(-90, tile_deg, -180, tile_deg, merge_adjacent=False)
        self._reader = _CoverageReader(client, model.step_seconds)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the owned HTTP client; an injected client remains caller-owned."""
        self._reader.client.close()

    @property
    def stats(self) -> list[TransferStats]:
        """Network transfers, including catalog and coverage metadata requests."""
        return [self._reader.client.stats]

    def use_cache(self, cache: RangeCache) -> None:
        """Reuse immutable targeted GRIB responses across calls and processes."""
        self._reader.client.cache = cache

    def tiling(self, key: CubeKey) -> Tiling:
        """Partition occupied areas without merging long connected trajectories."""
        return self._tiling

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        """Nominal forecast window; unavailable native leads are flagged on read."""
        period = any(v.temporal != "instant" for v in self.variables.values())
        start = self.run + np.timedelta64(int(period), "ns")
        return start, self.run + np.timedelta64(self._model.horizon_seconds, "s")

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Native endpoints in the same explicit run, at second precision."""
        step_ns = self._model.step_seconds * 1_000_000_000
        stamp = np.datetime64(t0, "ns")
        if np.isnat(stamp):
            raise ValueError("valid time cannot be NaT")
        left = np.datetime64(int(stamp.astype(np.int64)) // step_ns * step_ns, "ns")
        return CubeKey(left, self.run), CubeKey(left + self.time_step, self.run)

    def published_runs(self, *, refresh: bool = False) -> list[np.datetime64]:
        """Initializations in the catalog observed now, not historical availability."""
        return sorted(
            {
                run
                for identifier in self._reader.catalog(refresh=refresh)
                if (run := coverage_run(identifier)) is not None
            }
        )

    @classmethod
    def latest(cls, valid_time: np.datetime64, **kwargs: Any) -> Self:
        """Newest advertised run covering all requested native field endpoints.

        ``observed_at`` records this catalog check, not original publication.
        Missing field downloads remain explicit; no run is silently substituted.
        """
        valid = np.datetime64(valid_time, "ns")
        if np.isnat(valid):
            raise ValueError("valid time cannot be NaT")
        source = cls(run=np.datetime64("2000-01-01"), **kwargs)
        try:
            for run in reversed(source.published_runs()):
                source.run = run
                if source._covers(valid):
                    source.observed_at = np.datetime64("now", "ns")
                    return source
            raise DataUnavailableError("no published run covers the requested fields and time")
        except BaseException:
            source.close()
            raise

    def _covers(self, valid: np.datetime64) -> bool:
        begin, end = self.coverage()
        if not begin <= valid <= end:
            return False
        keys = self.cube_keys(valid)
        try:
            return all(
                self._reader.supports(name, self.run, _endpoints(_VARIABLES[name], valid, keys))
                for name in set(self.fields.values())
            )
        except DataUnavailableError:
            return False

    def _check_key(self, key: CubeKey) -> None:
        """Reject foreign runs, unaligned native endpoints and extrapolation."""
        delta = (key.valid - self.run) / self.time_step
        if key.run != self.run or not np.isfinite(delta) or delta != int(delta):
            raise ValueError("cube must belong to this run and align to its native time step")
        if delta < 0 or key.valid > self.coverage()[1]:
            raise DataUnavailableError("forecast lead is outside this model's horizon")

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        """Fetch only selected fields and the bounding interpolation neighbours."""
        self._check_key(key)
        selected = self.fields if fields is None else {out: self.fields[out] for out in fields}
        if not selected:
            raise ValueError("at least one surface field is required")
        arrays = {n: self._reader.field(n, key, domain) for n in set(selected.values())}
        return _surface_cube(key, selected, arrays)


class AromePISource(_ImmediateSource):
    """AROME-PI surface forecasts every 15 minutes, preserving one hourly run.

    ``resolution`` selects the native 0.025° or 0.01° grid. The finer grid does
    not provide mean u/v wind; its defaults are t2m, precipitation and gusts.
    The 0.025° defaults also include u10/v10; rain is optional on either grid.
    ``token`` may be a callable for credential renewal. API keys use the default
    ``auth='apikey'``; OAuth access tokens require ``auth='bearer'``.
    """

    def __init__(  # noqa: PLR0913 - explicit source and transport configuration
        self,
        *,
        run: np.datetime64 | datetime,
        token: str | Callable[[], str],
        resolution: Literal["0.025", "0.01"] = "0.025",
        fields: dict[str, str] | None = None,
        tile_deg: float = 0.25,
        auth: Literal["apikey", "bearer"] = "apikey",
        client: httpx.Client | None = None,
    ) -> None:
        if resolution not in {"0.025", "0.01"}:
            raise ValueError("AROME-PI resolution must be 0.025 or 0.01 degrees")
        grid = "0025" if resolution == "0.025" else "001"
        defaults: tuple[str, ...] = ("t2m", "precipitation", "gusts")
        if resolution == "0.025":
            defaults = ("u10", "v10", *defaults)
        model = _Model(
            _AROME_URL + f"MF-NWP-HIGHRES-AROMEPI-{grid}-FRANCE-WCS", 900, 3600, 21600, defaults
        )
        # Validate user configuration before allocating an HTTP client.
        _stamp(run, model.run_seconds)
        transport = WCSClient(model.endpoint, token, auth=auth, client=client)
        try:
            super().__init__(model, run, transport, fields=fields, tile_deg=tile_deg)
        except BaseException:
            transport.close()
            raise


class PiafSource(_ImmediateSource):
    """Deterministic PIAF precipitation, native 5-minute amounts up to 195 minutes.

    Both run cadence and output step are five minutes. Despite its WCS name
    containing RATE, the supported product contains interval amounts in mm,
    verified against the GRIB parameter, units and statistical interval.
    """

    def __init__(  # noqa: PLR0913 - explicit source and transport configuration
        self,
        *,
        run: np.datetime64 | datetime,
        token: str | Callable[[], str],
        fields: dict[str, str] | None = None,
        tile_deg: float = 0.25,
        auth: Literal["apikey", "bearer"] = "apikey",
        client: httpx.Client | None = None,
    ) -> None:
        model = _Model(_PIAF_URL, 300, 300, 11700, ("precipitation",))
        _stamp(run, model.run_seconds)
        transport = WCSClient(model.endpoint, token, auth=auth, client=client)
        try:
            super().__init__(model, run, transport, fields=fields, tile_deg=tile_deg)
        except BaseException:
            transport.close()
            raise
