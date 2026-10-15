"""Build bounded native weather windows using Barograph's targeted range reader."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met.arco_surface import ArcoRadiationSource
from baro.met.base import CubeKey
from baro.met.domain import Cube, Domain
from baro.met.sources import GridSource
from baro.met.weather import RadiationCube, WeatherCube

__all__ = ["WeatherRequest", "materialize_radiation", "materialize_weather"]

_UNITS = {
    "u": "m s-1",
    "v": "m s-1",
    "t": "K",
    "q": "kg kg-1",
    "omega": "Pa s-1",
    "ciwc": "kg kg-1",
    "tau_cirrus": "1",
    "geopotential": "m2 s-2",
}


@dataclass(frozen=True)
class WeatherRequest:
    """Region, UTC time bounds and pressure range (Pa) requiring interpolation.

    Includes neighbouring native times and pressure levels, without imposing
    a fixed integration horizon. Callers supply margins for lower probes,
    sedimentation and evolution; a window is not a bound on future motion.
    """

    domain: Domain
    start: np.datetime64
    end: np.datetime64
    pressure_pa: tuple[float, float]

    def __post_init__(self) -> None:
        start, end = np.datetime64(self.start, "ns"), np.datetime64(self.end, "ns")
        if np.isnat(start) or np.isnat(end) or end < start:
            raise ValueError("weather request needs ordered finite UTC time bounds")
        lo, hi = self.pressure_pa
        if not np.isfinite([lo, hi]).all() or not 0 < lo <= hi:
            raise ValueError("weather request needs ordered positive pressures in Pa")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _times(
    request: WeatherRequest,
    origin: np.datetime64,
    step: np.timedelta64,
    coverage: tuple[np.datetime64, np.datetime64] | None = None,
) -> NDArray[np.datetime64]:
    step_ns = int(step / np.timedelta64(1, "ns"))
    if step_ns <= 0:
        raise ValueError("source time step must be positive")
    epoch = np.datetime64(origin, "ns").astype(np.int64)
    start = (request.start.astype(np.int64) - epoch) // step_ns
    end = -(-(request.end.astype(np.int64) - epoch) // step_ns)
    if coverage is not None and not coverage[0] <= request.start <= request.end <= coverage[1]:
        raise ValueError("weather request exceeds source time coverage")
    if start == end:
        if coverage is not None and epoch + (end + 1) * step_ns > coverage[1].astype(
            "datetime64[ns]"
        ).astype(np.int64):
            start -= 1
        else:
            end += 1  # engine interpolators require two time samples
    return (epoch + np.arange(start, end + 1) * step_ns).astype("datetime64[ns]")


def _check_domain(domain: Domain, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> None:
    centre = (float(lon[0]) + float(lon[-1])) / 2
    shift = 360 * round((centre - (domain.lon_min + domain.lon_max) / 2) / 360)
    if not (
        float(np.min(lat)) <= domain.lat_min <= domain.lat_max <= float(np.max(lat))
        and lon[0] <= domain.lon_min + shift <= domain.lon_max + shift <= lon[-1]
    ):
        raise ValueError("weather request exceeds source spatial coverage")


def _tau_column(column: Any) -> NDArray[np.float64]:
    """Use the optional engine's grid calculation, before vertical cropping."""
    try:
        physics = importlib.import_module("pycontrails.models.tau_cirrus")
    except ImportError as exc:
        raise ImportError("cirrus optical depth requires the optional pycontrails package") from exc
    met = column.to_pycontrails(require_cocip=False)
    return np.asarray(
        physics.tau_cirrus(met).transpose("longitude", "latitude", "level", "time"),
        dtype=np.float64,
    )


def _validate_units(source: GridSource, names: Sequence[str], metadata: Mapping[str, str]) -> None:
    if set(names) - metadata.keys():
        raise ValueError("units must be supplied for every requested field")
    for name in names:
        if name not in source.fields:
            raise ValueError(f"source does not supply {name}")
        native_unit = source.store.array(source.fields[name]).attrs.get("units")
        if native_unit is not None and str(native_unit).replace("**", "") != metadata[name]:
            raise ValueError(
                f"{name}: native units {native_unit!r} do not match {metadata[name]!r}"
            )


def _validate_tau(source: GridSource) -> None:
    native = source.store.read_1d("level").astype(np.int64)
    if not np.array_equal(native, source.vertical.ids):
        raise ValueError("cirrus calculation requires the complete native level axis")
    dependencies = (
        ("t", "ciwc", "geopotential") if "geopotential" in source.fields else ("t", "ciwc")
    )
    _validate_units(source, dependencies, _UNITS)
    try:
        importlib.import_module("pycontrails.models.tau_cirrus")
    except ImportError as exc:
        raise ImportError("cirrus optical depth requires the optional pycontrails package") from exc


def _validate_window(source: GridSource, request: WeatherRequest, names: Sequence[str]) -> None:
    if not names or len(set(names)) != len(names):
        raise ValueError("select unique weather fields")
    if source.vertical.kind != "pressure":
        raise ValueError("native weather windows require pressure levels")
    if (
        not source.vertical.a[0]
        <= request.pressure_pa[0]
        <= request.pressure_pa[1]
        <= source.vertical.a[-1]
    ):
        raise ValueError("weather request exceeds source pressure coverage")


def _check_grids(cubes: Sequence[Cube]) -> None:
    first = cubes[0]
    for cube in cubes[1:]:
        if not all(
            np.array_equal(getattr(cube, a), getattr(first, a)) for a in ("lat", "lon")
        ) or not np.array_equal(cube.vertical.ids, first.vertical.ids):
            raise ValueError("weather window grids must agree across times")


def _provenance(source: GridSource, tau_cirrus: bool) -> dict[str, str]:
    provenance = {"source": source.source_id}
    if source.source_id.startswith("arco-era5-pl"):
        provenance.update(provider="ECMWF", dataset="ERA5", product="reanalysis")
    if tau_cirrus:
        provenance["cirrus_height"] = "geopotential" if "geopotential" in source.fields else "ISA"
    return provenance


def _stack_fields(cubes: Sequence[Cube], names: Sequence[str]) -> dict[str, NDArray[Any]]:
    """Reorder descending-latitude slices into native engine axis order."""
    return {
        name: np.stack([c.fields[name][:, ::-1].transpose(2, 1, 0) for c in cubes], axis=-1)
        for name in names
    }


def materialize_weather(  # noqa: PLR0913 - explicit retrieval/preprocessing options
    source: GridSource,
    request: WeatherRequest,
    *,
    fields: Sequence[str] = ("u", "v", "t", "q", "omega"),
    tau_cirrus: bool = False,
    units: Mapping[str, str] | None = None,
    executor: Executor | None = None,
) -> WeatherCube:
    """Read a pressure-level window, extending only cirrus dependencies upward.

    ``source`` must map the requested aliases to native variables. Deriving
    ``tau_cirrus`` additionally requires ``t`` and ``ciwc``, the complete native
    pressure-level axis, and optional pycontrails. Its height convention is
    pycontrails' ISA approximation unless ``geopotential`` is configured.
    Hybrid-level conversion is deliberately not implicit.
    """
    names = tuple(fields)
    _validate_window(source, request, names)
    if tau_cirrus and "tau_cirrus" in names:
        raise ValueError("choose supplied or derived tau_cirrus, not both")
    if tau_cirrus:
        _validate_tau(source)
    metadata = dict(_UNITS if units is None else units)
    _validate_units(source, names, metadata)
    selected = source.select_fields(names)
    times = _times(request, np.datetime64("1970-01-01"), source.time_step, source.coverage())
    with ThreadPoolExecutor(max_workers=8) if executor is None else nullcontext(executor) as pool:
        cubes = [
            selected.fetch(CubeKey(t), request.domain, request.pressure_pa, executor=pool)
            for t in times
        ]
    first = cubes[0]
    _check_domain(request.domain, first.lat, first.lon)
    _check_grids(cubes)
    values = _stack_fields(cubes, names)
    if tau_cirrus:
        values["tau_cirrus"] = _derive_tau(source, request, first.vertical.ids, times)
        metadata["tau_cirrus"] = "1"
    return WeatherCube(
        lon=first.lon,
        lat=first.lat[::-1],
        level_hpa=first.vertical.ids,
        time_ns=times.astype(np.int64),
        fields=values,
        units={name: metadata[name] for name in values},
        provenance=_provenance(source, tau_cirrus),
    )


def _derive_tau(
    source: GridSource,
    request: WeatherRequest,
    levels: NDArray[np.int64],
    times: NDArray[np.datetime64],
) -> NDArray[np.float64]:
    native = source.store.read_1d("level").astype(np.int64)
    if not np.array_equal(native, source.vertical.ids):
        raise ValueError("cirrus calculation requires the complete native level axis")
    dependencies = ["t", "ciwc"]
    if "geopotential" in source.fields:
        dependencies.append("geopotential")
    # One extra level below the deepest output ensures its height derivative
    # matches that on the complete native column (or uses the native bottom).
    bottom = min(int(np.searchsorted(native, levels[-1])) + 1, len(native) - 1)
    column_request = WeatherRequest(
        request.domain, times[0], times[-1], (float(native[0] * 100), float(native[bottom] * 100))
    )
    column = materialize_weather(source, column_request, fields=dependencies)
    tau = _tau_column(column)
    positions = np.searchsorted(column.level_hpa, levels)
    return tau[:, :, positions, :]


def materialize_radiation(source: ArcoRadiationSource, request: WeatherRequest) -> RadiationCube:
    """Read only bracketing ERA5 midpoint fluxes on the requested domain."""
    times = _times(request, source.time_origin, source.time_step, source.coverage())
    cubes = [source.fetch(CubeKey(t), request.domain) for t in times]
    first = cubes[0]
    _check_domain(request.domain, first.lat, first.lon)
    return RadiationCube(
        lon=first.lon,
        lat=first.lat,
        time_ns=times.astype(np.int64),
        fields={n: np.stack([c.fields[n].T for c in cubes], axis=-1) for n in first.fields},
        units=dict.fromkeys(first.fields, "W m-2"),
        time_semantics="interval_midpoint",
        provenance={
            "source": "ARCO-ERA5",
            "provider": "ECMWF",
            "dataset": "ERA5",
            "product": "reanalysis",
        },
    )
