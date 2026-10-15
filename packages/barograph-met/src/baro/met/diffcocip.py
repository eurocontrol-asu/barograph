"""Optional, forward-only diffcocip execution with checked weather coverage.

The adapter targets diffcocip's v1 formation/evolution API. Retrieval runs on
host Python, outside JAX transformations. No global engine hooks are installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

from baro.met.base import DataUnavailableError

Array = npt.NDArray[np.float64]
Weather = tuple[Mapping[str, Any], Mapping[str, Any]]
_MET = (
    "air_temperature",
    "specific_humidity",
    "eastward_wind",
    "northward_wind",
    "lagrangian_tendency_of_air_pressure",
    "tau_cirrus",
)
_RAD = ("top_net_solar_radiation", "top_net_thermal_radiation")
_AXES = ("lon", "lat", "level", "time_s")
_UNSUPPORTED = (
    "radiative_heating_effects",
    "revised_contrail_ice_budget",
    "contrail_contrail_overlapping",
    "parametric_rf_model_s2025",
    "unterstrasser_ice_survival_fraction",
    "vpm_activation",
)


@dataclass(frozen=True)
class WeatherDemand:
    """Exact probe coordinates to cover, including interpolation neighbours.

    ``points`` columns are longitude degrees, latitude degrees, pressure hPa,
    and seconds since 2000-01-01. Radiation ignores the pressure column.
    A provider returns complete diffcocip met/radiation dictionaries covering
    all probes in this request. Radiation values must be W/m² at midpoints.
    Spatial/time bounds are requests, never permission to extrapolate.
    """

    points: Array

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Minimum/maximum requested coordinates, in diffcocip units."""
        return {
            axis: (float(self.points[:, i].min()), float(self.points[:, i].max()))
            for i, axis in enumerate(_AXES)
        }


@dataclass(frozen=True)
class EvolutionResult:
    """Forward result; ``age_limit_reached`` is conservative, never extinction.

    Diagnostics contain only executed step ends; trailing extinct steps from
    the dense scan are omitted. ``energy_forcing_j`` sums the same hard EF.
    """

    energy_forcing_j: float
    stop_reason: str
    formation: Mapping[str, Any]
    final_state: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    times: npt.NDArray[np.datetime64]
    weather_requests: int


def _points(lon: Any, lat: Any, level: Any, time: Any) -> Array:
    arrays = np.broadcast_arrays(lon, lat, level, time)
    return np.column_stack([a.ravel() for a in arrays]).astype(np.float64)


def _axis_contains(data: Mapping[str, Any], axis: str, points: Array) -> bool:
    if axis not in data:
        return False
    coord = np.asarray(data[axis])
    if coord.ndim != 1 or len(coord) < 2 or not np.isfinite(coord).all():  # noqa: PLR2004
        raise ValueError(f"Invalid diffcocip coordinate: {axis}")
    if not np.all(np.diff(coord) > 0):
        raise ValueError(f"diffcocip coordinate must increase: {axis}")
    query = points[:, _AXES.index(axis)]
    return bool(np.all(query >= coord[0]) and np.all(query <= coord[-1]))


def _grid_contains(
    data: Mapping[str, Any], fields: tuple[str, ...], axes: tuple[str, ...], points: Array
) -> bool:
    if not all(_axis_contains(data, axis, points) for axis in axes):
        return False
    shape = tuple(len(data[axis]) for axis in axes)
    for field in fields:
        if field not in data:
            return False
        if np.shape(data[field]) != shape:
            raise ValueError(f"Invalid diffcocip field shape: {field}")
    return True


class _Coverage:
    def __init__(self, provider: Callable[[WeatherDemand], Weather]) -> None:
        self.provider = provider
        self.met: Mapping[str, Any] = {}
        self.rad: Mapping[str, Any] = {}
        self.points = np.empty((0, 4))
        self.requests = 0

    def reset(self) -> None:
        self.points = np.empty((0, 4))

    def ensure(self, points: Array) -> bool:
        """Return True if weather was replaced (caller must replay its trial)."""
        if not len(points):
            return False
        if not np.isfinite(points).all():
            raise DataUnavailableError("Nonfinite meteorological probe in active model state")
        if np.any(np.abs(points[:, 1]) > 80):  # noqa: PLR2004
            raise NotImplementedError("diffcocip v1 cannot evolve beyond |latitude|=80 degrees")
        self.points = np.unique(np.concatenate([self.points, points]), axis=0)
        if self._contains(self.points):
            return False
        request_points = self.points.copy()
        request_points.setflags(write=False)
        self.met, self.rad = self.provider(WeatherDemand(request_points))
        self.requests += 1
        if not self._contains(self.points):
            raise DataUnavailableError("Weather provider did not cover all requested probes")
        return True

    def _contains(self, points: Array) -> bool:
        return _grid_contains(self.met, _MET, _AXES, points) and _grid_contains(
            self.rad, _RAD, ("lon", "lat", "time_s"), points
        )

    def temperature(self, points: Array) -> Array:
        interp = import_module("diffcocip.interpolation")
        values, _ = interp.interp4d_from_dict(self.met["air_temperature"], self.met, *points.T)
        return np.asarray(values)

    def check_finite(self, points: Array) -> None:
        """Reject missing field values at required sites before accepting physics."""
        interp = import_module("diffcocip.interpolation")
        for field in _MET:
            values, _ = interp.interp4d_from_dict(self.met[field], self.met, *points.T)
            if not np.isfinite(values).all():
                raise DataUnavailableError(f"Nonfinite weather at required probe: {field}")
        for field in _RAD:
            values, _ = interp.interp3d_from_dict(
                self.rad[field], self.rad, points[:, 0], points[:, 1], points[:, 3]
            )
            if not np.isfinite(values).all():
                raise DataUnavailableError(f"Nonfinite weather at required probe: {field}")


def _lower(points: Array, temperature: Any, dz: float) -> Array:
    thermo = import_module("diffcocip.thermo")
    out = points.copy()
    out[:, 2] = np.asarray(thermo.pressure_dz(temperature, points[:, 2] * 100, dz)) / 100
    return out


def _state_points(state: Mapping[str, Any], mask: Any) -> Array:
    return _points(state["longitude"], state["latitude"], state["level"], state["time_s"])[
        np.asarray(mask, dtype=bool)
    ]


def _formation(flight: dict[str, Any], params: dict[str, Any], weather: _Coverage) -> Any:
    formation = import_module("diffcocip.formation")
    units = import_module("diffcocip.units")
    bridge = import_module("diffcocip.met").time_bridge(flight["time"])
    original = _points(
        flight["longitude"], flight["latitude"], units.m_to_pl(flight["altitude"]), bridge["time_s"]
    )
    weather.ensure(original)
    lower = _lower(original, weather.temperature(original), params["dz_m"])
    weather.ensure(lower)
    weather.check_finite(np.concatenate([original, lower]))
    for _ in range(8):
        state = formation.initial_contrail_state(flight, weather.met, params)
        # Downwash is evaluated even for unformed lanes. Engine-invalid input
        # lanes can emit NaN level_1; their physical rejection is kept intact.
        downwash = original.copy()
        downwash[:, 2] = np.asarray(state["level_1"])
        downwash = downwash[np.isfinite(downwash).all(axis=1)]
        if not weather.ensure(downwash):
            weather.check_finite(downwash)
            return state
    raise DataUnavailableError("Downwash weather coverage did not converge")


def _prepare_state(
    state: Mapping[str, Any], mask: Any, params: dict[str, Any], weather: _Coverage
) -> None:
    points = _state_points(state, mask)
    weather.ensure(points)
    lower = _lower(points, np.asarray(state["air_temperature"])[mask], params["dz_m"])
    weather.ensure(lower)
    weather.check_finite(np.concatenate([points, lower]))


def _checked_step(
    state: dict[str, Any],
    step_time: dict[str, Any],
    params: dict[str, Any],
    weather: _Coverage,
    active: Any,
) -> Any:
    evolution = import_module("diffcocip.evolution")
    units = import_module("diffcocip.units")
    weather.reset()
    _prepare_state(state, active, params, weather)
    for _ in range(8):
        # step returns new dictionaries. Rejected trial state is never carried.
        candidate, diag = evolution.step(state, step_time, weather.met, weather.rad, params)
        endpoint = _points(
            diag["longitude"],
            diag["latitude"],
            units.m_to_pl(diag["altitude"]),
            step_time["time_s"],
        )[active]
        if weather.ensure(endpoint):
            continue
        lower = _lower(endpoint, weather.temperature(endpoint), params["dz_m"])
        if weather.ensure(lower):
            continue
        weather.check_finite(np.concatenate([endpoint, lower]))
        return candidate, diag
    raise DataUnavailableError("Evolution weather coverage did not converge")


def _validate_flight(flight: Mapping[str, Any]) -> None:
    times = np.asarray(flight["time"])
    if times.ndim != 1 or not times.size or not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("Flight needs a nonempty one-dimensional datetime array")
    if np.isnat(times).any() or np.any(times[1:] < times[:-1]):
        raise ValueError("Flight timestamps must be valid and nondecreasing")
    lanes = (
        "longitude",
        "latitude",
        "altitude",
        "true_airspeed",
        "fuel_flow",
        "engine_efficiency",
        "aircraft_mass",
        "nvpm_ei_n",
        "wingspan",
    )
    for name in lanes:
        if np.shape(flight[name]) != times.shape:
            raise ValueError(f"Flight {name} must have one value per timestamp")
    longitude = np.asarray(flight["longitude"])
    if np.any(np.abs(longitude) > 180):  # noqa: PLR2004
        raise ValueError("Flight longitudes must use [-180, 180] degrees")


def _validate_engine_options(options: Mapping[str, Any]) -> None:
    for key in _UNSUPPORTED:
        if options.get(key, False):
            raise NotImplementedError(f"diffcocip v1 does not support {key}")
    if options.get("persistent_buffer") is not None or options.get("smoothing_beta") is not None:
        raise NotImplementedError("Forward adaptive execution supports hard persistence only")


def _validate_steps(options: Mapping[str, Any]) -> None:
    for name in ("dt_integration_s", "max_age_s"):
        value = options[name]
        if not np.isfinite(value) or not 1e-9 <= value < np.iinfo(np.int64).max / 1e9:  # noqa: PLR2004
            raise ValueError(f"{name} must be positive and representable in nanoseconds")
    if not np.isfinite(options["dz_m"]) or options["dz_m"] <= 0:
        raise ValueError("dz_m must be positive and finite")


def _validate(
    flight: dict[str, Any], params: Mapping[str, Any], max_age_s: float
) -> dict[str, Any]:
    if not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_age_s must be positive and finite")
    options = dict(params, max_age_s=max_age_s)
    _validate_engine_options(options)
    _validate_steps(options)
    _validate_flight(flight)
    return options


def _activation(state: Mapping[str, Any], end: float, previous: float) -> tuple[Any, Any, Any]:
    ft = np.asarray(state["formation_time_s"])
    formed = np.asarray(state["formed"]) == 1
    new = (ft < end) & (ft >= previous) & formed
    active = new | (np.asarray(state["alive"]) == 1)
    return ft, active, (ft >= end) & formed


def _initialize(
    flight: dict[str, Any], options: dict[str, Any], weather: _Coverage
) -> tuple[Any, Any, Any]:
    evolution = import_module("diffcocip.evolution")
    formed = _formation(flight, options, weather)
    # Check downwash lower probes before accepting initial radiative properties.
    seed = {
        "longitude": formed["longitude"],
        "latitude": formed["latitude"],
        "level": formed["level_1"],
        "time_s": formed["formation_time_s"],
        "air_temperature": formed["air_temperature_1"],
    }
    _prepare_state(seed, np.asarray(formed["formed"]) == 1, options, weather)
    state = evolution.initial_evolution_state(formed, weather.met, weather.rad, options)
    times = evolution.set_timesteps(
        np.asarray(flight["time"]), options["dt_integration_s"], options["max_age_s"]
    )[:-1]
    return formed, state, times


def _evolve(
    state: dict[str, Any], times: Any, options: dict[str, Any], weather: _Coverage
) -> tuple[Any, Any, Any, bool, float]:
    evolution = import_module("diffcocip.evolution")
    tb = import_module("diffcocip.met").time_bridge(times)
    previous = -np.inf
    records: list[dict[str, Any]] = []
    executed = []
    cap = False
    for i, timestamp in enumerate(times):
        end = tb["time_s"][i]
        ft, active, pending = _activation(state, end, previous)
        if not active.any() and not pending.any():
            break
        step_time = {key: tb[key][i] for key in ("time_s", "days_since_2000", "hours_of_day")}
        step_time["t_prev_s"] = previous
        if active.any():
            cap = cap or bool(np.any((end - ft[active]) >= options["max_age_s"]))
            state, diag = _checked_step(state, step_time, options, weather, active)
        else:
            state, diag = evolution.step(state, step_time, weather.met, weather.rad, options)
        records.append({key: np.asarray(value) for key, value in diag.items()})
        executed.append(timestamp)
        previous = end
    return state, records, executed, cap, previous


def _limit_reached(state: Mapping[str, Any], previous: float) -> bool:
    remaining = (np.asarray(state["formed"]) == 1) & (
        np.asarray(state["formation_time_s"]) >= previous
    )
    return bool(np.any(np.asarray(state["alive"]) == 1) or np.any(remaining))


def run_diffcocip(
    flight: dict[str, Any],
    provider: Callable[[WeatherDemand], Weather],
    params: Mapping[str, Any],
    *,
    max_age_s: float,
) -> EvolutionResult:
    """Evolve one flight with demand-driven weather, stopping at extinction/cap.

    ``provider`` supplies native-grid dicts in diffcocip's documented units.
    It may use Barograph caches and return larger coverage than requested.
    Requests include lower meteorological probes and wake downwash. Trial
    steps outside coverage are discarded and replayed with complete weather.
    The input flight follows diffcocip's dict contract (altitude in metres).
    This host driver does not support autodifferentiation or smoothing.
    Provider responses must describe one consistent meteorological dataset:
    extending a window must not change values at previously covered nodes.
    Longitudes use [-180, 180]; seam crossings can request a wide domain.
    ``max_age_s`` is mandatory and overrides the corresponding params value.
    """
    options = _validate(flight, params, max_age_s)
    weather = _Coverage(provider)
    formed, state, times = _initialize(flight, options, weather)
    state, records, executed, cap, previous = _evolve(state, times, options, weather)
    cap = cap or _limit_reached(state, previous)
    diagnostics = (
        {key: np.stack([record[key] for record in records]) for key in records[0]}
        if records
        else {}
    )
    return EvolutionResult(
        float(np.sum(diagnostics.get("ef", 0))),
        "age_limit_reached" if cap else "extinct",
        formed,
        state,
        diagnostics,
        np.asarray(executed, dtype="datetime64[ns]"),
        weather.requests,
    )
