"""Optional engine parity on synthetic native grids; no network access."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from baro.met.base import DataUnavailableError
from baro.met.diffcocip import WeatherDemand, run_diffcocip


def test_demand_bounds() -> None:
    request = WeatherDemand(np.array([[1.0, 2.0, 250.0, 0.0], [3.0, 4.0, 300.0, 3600.0]]))
    assert request.bounds["level"] == (250.0, 300.0)


def test_invalid_cap_without_engine() -> None:
    with pytest.raises(ValueError, match="max_age"):
        run_diffcocip({}, lambda _: ({}, {}), {}, max_age_s=0)


@pytest.fixture
def scenario() -> Any:

    n = 3
    flight: dict[str, Any] = {
        key: np.full(n, value)
        for key, value in {
            "latitude": 50.0,
            "altitude": 11000.0,
            "true_airspeed": 230.0,
            "fuel_flow": 1.0,
            "engine_efficiency": 0.35,
            "aircraft_mass": 65000.0,
            "nvpm_ei_n": 1e15,
            "wingspan": 35.8,
        }.items()
    }
    flight.update(
        longitude=np.array([0.0, 0.1, 0.2]),
        time=np.datetime64("2020-01-01T12:00") + np.arange(n) * np.timedelta64(1, "m"),
        ei_h2o=1.23,
        q_fuel=43130000.0,
    )
    params = {
        "dz_m": 200.0,
        "effective_vertical_resolution": 2000.0,
        "wind_shear_enhancement_exponent": 0.5,
        "turbulent_vertical_velocity_scale": 0.1,
        "initial_wake_vortex_depth": 0.5,
        "min_ice_particle_number_nvpm_ei_n": 1e10,
        "sedimentation_impact_factor": 0.5,
        "max_depth": 1500.0,
        "max_horizontal_diffusivity": None,
        "max_vertical_diffusivity": None,
        "rf_lw_enhancement_factor": 1.0,
        "rf_sw_enhancement_factor": 1.0,
        "habit_distributions": np.eye(8)[np.array([6, 6, 6, 6, 6, 6])],
        "radius_threshold_um": np.array([5.0, 9.5, 23.0, 190.0, 310.0]),
        "max_age_s": 900.0,
        "dt_integration_s": 300.0,
        "min_tau": 1e-6,
        "max_tau": 1e10,
        "min_n_ice_per_m3": 1000.0,
        "max_n_ice_per_m3": 1e20,
        "min_altitude_m": 6000.0,
        "max_altitude_m": 13000.0,
        "max_seg_length_m": 40000.0,
        "rhi_adj": 0.97,
        "rhi_boost_exponent": 1.7,
        "clip_upper": 1.7,
    }
    times = (
        np.array(["2020-01-01T11:00", "2020-01-01T14:00"], dtype="datetime64[s]")
        - np.datetime64("2000-01-01")
    ).astype(float)
    met = {
        "lon": np.linspace(-2, 2, 41),
        "lat": np.array([49.0, 51.0]),
        "level": np.array([150.0, 200.0, 250.0, 300.0, 400.0]),
        "time_s": times,
    }
    for field, value in {
        "air_temperature": 220.0,
        "specific_humidity": 0.0002,
        "eastward_wind": 60.0,
        "northward_wind": 0.0,
        "lagrangian_tendency_of_air_pressure": 0.0,
        "tau_cirrus": 0.1,
    }.items():
        met[field] = np.full((41, 2, 5, 2), value)
    rad = {key: met[key] for key in ["lon", "lat", "time_s"]}
    rad.update(
        top_net_solar_radiation=np.full((41, 2, 2), 400.0),
        top_net_thermal_radiation=np.full((41, 2, 2), -250.0),
    )
    return flight, params, met, rad


def test_adaptive_matches_dense_with_extensions(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    jnp = import_module("jax.numpy")
    cocip_ef = import_module("diffcocip.model").cocip_ef

    flight, params, met, rad = scenario
    params["habit_distributions"] = jnp.asarray(params["habit_distributions"])
    reference, aux = cocip_ef(flight, met, rad, params, return_aux=True)
    calls = []

    def provider(request: WeatherDemand) -> Any:
        calls.append(request)
        lo, hi = request.bounds["lon"]
        left = max(0, np.searchsorted(met["lon"], lo, side="right") - 1)
        right = min(len(met["lon"]), np.searchsorted(met["lon"], hi) + 2)
        out_met = {
            k: (v[left:right] if k not in ("lat", "level", "time_s") else v) for k, v in met.items()
        }
        out_rad = {k: (v[left:right] if k not in ("lat", "time_s") else v) for k, v in rad.items()}
        return out_met, out_rad

    result = run_diffcocip(flight, provider, params, max_age_s=900.0)
    assert len(calls) > 1
    assert np.any(aux["formation"]["formed"])
    assert abs(float(reference)) > 0
    np.testing.assert_allclose(result.energy_forcing_j, reference, rtol=1e-10)
    np.testing.assert_array_equal(result.diagnostics["alive"], aux["diags"]["alive"])
    np.testing.assert_allclose(
        result.diagnostics["longitude"], aux["diags"]["longitude"], atol=1e-12
    )
    assert result.stop_reason == "age_limit_reached"


def test_missing_radiation_is_error(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    broken = {**rad, "time_s": rad["time_s"] + 86400}
    with pytest.raises(DataUnavailableError, match="cover"):
        run_diffcocip(flight, lambda _: (met, broken), params, max_age_s=900.0)


def test_nonfinite_weather_is_error(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    broken = {**met, "specific_humidity": np.full_like(met["specific_humidity"], np.nan)}
    with pytest.raises(DataUnavailableError, match="Nonfinite"):
        run_diffcocip(flight, lambda _: (broken, rad), params, max_age_s=900.0)


def test_polar_input_is_error(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    flight["latitude"] = np.full(3, 81.0)
    with pytest.raises(NotImplementedError, match="latitude"):
        run_diffcocip(flight, lambda _: (met, rad), params, max_age_s=900.0)


def test_future_formations_survive_empty_interval(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    jnp = import_module("jax.numpy")
    cocip_ef = import_module("diffcocip.model").cocip_ef

    flight, params, met, rad = scenario
    flight["time"] = flight["time"][0] + np.array([0, 1800, 1860]) * np.timedelta64(1, "s")
    params["habit_distributions"] = jnp.asarray(params["habit_distributions"])
    reference, aux = cocip_ef(flight, met, rad, params, return_aux=True)
    result = run_diffcocip(flight, lambda _: (met, rad), params, max_age_s=900.0)
    np.testing.assert_allclose(result.energy_forcing_j, reference, rtol=1e-10)
    assert result.times[-1] > flight["time"][1]
    assert np.any(np.sum(result.diagnostics["alive"], axis=1) == 0)
    np.testing.assert_array_equal(result.diagnostics["alive"], aux["diags"]["alive"])


def test_no_formation_is_extinct(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    warm = {**met, "air_temperature": np.full_like(met["air_temperature"], 280.0)}
    result = run_diffcocip(flight, lambda _: (warm, rad), params, max_age_s=900.0)
    assert result.stop_reason == "extinct"
    assert result.energy_forcing_j == 0
    assert not len(result.times)


def test_smoothing_refused(scenario: Any) -> None:
    flight, params, met, rad = scenario
    with pytest.raises(NotImplementedError, match="hard persistence"):
        run_diffcocip(
            flight, lambda _: (met, rad), {**params, "smoothing_beta": 1.0}, max_age_s=900.0
        )


def test_coverage_reuses_window(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    requests = []

    def provider(demand: WeatherDemand) -> Any:
        requests.append(demand)
        return met, rad

    result = run_diffcocip(flight, provider, params, max_age_s=900.0)
    assert len(result.times) > 1  # multiple physical steps reuse the same window
    assert result.weather_requests == 1
    assert len(requests) == 1
    assert not requests[0].points.flags.writeable


@pytest.fixture
def coordinate_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordinate-only stand-in: malformed weather fails before any physics.

    These public-boundary rejection tests run without the optional JAX engine.
    No formation/evolution behavior is mocked or used as a physics oracle.
    """
    modules = {
        "diffcocip.evolution": SimpleNamespace(),
        "diffcocip.formation": SimpleNamespace(),
        "diffcocip.units": SimpleNamespace(m_to_pl=lambda altitude: np.full_like(altitude, 250.0)),
        "diffcocip.met": SimpleNamespace(
            time_bridge=lambda times: {
                "time_s": (times - np.datetime64("2000-01-01")) / np.timedelta64(1, "s")
            }
        ),
    }
    monkeypatch.setattr("baro.met.diffcocip.import_module", modules.__getitem__)


@pytest.mark.parametrize("broken", ["descending", "singleton", "shape", "missing"])
def test_coverage_rejects_invalid_provider(
    scenario: Any, broken: Any, coordinate_engine: None
) -> None:
    flight, params, met, rad = scenario
    met = dict(met)
    if broken == "descending":
        met["lon"] = met["lon"][::-1]
    if broken == "singleton":
        met["lon"] = met["lon"][:1]
    if broken == "shape":
        met["air_temperature"] = np.zeros((2, 2))
    if broken == "missing":
        met.pop("specific_humidity")
    with pytest.raises((ValueError, DataUnavailableError)):
        run_diffcocip(flight, lambda _: (met, rad), params, max_age_s=900.0)


def test_coverage_invalid_probe_without_engine(scenario: Any, coordinate_engine: None) -> None:
    flight, params, _, _ = scenario
    with pytest.raises(DataUnavailableError, match="Nonfinite"):
        run_diffcocip(
            {**flight, "latitude": np.full(3, np.nan)}, lambda _: ({}, {}), params, max_age_s=900.0
        )
    with pytest.raises(NotImplementedError, match="latitude"):
        run_diffcocip(
            {**flight, "latitude": np.full(3, 81.0)}, lambda _: ({}, {}), params, max_age_s=900.0
        )


@pytest.mark.parametrize(
    "key,value", [("dz_m", 0.0), ("dz_m", float("nan")), ("dt_integration_s", 1e-12)]
)
def test_invalid_physical_step(scenario: Any, key: Any, value: Any) -> None:
    flight, params, _, _ = scenario
    with pytest.raises(ValueError, match=key):
        run_diffcocip(flight, lambda _: ({}, {}), {**params, key: value}, max_age_s=900.0)


def test_bad_flight_shapes_and_time_order(scenario: Any) -> None:
    flight, params, _, _ = scenario
    with pytest.raises(ValueError, match="nondecreasing"):
        run_diffcocip(
            {**flight, "time": flight["time"][::-1]}, lambda _: ({}, {}), params, max_age_s=900.0
        )
    with pytest.raises(ValueError, match="one value"):
        run_diffcocip(
            {**flight, "altitude": flight["altitude"][:1]},
            lambda _: ({}, {}),
            params,
            max_age_s=900.0,
        )


def test_unactivated_formation_is_not_extinct(scenario: Any) -> None:
    pytest.importorskip("diffcocip")
    flight, params, met, rad = scenario
    flight = {
        key: (np.asarray(value)[:1] if np.ndim(value) else value) for key, value in flight.items()
    }
    result = run_diffcocip(flight, lambda _: (met, rad), params, max_age_s=1.0)
    assert np.any(result.formation["formed"])
    assert result.stop_reason == "age_limit_reached"
    assert result.energy_forcing_j == 0
