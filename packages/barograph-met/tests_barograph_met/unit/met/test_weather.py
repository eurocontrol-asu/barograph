"""Engine-independent grid validation and optional export parity."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from baro.met.weather import COCIP_FIELDS, MET_FIELDS, RAD_FIELDS, RadiationCube, WeatherCube


def met_kwargs() -> dict[str, Any]:
    return {
        "lon": [1, 2],
        "lat": [40, 41],
        "level_hpa": [200, 300],
        "time_ns": np.array(
            ["2020-01-01T00:00", "2020-01-01T03:00"], dtype="datetime64[ns]"
        ).astype("int64"),
        "fields": {k: np.arange(16.0).reshape(2, 2, 2, 2) for k in COCIP_FIELDS},
        "units": {k: MET_FIELDS[k][1] for k in COCIP_FIELDS},
        "provenance": {"provider": "ECMWF", "dataset": "ERA5", "product": "reanalysis"},
    }


def rad_kwargs() -> dict[str, Any]:
    kwargs = met_kwargs()
    kwargs.pop("level_hpa")
    kwargs.update(
        fields={k: np.ones((2, 2, 2)) for k in RAD_FIELDS},
        units=dict.fromkeys(RAD_FIELDS, "W m-2"),
        time_semantics="interval_midpoint",
    )
    return kwargs


def test_engine_dict_dimensions_and_time() -> None:
    cube = WeatherCube(**met_kwargs())
    arrays = cube.to_diffcocip()
    assert arrays["eastward_wind"].shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(arrays["level"], [200, 300])
    assert np.diff(arrays["time_s"])[0] == 10800
    np.testing.assert_array_equal(arrays["hours_of_day"], [0, 3])
    assert cube.coverage["time"][0] == np.datetime64("2020-01-01")
    arrays["eastward_wind"][0, 0, 0, 0] = -1
    assert np.asarray(cube.fields["u"])[0, 0, 0, 0] == 0


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("lon", [1, 1], "ascending"),
        ("lat", [90, 91], "latitude"),
        ("level_hpa", [-1, 1], "positive"),
        ("time_ns", [1.0, 2.0], "signed UTC"),
        ("time_ns", [2, 1], "ascending"),
        ("time_ns", [np.iinfo(np.int64).min, 1], "NaT"),
        ("time_semantics", "interval_midpoint", "valid timestamps"),
    ],
)
def test_bad_coordinates(key: str, value: Any, match: str) -> None:
    kwargs = met_kwargs()
    kwargs[key] = value
    with pytest.raises(ValueError, match=match):
        WeatherCube(**kwargs)


@pytest.mark.parametrize("value", [np.zeros((2, 2)), np.full((2, 2, 2, 2), np.nan)])
def test_invalid_fields(value: Any) -> None:
    kwargs = met_kwargs()
    kwargs["fields"]["u"] = value
    with pytest.raises(ValueError, match="shape"):
        WeatherCube(**kwargs)


def test_units_and_incomplete_weather() -> None:
    kwargs = met_kwargs()
    kwargs["units"]["omega"] = "m s-1"
    with pytest.raises(ValueError, match="units"):
        WeatherCube(**kwargs)
    kwargs = met_kwargs()
    del kwargs["fields"]["tau_cirrus"]
    with pytest.raises(ValueError, match="identical"):
        WeatherCube(**kwargs)
    del kwargs["units"]["tau_cirrus"]
    cube = WeatherCube(**kwargs)
    with pytest.raises(ValueError, match="tau_cirrus"):
        cube.to_diffcocip()


def test_singleton_and_old_epoch_rejected_by_diffcocip() -> None:
    kwargs = met_kwargs()
    kwargs["lon"] = [1]
    kwargs["fields"] = {k: v[:1] for k, v in kwargs["fields"].items()}
    with pytest.raises(ValueError, match="two points"):
        WeatherCube(**kwargs).to_diffcocip()
    kwargs = met_kwargs()
    kwargs["time_ns"] = np.array([0, 1], dtype="int64")
    with pytest.raises(ValueError, match="2000"):
        WeatherCube(**kwargs).to_diffcocip()


def test_radiation_no_implicit_shift_or_conversion() -> None:
    kwargs = rad_kwargs()
    cube = RadiationCube(**kwargs)
    np.testing.assert_array_equal(
        cube.to_diffcocip()["time_s"], WeatherCube(**met_kwargs()).to_diffcocip()["time_s"]
    )
    kwargs["time_semantics"] = "interval_end"
    with pytest.raises(ValueError, match="Resolve radiation"):
        RadiationCube(**kwargs)
    kwargs["time_semantics"] = "valid"
    kwargs["units"]["top_net_solar_radiation"] = "J m-2"
    with pytest.raises(ValueError, match="units"):
        RadiationCube(**kwargs)


@pytest.mark.skipif(importlib.util.find_spec("pycontrails") is None, reason="optional engine")
def test_pycontrails_preserves_radiation_time_and_values() -> None:
    process_rad = importlib.import_module("pycontrails.models.cocip.cocip")._process_rad

    cube = RadiationCube(**rad_kwargs())
    rad = cube.to_pycontrails()
    processed = process_rad(rad)
    np.testing.assert_array_equal(processed.data.time.values.astype("int64"), cube.time_ns)
    assert processed.data.top_net_solar_radiation.attrs["units"] == "W m**-2"
    met = WeatherCube(**met_kwargs()).to_pycontrails()
    np.testing.assert_array_equal(met.data.eastward_wind.values, met_kwargs()["fields"]["u"])
    assert met.data.attrs["provider"] == "ECMWF"


@pytest.mark.skipif(importlib.util.find_spec("diffcocip") is None, reason="optional engine")
def test_diffcocip_loader_parity(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    engine = importlib.import_module("diffcocip.met")
    load_met, load_rad = engine.load_met, engine.load_rad

    for cube, loader, name in [
        (WeatherCube(**met_kwargs()), load_met, "met"),
        (RadiationCube(**rad_kwargs()), load_rad, "rad"),
    ]:
        path = tmp_path / f"{name}.nc"
        cube.to_pycontrails().data.to_netcdf(path)
        expected = loader(path)
        actual = cube.to_diffcocip()
        assert set(actual) == set(expected)
        for field in expected:
            np.testing.assert_array_equal(actual[field], expected[field])


def test_partial_pycontrails_export() -> None:
    if importlib.util.find_spec("pycontrails") is None:
        pytest.skip("optional engine")
    kwargs = met_kwargs()
    kwargs["fields"] = {"t": kwargs["fields"]["t"]}
    kwargs["units"] = {"t": "K"}
    cube = WeatherCube(**kwargs)
    with pytest.raises(ValueError, match="Missing required"):
        cube.to_pycontrails()
    assert list(cube.to_pycontrails(require_cocip=False).data.data_vars) == ["air_temperature"]
