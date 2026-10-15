"""ERA5 radiation conversion, midpoint interpolation and targeted reads."""

from pathlib import Path

import numpy as np
import pytest

from baro.met.arco_surface import ArcoRadiationSource
from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Domain
from baro.met.rangecache import RangeCache
from baro.met.sources import GridSource
from baro.met.store import ZarrStore
from baro.met.surface import SurfaceEnricher
from baro.met.vertical import pressure_levels
from synthetic import ArraySpec, memory_zarr

SOLAR = "top_net_solar_radiation"
THERMAL = "top_net_thermal_radiation"


def source(unit: str = "J m**-2", attrs: bool = True) -> ArcoRadiationSource:
    flux = np.broadcast_to(np.array([3600, 7200, 10800], dtype="<f4")[:, None, None], (3, 3, 4))
    specs = [
        ArraySpec(
            "time", np.arange(3, dtype="<i8"), (3,), ("time",), {"units": "hours since 2020-01-01"}
        ),
        ArraySpec("level", np.array([200, 300], dtype="<i8"), (2,), ("level",)),
        ArraySpec("latitude", np.array([50, 49, 48], dtype="<f4"), (3,), ("latitude",)),
        ArraySpec("longitude", np.arange(4, dtype="<f4"), (4,), ("longitude",)),
    ]
    for name in (SOLAR, THERMAL):
        specs.append(
            ArraySpec(name, flux, (1, 3, 4), ("time", "latitude", "longitude"), {"units": unit})
        )
    metadata = (
        {"valid_time_start": "2020-01-01T00", "valid_time_stop": "2020-01-01T02"} if attrs else {}
    )
    grid = GridSource(
        "test",
        ZarrStore(memory_zarr(specs, metadata)),
        pressure_levels([200, 300]),
        fields={SOLAR: SOLAR, THERMAL: THERMAL},
        level_array="level",
    )
    return ArcoRadiationSource(grid_source=grid)


def test_midpoints_both_sides_of_half_hour() -> None:
    src = source()
    result = SurfaceEnricher(src).enrich(
        np.array(["2020-01-01T00:15", "2020-01-01T00:30", "2020-01-01T00:45"]),
        [49.5] * 3,
        [1.5] * 3,
    )
    np.testing.assert_allclose(result[SOLAR], [1.75, 2, 2.25])
    assert result.valid.all()
    assert src.coverage() == (np.datetime64("2019-12-31T23:30"), np.datetime64("2020-01-01T01:30"))


def test_requested_field_only_and_cache(tmp_path: Path) -> None:
    src = source()
    src.use_cache(RangeCache(tmp_path))
    key = CubeKey(np.datetime64("2020-01-01T00:30"))
    cube = src.fetch(key, Domain(49, 50, 1, 2), fields=[SOLAR])
    assert set(cube.fields) == {SOLAR}
    np.testing.assert_allclose(cube.fields[SOLAR], 2)
    before = sum(s.bytes for s in src.stats)
    src.fetch(key, Domain(49, 50, 1, 2), fields=[SOLAR])
    assert sum(s.bytes for s in src.stats) == before
    src.tiling(key)


def test_watts_not_divided_twice() -> None:
    cube = source("W m**-2").fetch(CubeKey(np.datetime64("2020-01-01T00:30")), Domain(49, 50, 1, 2))
    np.testing.assert_allclose(cube.fields[SOLAR], 7200)


def test_refuses_bad_units_keys_fields_and_missing_coverage() -> None:
    with pytest.raises(ValueError, match="unit"):
        source("unknown")
    with pytest.raises(ValueError):
        ArcoRadiationSource(fields=[])
    with pytest.raises(DataUnavailableError):
        source(attrs=False).coverage()
    src = source()
    domain = Domain(49, 50, 1, 2)
    with pytest.raises(ValueError, match="midpoint"):
        src.fetch(CubeKey(np.datetime64("2020-01-01T00")), domain)
    with pytest.raises(ValueError, match="configured"):
        src.fetch(CubeKey(np.datetime64("2020-01-01T00:30")), domain, fields=["u"])
    with pytest.raises(DataUnavailableError):
        src.fetch(CubeKey(np.datetime64("2020-01-04T00:30")), domain)


def test_radiation_view_excludes_unrelated_fields_and_rejects_wrong_dimensions() -> None:
    grid = source().grid
    grid.fields = {"u": "nonexistent", **grid.fields}
    radiation = ArcoRadiationSource(grid_source=grid)
    assert tuple(radiation.grid.fields) == (SOLAR, THERMAL)
    assert "u" in grid.fields
    grid.fields.pop(SOLAR)
    with pytest.raises(ValueError, match="configured"):
        ArcoRadiationSource(grid_source=grid)
    grid.fields[SOLAR] = SOLAR
    grid.store.array(SOLAR).attrs["_ARRAY_DIMENSIONS"] = ["time", "level", "latitude", "longitude"]
    with pytest.raises(ValueError, match="dimensions"):
        ArcoRadiationSource(grid_source=grid)
