"""Surface archive range reads, native periods and cache, using only local mirrors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from baro.met import (
    CubeKey,
    Domain,
    Flag,
    OpenMeteoSource,
    OpenMeteoSurfaceSource,
    RangeCache,
    SurfaceEnricher,
)
from baro.met.base import DataUnavailableError
from synthetic import EPOCH, OM_MODEL, write_openmeteo_mirror, write_openmeteo_surface_mirror

pytestmark = pytest.mark.integration
pytest.importorskip("omfiles")


@pytest.fixture(scope="module", params=[1, 3])
def surface_mirror(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, int]:
    step = request.param
    return write_openmeteo_surface_mirror(tmp_path_factory.mktemp(f"surface-{step}"), step), step


def test_surface_native_steps_layouts_and_fractional_periods(
    surface_mirror: tuple[Path, int],
) -> None:
    mirror, step = surface_mirror
    with OpenMeteoSurfaceSource(OM_MODEL, base_url=str(mirror), tile_deg=1) as source:
        first, last = source.coverage()
        boundary = first + np.timedelta64(24 * step, "h")
        times = np.array(
            [boundary - np.timedelta64(30, "m"), boundary, boundary + np.timedelta64(30, "m"), last]
        )
        result = source.enrich(times, 45, 2)
        assert result.valid.all()
        hours = (times - EPOCH) / np.timedelta64(1, "h")
        np.testing.assert_allclose(result["u10"], 2 + 4.5 + 0.4 + 0.5 * hours, atol=0.02)
        np.testing.assert_allclose(result["v10"], 3 - 4.5 + 0.1 - 0.2 * hours, atol=0.02)
        np.testing.assert_allclose(result["t2m"], 273.15 + 10 + 13.5 + 0.5 * hours, atol=0.02)
        ends = result.interval_end["precipitation"]
        native_hours = (ends - EPOCH) / np.timedelta64(1, "h")
        np.testing.assert_allclose(result["precipitation"], native_hours % 5, atol=0.02)
        np.testing.assert_allclose(
            result["rain"], np.maximum(native_hours % 5 - 0.5 * (native_hours % 3), 0), atol=0.02
        )
        np.testing.assert_allclose(result["gusts"], 10 + native_hours % 7, atol=0.02)
        np.testing.assert_array_equal(
            ends - result.interval_start["precipitation"], np.timedelta64(step, "h")
        )
        np.testing.assert_array_equal(source.enrich(times, 45, 2).fields["u10"], result["u10"])
        assert source.enrich(times, 45, 2).stats["bytes"] == 0
        assert np.isnat(result.run).all()
        assert source.chunk_hours == 24 * step
        assert source.reader._cache_bytes > 0


def test_select_only_requested_dependencies(
    surface_mirror: tuple[Path, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror, _ = surface_mirror
    with OpenMeteoSurfaceSource(OM_MODEL, base_url=str(mirror)) as source:
        calls: list[list[tuple[str, int]]] = []
        original = source.reader.blocks

        def blocks(
            wanted: list[tuple[str, int]], chunk: int, rows: np.ndarray, cols: np.ndarray
        ) -> Any:
            calls.append(wanted)
            return original(wanted, chunk, rows, cols)

        monkeypatch.setattr(source.reader, "blocks", blocks)
        first, _ = source.coverage()
        cube = source.fetch(CubeKey(first), Domain.from_points([45], [2], 0), fields=["u10"])
        assert set(cube.fields) == {"u10"}
        assert calls == [[("wind_u_component_10m", 0)]]
        calls.clear()
        source.enrich(first + np.timedelta64(30, "m"), 45, 2)
        assert calls[0] == [
            ("temperature_2m", 0),
            ("wind_u_component_10m", 0),
            ("wind_v_component_10m", 0),
        ]
        assert ("precipitation", 0) in calls[1] and ("snowfall_water_equivalent", 0) in calls[1]


def test_surface_cache_mutability_and_disk_finality(
    surface_mirror: tuple[Path, int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror, _ = surface_mirror
    with OpenMeteoSurfaceSource(
        OM_MODEL, base_url=str(mirror), fields={"u10": "u10"}, cache_mb=1.0
    ) as source:
        source.use_cache(RangeCache(tmp_path))
        assert source.reader.disk is not None
        assert source.reader.disk.cacheable("data/model/wind_u_component_10m/chunk_0.om")
        assert not source.reader.disk.cacheable("data/model/wind_u_component_10m/chunk_99999999.om")
        source.reader.final = lambda _: False
        time = source.coverage()[0]
        assert source.enrich(time, 45, 2).stats["bytes"] > 0
        assert source.enrich(time, 45, 2).stats["bytes"] == 0
        monkeypatch.setattr("baro.met.openmeteo.REVALIDATE_S", 0)
        assert source.enrich(time, 45, 2).stats["bytes"] > 0


def test_native_fetch_validation_aliases_and_coverage(surface_mirror: tuple[Path, int]) -> None:
    mirror, step = surface_mirror
    with OpenMeteoSurfaceSource(OM_MODEL, base_url=str(mirror), fields={"wind": "u10"}) as source:
        first, last = source.coverage()
        domain = Domain.from_points([45], [2], 0)
        with pytest.raises(DataUnavailableError):
            source.fetch(CubeKey(last + source.time_step), domain)
        with pytest.raises(ValueError, match="aligned"):
            source.fetch(CubeKey(first + np.timedelta64(1, "ns")), domain)
        with pytest.raises(ValueError, match="runs"):
            source.fetch(CubeKey(first, first - np.timedelta64(1, "h")), domain)
        with pytest.raises(ValueError, match="at least"):
            source.fetch(CubeKey(first), domain, fields=[])
        result = SurfaceEnricher(source, cache_dir=mirror / "cache").enrich(
            [first, first - np.timedelta64(1, "ns"), last], 45, 2
        )
        np.testing.assert_array_equal(result.flags, [Flag.OK, Flag.OUT_OF_COVERAGE, Flag.OK])
        assert set(result.fields) == {"wind"}
        assert step > 0


@pytest.mark.parametrize(
    "kwargs", [{"fields": {}}, {"fields": {"x": "nope"}}, {"tile_deg": 0}, {"cache_mb": np.inf}]
)
def test_surface_configuration_validation(
    surface_mirror: tuple[Path, int], kwargs: dict[str, Any]
) -> None:
    mirror, _ = surface_mirror
    with pytest.raises(ValueError):
        OpenMeteoSurfaceSource(OM_MODEL, base_url=str(mirror), **kwargs)


def test_missing_dependency_is_explicit(tmp_path: Path) -> None:
    mirror = write_openmeteo_surface_mirror(tmp_path)
    (mirror / "data" / OM_MODEL / "snowfall_water_equivalent").rename(mirror / "snow")
    with pytest.raises(ValueError, match="snowfall_water_equivalent"):
        OpenMeteoSurfaceSource(OM_MODEL, base_url=str(mirror), fields={"rain": "rain"})
    with OpenMeteoSurfaceSource(
        OM_MODEL, base_url=str(mirror), fields={"precipitation": "precipitation"}
    ) as source:
        assert source.enrich(EPOCH, 45, 2).valid[0]


def test_surface_and_pressure_folder_names_do_not_collide(tmp_path: Path) -> None:
    write_openmeteo_mirror(tmp_path)
    write_openmeteo_surface_mirror(tmp_path)
    with (
        OpenMeteoSource(OM_MODEL, base_url=str(tmp_path)) as pressure,
        OpenMeteoSurfaceSource(OM_MODEL, base_url=str(tmp_path)) as surface,
    ):
        assert pressure.levels == [200, 300, 500]
        assert pressure.catalog.path("temperature", 300, 0).endswith(
            "temperature_300hPa/chunk_0.om"
        )
        assert surface.catalog.path("temperature_2m", 0, 0).endswith("temperature_2m/chunk_0.om")
        assert surface.enrich(EPOCH, 45, 2).valid[0]
