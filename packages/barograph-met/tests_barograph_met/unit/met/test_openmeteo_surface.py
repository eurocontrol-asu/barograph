"""Public surface source behavior with an in-memory archive reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from baro.met import CubeKey, Domain, OpenMeteoSurfaceSource

EPOCH = np.datetime64("2026-04-15T00", "s")
DOMAIN = Domain(0.0, 1.0, 0.0, 1.0)


class MemoryCatalog:
    """Known native timestamps and folders, without a filesystem or network."""

    def __init__(self, *args: Any) -> None:
        self.time_step = np.timedelta64(1, "h")
        self.step_seconds = 3600
        self.chunk_steps = 24
        self.chunk_seconds = 24 * 3600
        self.end = EPOCH + np.timedelta64(23, "h")
        self.available_variables = frozenset(
            {
                "precipitation",
                "snowfall_water_equivalent",
                "temperature_2m",
                "wind_u_component_10m",
                "wind_v_component_10m",
                "wind_gusts_10m",
            }
        )

    def time_index(self, valid: np.datetime64) -> int:
        return int(valid.astype("datetime64[s]").astype(np.int64)) // self.step_seconds

    def locate_first_chunk(self, variable: str, hpa: int) -> int:
        return self.time_index(EPOCH) // self.chunk_steps


class MemoryReader:
    """Serve native grid values while recording selected archive dependencies."""

    def __init__(self, *args: Any) -> None:
        self.wanted: list[tuple[str, int]] = []
        self.native = {
            "precipitation": np.array([[4.0, 1.0], [0.0, np.nan]], dtype=np.float32),
            "snowfall_water_equivalent": np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32),
            "temperature_2m": np.array([[-10.0, 0.0], [10.0, 20.0]], dtype=np.float32),
            "wind_u_component_10m": np.ones((2, 2), dtype=np.float32),
        }

    def close(self) -> None:
        pass

    def ensure_shape(self, *args: Any) -> tuple[int, int]:
        return (2, 2)

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])

    def tile_points(self) -> int:
        return 2

    def blocks(
        self, wanted: list[tuple[str, int]], *args: Any
    ) -> dict[tuple[str, int], dict[tuple[int, int], np.ndarray]]:
        self.wanted = wanted
        return {
            key: {(0, 0): np.repeat(self.native[key[0]][:, :, None], 24, axis=2)} for key in wanted
        }


@pytest.fixture
def memory_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baro.met.openmeteo_surface.ArchiveCatalog", MemoryCatalog)
    monkeypatch.setattr("baro.met.openmeteo_surface.TileReader", MemoryReader)


def test_rain_preserves_native_period_total_and_temperature_converts_to_kelvin(
    memory_archive: None,
) -> None:
    with OpenMeteoSurfaceSource(
        fields={"liquid": "rain", "total": "precipitation", "temperature": "t2m"}
    ) as source:
        reader = cast(MemoryReader, source.reader)
        before = {name: value.copy() for name, value in reader.native.items()}
        cube = source.fetch(CubeKey(EPOCH), DOMAIN)
        np.testing.assert_array_equal(cube.fields["liquid"], [[2.0, 0.0], [0.0, np.nan]])
        np.testing.assert_array_equal(cube.fields["total"], before["precipitation"])
        np.testing.assert_allclose(cube.fields["temperature"], [[263.15, 273.15], [283.15, 293.15]])
        for name in before:
            np.testing.assert_array_equal(reader.native[name], before[name])


def test_shared_precipitation_dependency_is_read_once_and_wind_is_not_requested(
    memory_archive: None,
) -> None:
    with OpenMeteoSurfaceSource(
        fields={"liquid": "rain", "total": "precipitation", "eastward": "u10"}
    ) as source:
        reader = cast(MemoryReader, source.reader)
        cube = source.fetch(CubeKey(EPOCH), DOMAIN, fields=["liquid", "total"])
        assert reader.wanted == [("precipitation", 0), ("snowfall_water_equivalent", 0)]
        assert set(cube.fields) == {"liquid", "total"}
        source.fetch(CubeKey(EPOCH), DOMAIN, fields=["eastward"])
        assert reader.wanted == [("wind_u_component_10m", 0)]


def test_selection_is_copied_and_defaults_include_requested_weather(memory_archive: None) -> None:
    fields = {"temperature": "t2m"}
    with OpenMeteoSurfaceSource(fields=fields) as source:
        fields["temperature"] = "u10"
        assert set(source.fetch(CubeKey(EPOCH), DOMAIN).fields) == {"temperature"}
        assert source.variables["temperature"].unit == "K"
    with OpenMeteoSurfaceSource() as source:
        assert set(source.variables) == {"u10", "v10", "t2m", "precipitation", "rain", "gusts"}


@pytest.mark.parametrize(
    "fields,pattern",
    [({}, "at least one"), ({"bad": "unknown_height"}, "unknown surface variables")],
)
def test_invalid_selection_is_rejected_before_data_access(
    fields: dict[str, str], pattern: str
) -> None:
    # A path that does not exist: invalid selections must fail before opening its catalog.
    with pytest.raises(ValueError, match=pattern):
        OpenMeteoSurfaceSource(fields=fields, base_url=str(Path("missing-surface-catalog")))


@pytest.mark.parametrize("tile,cache", [(0, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_spatial_or_cache_budget_is_rejected(tile: float, cache: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        OpenMeteoSurfaceSource(tile_deg=tile, cache_mb=cache, base_url="missing-surface-catalog")
