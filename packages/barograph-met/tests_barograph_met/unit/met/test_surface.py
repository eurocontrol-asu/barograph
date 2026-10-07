"""Surface interpolation, period support and provenance without I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from baro.met import CubeKey, Domain, Flag, SurfaceCube, SurfaceEnricher, SurfaceVariable
from baro.met.base import DataUnavailableError
from baro.met.rangecache import RangeCache
from baro.met.store import TransferStats
from baro.met.tiling import Tiling

EPOCH = np.datetime64("2026-04-15T00", "ns")
AXIS = np.array([0.0, 1.0])


class FakeSurface:
    def __init__(self, fields: tuple[str, ...] = ("u10", "v10", "precipitation", "gusts")) -> None:
        metadata = {
            "u10": SurfaceVariable("m/s", 10),
            "v10": SurfaceVariable("m/s", 10),
            "precipitation": SurfaceVariable("mm", 0, "sum"),
            "gusts": SurfaceVariable("m/s", 10, "max"),
        }
        self.variables = {n: metadata[n] for n in fields}
        self.time_step = np.timedelta64(1, "h")
        self.stats = [TransferStats()]
        self.fetched: list[CubeKey] = []
        self.missing: set[int] = set()
        self.nan: set[tuple[int, str]] = set()
        self.run: np.datetime64 | None = None
        self.cross_run = False
        self.unavailable_keys = False

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        return EPOCH, EPOCH + np.timedelta64(3, "h")

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        if self.unavailable_keys:
            raise DataUnavailableError("missing run")
        right_run = (
            self.run + self.time_step if self.cross_run and self.run is not None else self.run
        )
        return CubeKey(t0, self.run), CubeKey(t0 + self.time_step, right_run)

    def tiling(self, key: CubeKey) -> Tiling | None:
        return None

    def use_cache(self, cache: RangeCache) -> None:
        self.cache = cache

    def fetch(
        self, key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None
    ) -> SurfaceCube:
        h = int((key.valid - EPOCH) / self.time_step)
        self.fetched.append(key)
        if h in self.missing:
            raise DataUnavailableError("missing field")
        self.stats[0].add(8, 0.0)
        selected = self.variables if fields is None else {n: self.variables[n] for n in fields}
        values: dict[str, np.ndarray] = {}
        for name in selected:
            value = {"u10": 2 * h, "v10": 0, "precipitation": 10 + h, "gusts": 20 + h}[name]
            values[name] = np.full(
                (2, 2), np.nan if (h, name) in self.nan else value, dtype=np.float32
            )
        return SurfaceCube(key, AXIS, AXIS, values, selected)


def test_instant_interpolation_and_native_interval_values() -> None:
    source = FakeSurface()
    time = np.array([EPOCH, EPOCH + np.timedelta64(30, "m"), EPOCH + np.timedelta64(1, "h")])
    result = SurfaceEnricher(source).enrich(time, 0.5, 0.5)
    np.testing.assert_array_equal(result.flags, 0)
    np.testing.assert_allclose(result["u10"], [0, 1, 2])
    np.testing.assert_allclose(result["precipitation"], [10, 11, 11])
    np.testing.assert_allclose(result["gusts"], [20, 21, 21])
    np.testing.assert_allclose(result["wind_speed"], [0, 1, 2])
    assert np.isnan(result["wind_direction"][0])
    np.testing.assert_allclose(result["wind_direction"][1:], 270)
    np.testing.assert_array_equal(
        result.interval_end["precipitation"],
        [EPOCH, EPOCH + source.time_step, EPOCH + source.time_step],
    )
    np.testing.assert_array_equal(
        result.interval_start["precipitation"],
        result.interval_end["precipitation"] - source.time_step,
    )
    np.testing.assert_array_equal(result.interval_end["u10"], time)
    assert np.isnat(result.run).all() and np.isnat(result.lead).all()
    assert result.valid.all() and result.stats["bytes"] > 0


def test_period_only_query_needs_only_enclosing_endpoint() -> None:
    source = FakeSurface(("precipitation", "gusts"))
    result = SurfaceEnricher(source).enrich([EPOCH + np.timedelta64(1, "ns")], 0.5, 0.5)
    assert source.fetched == [CubeKey(EPOCH + source.time_step)]
    assert result["precipitation"][0] == 11
    assert result.valid[0]


def test_final_endpoint_and_nan_unused_left_period() -> None:
    source = FakeSurface()
    source.nan.add((0, "precipitation"))
    source.missing.add(4)
    result = SurfaceEnricher(source).enrich(
        [EPOCH + np.timedelta64(30, "m"), EPOCH + np.timedelta64(3, "h")], 0.5, 0.5
    )
    assert result.valid.all()
    assert result["u10"][1] == 6
    assert not any(k.valid == EPOCH + np.timedelta64(4, "h") for k in source.fetched)


def test_missing_right_does_not_invalidate_exact_left() -> None:
    source = FakeSurface()
    source.missing.add(1)
    result = SurfaceEnricher(source).enrich([EPOCH, EPOCH + np.timedelta64(30, "m")], 0.5, 0.5)
    np.testing.assert_array_equal(result.flags, [Flag.OK, Flag.MISSING_DATA])
    assert result["u10"][0] == 0 and np.isnan(result["u10"][1])


def test_fixed_run_provenance_and_cross_run_rejection() -> None:
    source = FakeSurface()
    source.run = EPOCH - np.timedelta64(6, "h")
    result = SurfaceEnricher(source).enrich(EPOCH + np.timedelta64(30, "m"), 0.5, 0.5)
    assert result.run[0] == source.run
    assert result.lead[0] == np.timedelta64(390, "m")
    source.cross_run = True
    with pytest.raises(ValueError, match="same run"):
        SurfaceEnricher(source).enrich(EPOCH, 0.5, 0.5)


def test_invalid_outside_coverage_and_missing_keys() -> None:
    source = FakeSurface()
    time = [
        np.datetime64("NaT", "ns"),
        EPOCH - np.timedelta64(1, "ns"),
        EPOCH + np.timedelta64(3, "h") + np.timedelta64(1, "ns"),
        EPOCH,
        EPOCH,
    ]
    result = SurfaceEnricher(source).enrich(
        time, [0.5, 0.5, 0.5, 100, 0.5], [0.5, 0.5, 0.5, 0.5, 2]
    )
    np.testing.assert_array_equal(
        result.flags,
        [
            Flag.INVALID_INPUT,
            Flag.OUT_OF_COVERAGE,
            Flag.OUT_OF_COVERAGE,
            Flag.INVALID_INPUT,
            Flag.OUTSIDE_DOMAIN,
        ],
    )
    source.unavailable_keys = True
    assert SurfaceEnricher(source).enrich(EPOCH, 0.5, 0.5).flags[0] == Flag.MISSING_DATA


def test_empty_inputs_shape_validation_and_cache(tmp_path: Path) -> None:
    source = FakeSurface()
    enricher = SurfaceEnricher(source, cache_dir=tmp_path, wind_components=None)
    assert hasattr(source, "cache")
    result = enricher.enrich([], [], [])
    assert not result.fields["u10"].size and result.stats == {"bytes": 0}
    with pytest.raises(ValueError, match="same shape"):
        enricher.enrich([EPOCH, EPOCH], [0, 1, 2], [0])


def test_bilinear_surface_grid_and_unused_nan_corner() -> None:
    cube = SurfaceCube(
        CubeKey(EPOCH),
        AXIS,
        AXIS,
        {"x": np.array([[1, np.nan], [3, 5]], dtype=np.float32)},
        {"x": SurfaceVariable("K", 2)},
    )
    values, flags = cube.sample([0, 1, 1, 0.5, np.nan, 0.5], [0, 0, 1, 0.5, 0, np.inf])
    np.testing.assert_array_equal(
        flags, [0, 0, 0, Flag.MISSING_DATA, Flag.INVALID_INPUT, Flag.INVALID_INPUT]
    )
    np.testing.assert_array_equal(values["x"][:3], [1, 3, 5])
    cube = replace(cube, fields={"x": np.array([[1, 2], [3, 4]], dtype=np.float32)})
    assert cube.sample([0.5], [0.5])[0]["x"][0] == 2.5


@pytest.mark.parametrize(
    "change",
    [
        {"lat": np.array([0.0])},
        {"lat": np.array([1.0, 0.0])},
        {"lon": np.array([0.0, 1.0, 3.0])},
        {"fields": {"x": np.zeros((3, 2), dtype=np.float32)}},
        {"variables": {}},
    ],
)
def test_invalid_surface_cube_contract(change: dict[str, Any]) -> None:
    cube = SurfaceCube(
        CubeKey(EPOCH),
        AXIS,
        AXIS,
        {"x": np.ones((2, 2), dtype=np.float32)},
        {"x": SurfaceVariable("K", 2)},
    )
    with pytest.raises(ValueError):
        replace(cube, **change)


def test_wind_derived_after_vector_interpolation_and_alias_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSurface(("u10", "v10"))
    original = source.fetch

    def fetch(key: CubeKey, domain: Domain, *, fields: Sequence[str] | None = None) -> SurfaceCube:
        cube = original(key, domain, fields=fields)
        h = int((key.valid - EPOCH) / source.time_step)
        angle = np.radians(359 if h == 0 else 1)
        return replace(
            cube,
            fields={
                "u10": np.full((2, 2), -np.sin(angle), dtype=np.float32),
                "v10": np.full((2, 2), -np.cos(angle), dtype=np.float32),
            },
        )

    monkeypatch.setattr(source, "fetch", fetch)
    result = SurfaceEnricher(source).enrich(EPOCH + np.timedelta64(30, "m"), 0.5, 0.5)
    assert result["wind_direction"][0] == pytest.approx(0)
    assert result["wind_speed"][0] == pytest.approx(np.cos(np.radians(1)))
    result = SurfaceEnricher(source, wind_components=None).enrich(EPOCH, 0.5, 0.5)
    assert "wind_speed" not in result.fields
