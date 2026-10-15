"""Persistent decoded reuse preserves sampling, run identity and memory bounds."""

import gc
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from baro.met import CubeKey, Flag, WeatherSession
from baro.met.domain import Cube
from synthetic import EPOCH, FakeSource


@pytest.fixture
def source(make_cube: Callable[..., Cube]) -> FakeSource:
    return FakeSource("session", 3.0, (0, 10), make_cube)


def _sample(session: WeatherSession, hour: int = 0):  # type: ignore[no-untyped-def]
    return session.sample([EPOCH + np.timedelta64(hour, "h")], [45.0], [0.0], pressure_pa=[3e4])


def test_repeated_points_reuse_and_clear(source: FakeSource) -> None:
    session = WeatherSession(source, vertical="linear")
    first, second = _sample(session), _sample(session)
    np.testing.assert_array_equal(first["u"], second["u"])
    assert len(source.fetched) == 1
    assert second.stats["decoded_cache_hits"] == 1
    assert session.cache_info()["bytes"] > 0
    session.clear()
    assert session.cache_info()["entries"] == 0
    _sample(session)
    assert len(source.fetched) == 2


def test_eviction_and_oversize_do_not_change_results(source: FakeSource) -> None:
    measurement = WeatherSession(source, vertical="linear")
    _sample(measurement)
    size = measurement.cache_info()["bytes"]
    measurement.clear()
    source.fetched.clear()
    session = WeatherSession(source, vertical="linear", max_cache_bytes=size, prefetch=1)
    for hour in (0, 1, 0):
        assert _sample(session, hour).valid.all()
        assert session.cache_info()["bytes"] <= size
    assert len(source.fetched) == 3
    assert session.cache_info()["evictions"] == 2
    tiny = WeatherSession(source, vertical="linear", max_cache_bytes=1)
    assert _sample(tiny).valid.all()
    assert tiny.cache_info()["bytes"] == 0


def test_enclosed_requests_reuse_but_expansion_fetches(source: FakeSource) -> None:
    session = WeatherSession(source, vertical="linear")
    session.sample([EPOCH, EPOCH], [44.0, 46.0], [-1.0, 1.0], pressure_pa=[25000, 35000])
    assert _sample(session).stats["decoded_cache_hits"] == 1
    session.sample([EPOCH], [45.0], [0.0], pressure_pa=[40000])
    assert len(source.fetched) == 2


def test_run_identity_is_part_of_cache_key(
    source: FakeSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected_run = EPOCH

    def keys(time: np.datetime64) -> tuple[CubeKey, CubeKey]:
        return CubeKey(time, selected_run), CubeKey(time + source.time_step, selected_run)

    monkeypatch.setattr(source, "cube_keys", keys)
    session = WeatherSession(source, vertical="linear")
    _sample(session)
    selected_run = EPOCH - np.timedelta64(1, "h")
    _sample(session)
    assert len(source.fetched) == 2
    assert source.fetched[0].run != source.fetched[1].run


def test_nanosecond_weights_do_not_use_float_epoch_hours(source: FakeSource) -> None:
    make = source.make

    def varying(**kwargs):  # type: ignore[no-untyped-def]
        cube = make(**kwargs)
        hour = int((cube.hour - EPOCH) / np.timedelta64(1, "h"))
        source.value = float(hour)
        return cube

    source.make = varying
    # FakeSource reads value after make; use serial fetching for that test fixture.
    session = WeatherSession(source, vertical="linear", prefetch=1)
    offset = 1_234_567_891
    result = session.sample(
        [EPOCH.astype("datetime64[ns]") + np.timedelta64(offset, "ns")],
        [45.0],
        [0.0],
        pressure_pa=[30000],
    )
    assert result["u"][0] == pytest.approx(offset / 3_600_000_000_000, rel=1e-14)


def test_unavailable_points_stay_flagged(source: FakeSource) -> None:
    session = WeatherSession(source, vertical="linear", on_uncovered="flag")
    assert _sample(session, 11).flags[0] == Flag.OUT_OF_COVERAGE
    assert session.cache_info()["entries"] == 0


def test_configuration_rejected(source: FakeSource) -> None:
    with pytest.raises(ValueError, match="max_cache_bytes"):
        WeatherSession(source, vertical="linear", max_cache_bytes=-1)
    with pytest.raises(TypeError, match="select_fields"):
        WeatherSession(source, vertical="linear", fields=["u"])


def test_cached_arrays_do_not_retain_large_backing_allocations(
    source: FakeSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    references: list[Any] = []
    backing_sizes = []
    fetch = source.fetch

    def large_backing(*args: Any, **kwargs: Any) -> Cube:
        cube = fetch(*args, **kwargs)
        shape = cube.fields["u"].shape
        large = np.full((shape[0] * 100, *shape[1:]), 3.0, dtype=np.float32)
        cube.fields["u"] = large[: shape[0]]
        references.append(weakref.ref(large))
        backing_sizes.append(large.nbytes)
        return cube

    monkeypatch.setattr(source, "fetch", large_backing)
    session = WeatherSession(source, vertical="linear")
    first = _sample(session)
    gc.collect()
    assert references[0]() is None
    assert 0 < session.cache_info()["bytes"] < backing_sizes[0]
    second = _sample(session)
    np.testing.assert_array_equal(first["u"], second["u"])
    assert second.stats["decoded_cache_hits"] == 1


def test_concurrent_calls_share_decoded_result(source: FakeSource) -> None:
    from concurrent.futures import ThreadPoolExecutor

    session = WeatherSession(source, vertical="linear")
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: _sample(session), range(8)))
    assert all(r.valid.all() for r in results)
    assert len(source.fetched) == 1


def test_unavailable_cube_is_retried(source: FakeSource) -> None:
    from baro.met import DataUnavailableError

    make = source.make

    def unavailable(**kwargs):  # type: ignore[no-untyped-def]
        raise DataUnavailableError("not published")

    source.make = unavailable
    session = WeatherSession(source, vertical="linear")
    assert _sample(session).flags[0] == Flag.MISSING_DATA
    assert session.cache_info()["entries"] == 0
    source.make = make
    assert _sample(session).valid.all()
