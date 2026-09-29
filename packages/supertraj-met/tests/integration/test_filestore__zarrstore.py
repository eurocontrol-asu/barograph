from __future__ import annotations

from pathlib import Path

import numcodecs
import numpy as np
import pytest
from synthetic import GRID_LAT, write_stores

from supertraj_met.store import FileStore, ZarrStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_stores(tmp_path_factory.mktemp("zarr"), n_hours=2, missing_hours=(1,))


def _full(root: Path, key: str) -> np.ndarray:
    return np.frombuffer(
        bytes(numcodecs.Blosc().decode((root / key).read_bytes())), dtype="<f4"
    ).reshape(18, 96, 256)


def test_band_reads_equal_the_decoded_chunk_and_fetch_less(stores: tuple[Path, Path]) -> None:
    ml, _ = stores
    fs = FileStore(ml)
    z = ZarrStore(fs)
    before = fs.stats.snapshot()["bytes"]
    lead = [(0, 40), (0, 41), (0, 75)]
    out = z.read_rows("temperature", lead, slice(10, 30))
    fetched = fs.stats.snapshot()["bytes"] - before
    c2, c4 = _full(ml, "temperature/0.2.0.0"), _full(ml, "temperature/0.4.0.0")
    np.testing.assert_array_equal(out[0], c2[40 - 36, 10:30])
    np.testing.assert_array_equal(out[1], c2[41 - 36, 10:30])
    np.testing.assert_array_equal(out[2], c4[75 - 72, 10:30])
    chunk_bytes = (ml / "temperature/0.2.0.0").stat().st_size + (
        ml / "temperature/0.4.0.0"
    ).stat().st_size
    assert fetched < 0.5 * chunk_bytes


def test_missing_chunks_read_as_nan(stores: tuple[Path, Path]) -> None:
    ml, _ = stores
    out = ZarrStore(FileStore(ml)).read_rows("temperature", [(1, 50)], slice(0, 4))
    assert np.isnan(out).all()


def test_small_arrays_without_level_dimension(stores: tuple[Path, Path]) -> None:
    _, pl = stores
    z = ZarrStore(FileStore(pl))
    out = z.read_rows("surface_pressure", [(0,)], slice(0, 96))
    assert out.shape == (1, 96, 256) and np.isfinite(out).all()
    np.testing.assert_allclose(z.read_1d("latitude"), GRID_LAT)
