"""Shared synthetic atmosphere: fields linear in lat, lon, ln p and time.

Every interpolation step of the library (bilinear, ln-p, linear time) is exact
for such fields, so results can be compared with the analytic truth.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import numcodecs
import numpy as np
import pytest
from numpy.typing import NDArray

from supertraj_met.domain import Cube
from supertraj_met.vertical import model_levels

EPOCH = np.datetime64("2020-01-01T00", "h")


def truth(
    name: str,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    p: NDArray[np.float64],
    hours: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Analytic field; ``lon`` must be expressed in the cube frame."""
    coef = {"u": (1.0, 0.5, 20.0, 2.0), "v": (-0.3, 0.2, -5.0, -1.0), "t": (0.1, 0.05, 40.0, 0.5)}[
        name
    ]
    return coef[0] * lat + coef[1] * lon + coef[2] * np.log(p / 1e5) + coef[3] * hours


def surface_pressure(lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.float64]:
    return 101325.0 - 400.0 * (lat - 40.0) - 150.0 * lon


@pytest.fixture
def make_cube() -> Callable[..., Cube]:
    def build(
        hour: np.datetime64 = EPOCH,
        lat: NDArray[np.float64] | None = None,
        lon: NDArray[np.float64] | None = None,
        level_ids: NDArray[np.int64] | None = None,
    ) -> Cube:
        la = np.arange(50.0, 39.99, -0.25) if lat is None else lat
        lo = np.arange(-5.0, 10.01, 0.25) if lon is None else lon
        vertical = model_levels(np.arange(60, 138) if level_ids is None else level_ids)
        glat, glon = np.meshgrid(la, lo, indexing="ij")
        ps = surface_pressure(glat, glon)
        p = vertical.pressure(ps)  # (lat, lon, level)
        hours = float((np.datetime64(hour, "h") - EPOCH) / np.timedelta64(1, "h"))
        fields = {
            name: np.moveaxis(
                truth(name, glat[..., None], glon[..., None], p, np.float64(hours)), -1, 0
            ).astype(np.float64)
            for name in ("u", "v", "t")
        }
        return Cube(
            source="synthetic",
            hour=np.datetime64(hour, "h"),
            lat=la,
            lon=lo,
            vertical=vertical,
            fields=fields,  # type: ignore[arg-type]
            ps=ps.astype(np.float64),  # type: ignore[arg-type]
        )

    return build


# -- synthetic zarr-v2 stores laid out like ARCO-ERA5 ---------------------------------------
HOURS_1900 = int((EPOCH - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
GRID_LAT = 60.0 - 1.25 * np.arange(96)
GRID_LON = np.arange(256) * (360.0 / 256)
_BLOSC = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 16384}


def _write_array(
    root: Path,
    meta: dict[str, object],
    name: str,
    data: NDArray[np.generic],
    chunks: tuple[int, ...],
    dims: tuple[str, ...],
    attrs: dict[str, object] | None = None,
    skip: Callable[[tuple[int, ...]], bool] | None = None,
) -> None:
    codec = numcodecs.get_codec(_BLOSC)
    (root / name).mkdir(parents=True, exist_ok=True)
    meta[f"{name}/.zarray"] = {
        "chunks": list(chunks),
        "compressor": _BLOSC,
        "dtype": data.dtype.str,
        "fill_value": "NaN" if data.dtype.kind == "f" else None,
        "filters": None,
        "order": "C",
        "shape": list(data.shape),
        "zarr_format": 2,
    }
    meta[f"{name}/.zattrs"] = {"_ARRAY_DIMENSIONS": list(dims), **(attrs or {})}
    grid = [range(-(-s // c)) for s, c in zip(data.shape, chunks, strict=True)]
    for index in itertools.product(*grid):
        if skip is not None and skip(index):
            continue
        sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks, strict=True))
        block = data[sl]
        if block.shape != chunks:  # zarr v2 stores edge chunks padded to the full chunk shape
            padded = np.full(chunks, np.nan if data.dtype.kind == "f" else 0, dtype=data.dtype)
            padded[tuple(slice(0, s) for s in block.shape)] = block
            block = padded
        block = np.ascontiguousarray(block)
        (root / name / ".".join(map(str, index))).write_bytes(bytes(codec.encode(block)))


def write_stores(
    root: Path,
    n_hours: int = 3,
    missing_hours: tuple[int, ...] = (),
    names: tuple[str, str] = ("ml.zarr", "pl.zarr"),
) -> tuple[Path, Path]:
    """Model-level store (137 hybrid levels) and a pressure-level-like store with ``sp``."""
    ml_root, pl_root = root / names[0], root / names[1]
    glat, glon = np.meshgrid(GRID_LAT, GRID_LON, indexing="ij")
    ps = surface_pressure(glat, glon)
    vertical = model_levels()
    p = np.moveaxis(vertical.pressure(ps), -1, 0)  # (137, lat, lon)
    times = HOURS_1900 + np.arange(n_hours, dtype=np.int64)
    time_attrs = {"units": "hours since 1900-01-01 00:00:00", "calendar": "proleptic_gregorian"}
    skip = (lambda idx: idx[0] in missing_hours) if missing_hours else None

    meta: dict[str, object] = {".zgroup": {"zarr_format": 2}}
    _write_array(ml_root, meta, "time", times, (n_hours,), ("time",), time_attrs)
    _write_array(ml_root, meta, "hybrid", np.arange(1, 138, dtype="<f4"), (137,), ("hybrid",))
    _write_array(ml_root, meta, "latitude", GRID_LAT.astype("<f4"), (96,), ("latitude",))
    _write_array(ml_root, meta, "longitude", GRID_LON.astype("<f4"), (256,), ("longitude",))
    for name, array in (
        ("u", "u_component_of_wind"),
        ("v", "v_component_of_wind"),
        ("t", "temperature"),
    ):
        data = np.stack([truth(name, glat, glon, p, np.float64(h)) for h in range(n_hours)]).astype(
            "<f4"
        )
        _write_array(
            ml_root,
            meta,
            array,
            data,
            (1, 18, 96, 256),
            ("time", "hybrid", "latitude", "longitude"),
            skip=skip,
        )
    (ml_root / ".zmetadata").write_text(
        json.dumps({"metadata": meta, "zarr_consolidated_format": 1})
    )

    meta = {".zgroup": {"zarr_format": 2}}
    _write_array(pl_root, meta, "time", times, (n_hours,), ("time",), time_attrs)
    _write_array(pl_root, meta, "latitude", GRID_LAT.astype("<f4"), (96,), ("latitude",))
    _write_array(pl_root, meta, "longitude", GRID_LON.astype("<f4"), (256,), ("longitude",))
    sp = np.repeat(ps[None].astype("<f4"), n_hours, axis=0)
    _write_array(
        pl_root, meta, "surface_pressure", sp, (1, 96, 256), ("time", "latitude", "longitude")
    )
    (pl_root / ".zmetadata").write_text(
        json.dumps({"metadata": meta, "zarr_consolidated_format": 1})
    )
    return ml_root, pl_root


# -- synthetic ARPEGE-like GRIB2 mirror -------------------------------------------------------
ARPEGE_RUN = np.datetime64("2026-05-11T00", "h")
ARPEGE_LEVELS = (200, 250, 300, 400, 500)
_GRIB_PARAMS = {"u": (0, 2, 2), "v": (0, 2, 3), "t": (0, 0, 0)}


def arpege_grid() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """0.25 deg regional grid 72N..20N, 328E..42E (crosses Greenwich)."""
    return 72.0 - 0.25 * np.arange(209), 328.0 + 0.25 * np.arange(297)


def arpege_truth(
    name: str,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    p: NDArray[np.float64],
    hours: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Analytic field with ``lon`` expressed in the grid frame [328, 688)."""
    framed = 328.0 + np.mod(lon - 328.0, 360.0)
    return truth(name, lat, framed - 360.0, p, hours)


def _grib_keys(
    run: np.datetime64, param: tuple[int, int, int], hpa: int, step: int, nj: int, ni: int
) -> dict[str, float | int]:
    return {
        "discipline": param[0],
        "parameterCategory": param[1],
        "parameterNumber": param[2],
        "typeOfFirstFixedSurface": 100,
        "scaleFactorOfFirstFixedSurface": 0,
        "scaledValueOfFirstFixedSurface": hpa * 100,
        "Ni": ni,
        "Nj": nj,
        "latitudeOfFirstGridPointInDegrees": 72.0,
        "latitudeOfLastGridPointInDegrees": 20.0,
        "longitudeOfFirstGridPointInDegrees": 328.0,
        "longitudeOfLastGridPointInDegrees": 42.0,
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
        "jScansPositively": 0,
        "iScansNegatively": 0,
        "dataDate": int(str(run)[:10].replace("-", "")),
        "dataTime": int(str(run)[11:13]) * 100,
        "forecastTime": step,
        "bitsPerValue": 24,
    }


def write_arpege_mirror(
    root: Path, runs: tuple[np.datetime64, ...] = (ARPEGE_RUN,), steps: int = 13
) -> Path:
    """GRIB2 files laid out like object.data.gouv.fr/meteofrance-pnt/pnt (0.1-style, hourly)."""
    import eccodes

    lat, lon = arpege_grid()
    glat, glon = np.meshgrid(lat, lon, indexing="ij")
    for run in runs:
        stamp = str(run) + ":00:00Z"
        folder = root / stamp / "arpege" / "01" / "IP1"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"arpege__01__IP1__000H012H__{stamp}.grib2"
        lead0 = float((run - EPOCH) / np.timedelta64(1, "h"))
        with path.open("wb") as fh:
            for name, (disc, cat, num) in _GRIB_PARAMS.items():
                for hpa in ARPEGE_LEVELS:
                    for step in range(steps):
                        h = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
                        keys = _grib_keys(run, (disc, cat, num), hpa, step, len(lat), len(lon))
                        for key, value in keys.items():
                            eccodes.codes_set(h, key, value)
                        p = np.full_like(glat, hpa * 100.0)
                        values = arpege_truth(name, glat, glon, p, np.full_like(glat, lead0 + step))
                        eccodes.codes_set_values(h, values.ravel())
                        eccodes.codes_write(h, fh)
                        eccodes.codes_release(h)
    return root
