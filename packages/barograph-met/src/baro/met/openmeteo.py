"""Open-Meteo's public model archive on AWS (``s3://openmeteo``) as a source.

Open-Meteo keeps, for many NWP models (Météo-France ARPEGE, ECMWF IFS, DWD
ICON, NOAA GFS, ...), a continuous hourly series built from the most recent
run available at each hour (runs are stitched; lead times are not kept). Each
variable and level lives in its own ``.om`` files, one per block of
``chunk_time_length`` hours (e.g. 210 h). Inside a file the data are chunked as
``1 latitude row x 14 longitudes x all hours``: reading one hour costs as much
as reading the whole block, so blocks are read per geographic tile and kept in
an in-memory LRU, and every later hour and flight of the block is free.

Values are stored as scaled integers (lossy): e.g. temperature in 0.5 degC
steps. Licence: CC-BY 4.0 (credit Open-Meteo and the model's producer).
Reading needs ``omfiles`` (``pip install 'barograph-met[openmeteo]'``).

The same layout on a local disk (``<root>/data/<model>/...``) is read as a
mirror: pass its directory as ``base_url``.

Three parts: :class:`ArchiveCatalog` (what is archived), :class:`TileReader`
(reading and caching tiles) and :class:`OpenMeteoSource` (the source contract).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
from numpy.typing import NDArray

from baro.met.asyncfs import REVALIDATE_S, DiskPages, LocalAsyncFS, PagedAsyncFS
from baro.met.base import CubeKey, DataUnavailableError
from baro.met.domain import Cube, Domain, GridIndex
from baro.met.limits import run_async
from baro.met.store import USER_AGENT, TransferStats, list_s3
from baro.met.tiling import Tiling

if TYPE_CHECKING:
    from baro.met.rangecache import RangeCache
from baro.met.vertical import (
    MIN_LEVELS,
    STANDARD_SURFACE_PRESSURE,
    pressure_levels,
)

__all__ = [
    "OPENMETEO_BASE_URL",
    "OPENMETEO_VARIABLES",
    "STITCH_WINDOW",
    "ArchiveCatalog",
    "OpenMeteoSource",
    "TileReader",
    "gather_tiles",
]

# Open-Meteo keeps rewriting the recent hours of its archive with newer runs: only
# chunks that ended longer ago than this are treated as immutable and cached.
STITCH_WINDOW = np.timedelta64(7, "D")
_CHUNK_FILE = re.compile(r"chunk_(\d+)\.om$")

OPENMETEO_BASE_URL = "https://openmeteo.s3.amazonaws.com"

# output name -> (Open-Meteo variable prefix, offset added to convert to SI)
OPENMETEO_VARIABLES: dict[str, tuple[str, float]] = {
    "u": ("wind_u_component", 0.0),
    "v": ("wind_v_component", 0.0),
    "t": ("temperature", 273.15),  # degC -> K
    "r": ("relative_humidity", 0.0),
    "z": ("geopotential_height", 0.0),
}
_LEVEL = re.compile(r"^(?P<var>[a-z_]+)_(?P<hpa>\d+)hPa$")
_CHUNK = re.compile(r"^chunk_(\d+)\.om$")
_BBOX = re.compile(r"BBOX\[([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\]")
_HOUR_S = 3600
# archive files are (lat, lon, time) since early 2025, (lat * lon, time) before
_GRID_LAYOUT_NDIM = 3

type TileKey = tuple[str, int, int, tuple[int, int]]  # variable, hPa, chunk, tile


def _is_local(base: str) -> bool:
    return "://" not in base


# -- byte access for omfiles -----------------------------------------------------------------
# -- what is archived ------------------------------------------------------------------------
class ArchiveCatalog:
    """Time chunking, grid box, levels and first chunk of one archived model."""

    def __init__(
        self, base: str, model: str, variables: Sequence[str], stats: TransferStats
    ) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.stats = stats
        meta = json.loads(self._read_text(f"data/{model}/static/meta.json"))
        self.chunk_hours = int(meta["chunk_time_length"])
        if int(meta["temporal_resolution_seconds"]) != _HOUR_S:
            raise ValueError("only hourly archives are supported")
        self.end = np.datetime64(int(meta["data_end_time"]), "s").astype("datetime64[h]")
        bbox = _BBOX.search(meta["crs_wkt"])
        if bbox is None:
            raise ValueError("cannot read the grid bounding box from meta.json")
        lat_min, lon_min, lat_max, lon_max = (float(x) for x in bbox.groups())
        self.bbox = (lat_min, lon_min, lat_max, lon_max)
        self.levels_by_variable = self._levels(variables)
        self.first_chunk: int | None = None

    # listings, over HTTP (S3 XML) or on disk
    def _read_text(self, path: str) -> str:
        if _is_local(self.base):
            return (Path(self.base) / path).read_text()
        t0 = time.perf_counter()
        resp = httpx.get(
            f"{self.base}/{path}",
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        self.stats.add(len(resp.content), time.perf_counter() - t0)
        return resp.text

    def _children(self, prefix: str, dirs: bool) -> list[str]:
        """Names of the sub-folders (``dirs``) or files under ``prefix``."""
        if _is_local(self.base):
            folder = Path(self.base) / prefix
            if not folder.is_dir():
                return []
            return [p.name for p in folder.iterdir() if p.is_dir() == dirs]
        names = list_s3(self.base, prefix, dirs=dirs, stats=self.stats)
        return [name.rstrip("/").rsplit("/", 1)[-1] for name in names]

    def _levels(self, variables: Sequence[str]) -> dict[str, set[int]]:
        found: dict[str, set[int]] = {v: set() for v in variables}
        for name in self._children(f"data/{self.model}/", dirs=True):
            m = _LEVEL.match(name)
            if m and m["var"] in found:
                found[m["var"]].add(int(m["hpa"]))
        return found

    def common_levels(self, allowed: Sequence[int] | None) -> list[int]:
        """Levels archived for every variable (optionally restricted to ``allowed``)."""
        sets = list(self.levels_by_variable.values())
        levels = sorted(set.intersection(*sets)) if sets else []
        if allowed is not None:
            levels = [x for x in levels if x in set(allowed)]
        if len(levels) < MIN_LEVELS:
            raise ValueError(f"{self.model}: fewer than 2 common isobaric levels")
        return levels

    def locate_first_chunk(self, variable: str, hpa: int) -> int:
        """Index of the oldest archived time chunk."""
        if self.first_chunk is None:
            names = self._children(f"data/{self.model}/{variable}_{hpa}hPa/", dirs=False)
            chunks = [int(m[1]) for n in names if (m := _CHUNK.match(n))]
            if not chunks:
                raise ValueError(f"{self.model}: no archived chunks")
            self.first_chunk = min(chunks)
        return self.first_chunk

    def path(self, variable: str, hpa: int, chunk: int) -> str:
        """Relative path of one archive file."""
        return f"data/{self.model}/{variable}_{hpa}hPa/chunk_{chunk}.om"


# -- reading tiles ---------------------------------------------------------------------------
class TileReader:
    """Reads ``(lat, lon, all hours)`` tiles of archive files and keeps them in an LRU."""

    def __init__(
        self, catalog: ArchiveCatalog, stats: TransferStats, tile_deg: float, cache_mb: float
    ) -> None:
        self.catalog = catalog
        self.stats = stats
        self.tile_deg = tile_deg
        self.max_connections = 64
        self.shape: tuple[int, int] | None = None
        self._cache: OrderedDict[TileKey, NDArray[np.float32]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(cache_mb * 1e6)
        self._lock = threading.Lock()
        self._fs: PagedAsyncFS | LocalAsyncFS | None = None
        self.disk: DiskPages | None = None
        # files no later run can rewrite; tiles of the others are re-read after a while
        self.final: Callable[[str], bool] = lambda _path: False
        self._cached_at: dict[TileKey, float] = {}

    def _filesystem(self) -> PagedAsyncFS | LocalAsyncFS:
        if self._fs is None:  # only touched from the shared I/O loop thread
            base = self.catalog.base
            self._fs = (
                LocalAsyncFS(base, self.stats)
                if _is_local(base)
                else PagedAsyncFS(base, self.stats, self.max_connections)
            )
        if isinstance(self._fs, PagedAsyncFS):
            self._fs.disk = self.disk
            self._fs.immutable = self.final
        return self._fs

    def close(self) -> None:
        """Release the HTTP pool (the shared I/O loop keeps running)."""
        fs, self._fs = self._fs, None
        if fs is not None:
            run_async(fs.aclose())

    def ensure_shape(self, variable: str, hpa: int) -> tuple[int, int]:
        """Grid shape, read once from the newest file (always the (lat, lon, time) layout)."""
        if self.shape is None:
            last_chunk = int(self.catalog.end.astype(np.int64)) // self.catalog.chunk_hours
            self.shape = run_async(self._read_shape(self.catalog.path(variable, hpa, last_chunk)))
        return self.shape

    def grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Latitudes descending and longitudes ascending of the archive grid."""
        if self.shape is None:
            raise RuntimeError("grid shape unknown before the first read")
        nlat, nlon = self.shape
        lat_min, lon_min, lat_max, lon_max = self.catalog.bbox
        lat = lat_min + np.arange(nlat) * (lat_max - lat_min) / (nlat - 1)
        lon = lon_min + np.arange(nlon) * (lon_max - lon_min) / (nlon - 1)
        return lat[::-1].copy(), lon

    def tile_points(self) -> int:
        """Tile edge, in grid points."""
        if self.shape is None:
            raise RuntimeError("grid shape unknown before the first read")
        lat_min, _, lat_max, _ = self.catalog.bbox
        step = (lat_max - lat_min) / (self.shape[0] - 1)
        return max(round(self.tile_deg / step), 1)

    def blocks(
        self,
        wanted: list[tuple[str, int]],
        chunk: int,
        rows: NDArray[np.int64],
        cols: NDArray[np.int64],
    ) -> dict[tuple[str, int], dict[tuple[int, int], NDArray[np.float32]]]:
        """Tiles of every (variable, level) covering ``rows x cols`` for one time chunk."""
        tp = self.tile_points()
        tiles = [(int(tr), int(tc)) for tr in np.unique(rows // tp) for tc in np.unique(cols // tp)]
        out: dict[tuple[str, int], dict[tuple[int, int], NDArray[np.float32]]] = {
            w: {} for w in wanted
        }
        missing: list[TileKey] = []
        with self._lock:
            for variable, hpa in wanted:
                for tile in tiles:
                    key = (variable, hpa, chunk, tile)
                    if key in self._cache and self._fresh(key):
                        self._cache.move_to_end(key)
                        out[(variable, hpa)][tile] = self._cache[key]
                    else:
                        missing.append(key)
        if missing:
            read = run_async(self._read_tiles(missing, tp))
            with self._lock:
                for key, arr in zip(missing, read, strict=True):
                    out[(key[0], key[1])][key[3]] = arr
                    self._remember(key, arr)
        return out

    def _fresh(self, key: TileKey) -> bool:
        """A cached tile is usable if its file is final or it was read recently."""
        if self.final(self.catalog.path(key[0], key[1], key[2])):
            return True
        return time.monotonic() - self._cached_at.get(key, -np.inf) < REVALIDATE_S

    def _remember(self, key: TileKey, arr: NDArray[np.float32]) -> None:
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= old.nbytes
        self._cache[key] = arr
        self._cached_at[key] = time.monotonic()
        self._cache_bytes += arr.nbytes
        while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
            evicted, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
            self._cached_at.pop(evicted, None)

    async def _read_shape(self, path: str) -> tuple[int, int]:
        omfiles = _omfiles()
        try:
            reader = await omfiles.OmFileReaderAsync.from_fsspec(self._filesystem(), path)
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"{path} not archived") from exc
        shape = tuple(reader.shape)
        reader.close()
        if len(shape) != _GRID_LAYOUT_NDIM:
            raise ValueError(f"{self.catalog.model}: unexpected layout {shape} in the newest file")
        return int(shape[0]), int(shape[1])

    async def _read_tiles(self, keys: list[TileKey], tp: int) -> list[NDArray[np.float32]]:
        """Read every (variable, level, chunk, tile) block concurrently."""
        omfiles = _omfiles()
        fs = self._filesystem()
        paths = sorted({self.catalog.path(k[0], k[1], k[2]) for k in keys})
        try:
            opened = await asyncio.gather(
                *(omfiles.OmFileReaderAsync.from_fsspec(fs, p) for p in paths)
            )
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"{self.catalog.model}: {exc} not archived") from exc
        readers = dict(zip(paths, opened, strict=True))
        try:
            return list(
                await asyncio.gather(
                    *(
                        self._read_tile(readers[self.catalog.path(k[0], k[1], k[2])], k, tp)
                        for k in keys
                    )
                )
            )
        finally:
            for r in readers.values():
                try:
                    r.close()
                except RuntimeError:  # reader still busy after a failed read
                    pass

    async def _read_tile(self, reader: Any, key: TileKey, tp: int) -> NDArray[np.float32]:
        if self.shape is None:
            raise RuntimeError("grid shape unknown before the first read")
        nlat, nlon = self.shape
        r0, c0 = key[3][0] * tp, key[3][1] * tp
        r1, c1 = min(r0 + tp, nlat), min(c0 + tp, nlon)
        if len(reader.shape) == _GRID_LAYOUT_NDIM:  # (lat, lon, time), from 2025
            arr = await reader.read_array((slice(r0, r1), slice(c0, c1), slice(None)))
            return np.asarray(arr, dtype=np.float32)
        # older files: (lat * lon, time), row-major; read whole rows then crop
        if reader.shape[0] != nlat * nlon:
            raise ValueError(f"{self.catalog.model}: {reader.shape} does not match {nlat}x{nlon}")
        flat = await reader.read_array((slice(r0 * nlon, r1 * nlon), slice(None)))
        rows = np.asarray(flat, dtype=np.float32).reshape(r1 - r0, nlon, -1)
        return np.ascontiguousarray(rows[:, c0:c1])


def gather_tiles(
    tiles: dict[tuple[int, int], NDArray[np.float32]],
    rows: NDArray[np.int64],
    cols: NDArray[np.int64],
    tp: int,
    t_in: int,
) -> NDArray[np.float32]:
    """Values at archive ``rows x cols`` (output order) for one hour, from cached tiles."""
    out = np.empty((len(rows), len(cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        tr, lr = divmod(int(r), tp)
        for tc in np.unique(cols // tp):
            sel = np.flatnonzero(cols // tp == tc)
            out[i, sel] = tiles[(tr, int(tc))][lr, cols[sel] - tc * tp, t_in]
    return out


# -- the source ------------------------------------------------------------------------------
class OpenMeteoSource:
    """Isobaric fields of one Open-Meteo archived model.

    Parameters
    ----------
    model:
        Open-Meteo model folder, e.g. ``"meteofrance_arpege_europe"`` (0.1 deg,
        from 2023-12-01) or ``"meteofrance_arpege_world025"`` (0.25 deg, from
        2023-12-22).
    fields:
        Output name -> key of :data:`OPENMETEO_VARIABLES` (default u, v, t).
    levels:
        Isobaric levels (hPa) allowed; default: all archived levels.
    base_url:
        The public bucket, or a local mirror directory with the same layout.
    tile_deg:
        Size of the geographic tiles read and cached (degrees).
    cache_mb:
        Memory budget of the tile LRU.
    """

    time_step = np.timedelta64(1, "h")

    def __init__(  # noqa: PLR0913 - keyword-only configuration of a public source
        self,
        model: str = "meteofrance_arpege_europe",
        *,
        fields: dict[str, str] | None = None,
        levels: Sequence[int] | None = None,
        base_url: str = OPENMETEO_BASE_URL,
        tile_deg: float = 5.0,
        cache_mb: float = 1024.0,
    ) -> None:
        self.model = model
        self.fields = dict(fields or {"u": "u", "v": "v", "t": "t"})
        unknown = set(self.fields.values()) - set(OPENMETEO_VARIABLES)
        if unknown:
            raise ValueError(f"unknown variables {sorted(unknown)}")
        self.source_id = f"openmeteo-{model}"
        self._stats = TransferStats()
        variables = [OPENMETEO_VARIABLES[name][0] for name in self.fields.values()]
        self.catalog = ArchiveCatalog(base_url, model, variables, self._stats)
        self.levels = self.catalog.common_levels(levels)
        self.reader = TileReader(self.catalog, self._stats, tile_deg, cache_mb)
        self.reader.final = self._final_path
        self._probe = (variables[0], self.levels[-1])  # a file that always exists

    def __enter__(self) -> OpenMeteoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release network resources."""
        self.reader.close()

    @property
    def stats(self) -> list[TransferStats]:
        """Transfer counters of this source."""
        return [self._stats]

    @property
    def chunk_hours(self) -> int:
        """Hours per archive file."""
        return self.catalog.chunk_hours

    def tiling(self, key: CubeKey) -> Tiling | None:
        """The archive's square tiles of ``tile_deg`` degrees."""
        self.reader.ensure_shape(*self._probe)
        lat, lon = self.reader.grid()
        tp = self.reader.tile_points()
        dlat, dlon = abs(float(lat[0] - lat[1])), float(lon[1] - lon[0])
        return Tiling(
            lat0=float(lat[-1]) - dlat / 2,
            lat_step=tp * dlat,
            lon0=float(lon[0]) - dlon / 2,
            lon_step=tp * dlon,
        )

    def use_cache(self, cache: RangeCache) -> None:
        """Keep on disk the pages of archive chunks older than the stitching window."""
        self.reader.disk = DiskPages.at(cache, self.catalog.base, self._final_path)

    def _final_path(self, path: str) -> bool:
        """True for a chunk file that no later run can rewrite any more."""
        m = _CHUNK_FILE.search(path)
        if m is None:
            return False
        end = np.datetime64((int(m.group(1)) + 1) * self.catalog.chunk_hours, "h")
        return bool(end < np.datetime64("now", "h") - STITCH_WINDOW)

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        """First and last archived hours (the first chunk may start partly empty)."""
        first = self.catalog.locate_first_chunk(*self._probe)
        return np.datetime64(first * self.catalog.chunk_hours, "h"), self.catalog.end

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        """Hours bracketing ``t0`` (no run: the archive keeps the latest run only)."""
        h = np.datetime64(t0, "h")
        return CubeKey(h), CubeKey(h + self.time_step)

    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        """One hour over ``domain`` with the levels bracketing ``p_range`` (Pa)."""
        start, end = self.coverage()
        if not start <= key.valid <= end:
            raise DataUnavailableError(f"{self.source_id}: {key.valid} outside {start}..{end}")
        hours = int(key.valid.astype("datetime64[h]").astype(np.int64))
        chunk, t_in = divmod(hours, self.catalog.chunk_hours)
        vertical = pressure_levels(self.levels)
        positions = vertical.select_bracketing(
            p_range[0],
            p_range[1],
            STANDARD_SURFACE_PRESSURE,
            STANDARD_SURFACE_PRESSURE,
            level_margin,
        )
        nlat, _ = self.reader.ensure_shape(*self._probe)
        lat, lon = self.reader.grid()
        grid = GridIndex.locate(domain, lat, lon)
        rows = nlat - 1 - np.arange(grid.rows.start, grid.rows.stop)  # archive rows, lat ascending
        hpas = [int(h) for h in vertical.ids[positions]]
        wanted = [(OPENMETEO_VARIABLES[n][0], h) for n in self.fields.values() for h in hpas]
        blocks = self.reader.blocks(wanted, chunk, rows, grid.cols)
        tp = self.reader.tile_points()
        fields: dict[str, NDArray[np.float32]] = {}
        for out_name, name in self.fields.items():
            variable, offset = OPENMETEO_VARIABLES[name]
            layers = [
                gather_tiles(blocks[(variable, h)], rows, grid.cols, tp, t_in) + offset
                for h in hpas
            ]
            fields[out_name] = np.stack(layers).astype(np.float32)
        return Cube(
            source=self.source_id,
            hour=key.valid,
            lat=grid.lat,
            lon=grid.lon,
            vertical=vertical.subset(positions),
            fields=fields,
            p_range=(float(p_range[0]), float(p_range[1])),
            level_margin=level_margin,
            request_domain=domain,
        )


def _omfiles() -> Any:
    try:
        import omfiles
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading Open-Meteo archives needs: pip install 'barograph-met[openmeteo]'"
        ) from exc
    return omfiles
