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
Reading needs ``omfiles`` (``pip install 'supertraj-met[openmeteo]'``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from supertraj_met.base import CubeKey, DataUnavailableError
from supertraj_met.domain import Cube, Domain, GridIndex
from supertraj_met.limits import async_slot, run_async
from supertraj_met.store import TransferStats
from supertraj_met.vertical import STANDARD_SURFACE_PRESSURE, VerticalCoordinate, pressure_levels

__all__ = ["OPENMETEO_BASE_URL", "OPENMETEO_VARIABLES", "OpenMeteoSource"]

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
_BBOX = re.compile(r"BBOX\[([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\]")
_HOUR_S = 3600
_log = logging.getLogger("supertraj_met.http")


class _AsyncHttp:
    """Async fsspec-like object for ``omfiles`` (``_cat_file``/``_size``), with a page cache.

    ``omfiles`` issues one range request per internal data chunk, and archive
    chunks are tiny (1 latitude row x 14 longitudes, a few hundred bytes).
    Requests are therefore served from aligned pages of ``page_size`` bytes
    (a page holds several whole latitude rows); concurrent requests for the
    same page share one download. This trades a few extra bytes for ~50x fewer
    round trips, which dominate the cost.
    """

    def __init__(
        self,
        base_url: str,
        stats: TransferStats,
        max_connections: int,
        page_size: int = 512 << 10,
        cache_mb: float = 256.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.stats = stats
        self.page_size = page_size
        self.client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=max_connections),
        )
        self._pages: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._inflight: dict[tuple[str, int], asyncio.Future[bytes]] = {}
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._limit = int(cache_mb * 1e6)

    async def _request(self, method: str, path: str, headers: dict[str, str]) -> httpx.Response:
        url = f"{self.base_url}/{path}"
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                async with async_slot(url):
                    resp = await self.client.request(method, url, headers=headers)
            except httpx.TransportError:
                if attempt == 4:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                self.stats.throttle()
                _log.warning(
                    "HTTP %s from %s (attempt %d), backing off", resp.status_code, url, attempt + 1
                )
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code == 404:
                raise FileNotFoundError(path)
            resp.raise_for_status()
            self.stats.add(len(resp.content), time.perf_counter() - t0)
            return resp
        raise RuntimeError("unreachable")  # pragma: no cover

    async def _page(self, path: str, index: int) -> bytes:
        key = (path, index)
        if key in self._pages:
            self._pages.move_to_end(key)
            return self._pages[key]
        if key in self._inflight:
            return await self._inflight[key]
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            size = await self._size(path)
            start = index * self.page_size
            stop = min(start + self.page_size, size)
            resp = await self._request("GET", path, {"Range": f"bytes={start}-{stop - 1}"})
            data = resp.content
            self._pages[key] = data
            self._bytes += len(data)
            while self._bytes > self._limit and len(self._pages) > 1:
                _, old = self._pages.popitem(last=False)
                self._bytes -= len(old)
            future.set_result(data)
            return data
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            del self._inflight[key]

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **_: Any
    ) -> bytes:
        size = await self._size(path)
        start = 0 if start is None else start
        end = size if end is None else min(end, size)
        first, last = start // self.page_size, max(end - 1, start) // self.page_size
        pages = await asyncio.gather(*(self._page(path, i) for i in range(first, last + 1)))
        blob = b"".join(pages)
        offset = first * self.page_size
        return blob[start - offset : end - offset]

    async def _size(self, path: str, **_: Any) -> int:
        if path not in self._sizes:
            resp = await self._request("HEAD", path, {})
            self._sizes[path] = int(resp.headers["content-length"])
        return self._sizes[path]

    async def aclose(self) -> None:
        await self.client.aclose()


class OpenMeteoSource:
    """Isobaric fields of one Open-Meteo archived model.

    Parameters
    ----------
    model:
        Open-Meteo model folder, e.g. ``"meteofrance_arpege_europe"`` (0.1 deg,
        from 2023-12-01) or ``"meteofrance_arpege_world025"`` (0.25 deg, from
        2023-12-21).
    fields:
        Output name -> key of :data:`OPENMETEO_VARIABLES` (default u, v, t).
    levels:
        Isobaric levels (hPa) allowed; default: all archived levels.
    tile_deg:
        Size of the geographic tiles read and cached (degrees).
    cache_mb:
        Memory budget of the block LRU.
    """

    time_step = np.timedelta64(1, "h")

    def __init__(
        self,
        model: str = "meteofrance_arpege_europe",
        *,
        fields: dict[str, str] | None = None,
        levels: Sequence[int] | None = None,
        base_url: str = OPENMETEO_BASE_URL,
        tile_deg: float = 5.0,
        cache_mb: float = 1024.0,
        max_connections: int = 64,
    ) -> None:
        self.model = model
        self.fields = dict(fields or {"u": "u", "v": "v", "t": "t"})
        unknown = set(self.fields.values()) - set(OPENMETEO_VARIABLES)
        if unknown:
            raise ValueError(f"unknown variables {sorted(unknown)}")
        self.base_url = base_url.rstrip("/")
        self.source_id = f"openmeteo-{model}"
        self.max_connections = max_connections
        self._stats = TransferStats()
        self._http = httpx.Client(timeout=60.0, follow_redirects=True)
        meta = json.loads(self._get_text(f"data/{model}/static/meta.json"))
        self.chunk_hours = int(meta["chunk_time_length"])
        if int(meta["temporal_resolution_seconds"]) != _HOUR_S:
            raise ValueError("only hourly archives are supported")
        self._end = np.datetime64(int(meta["data_end_time"]), "s").astype("datetime64[h]")
        bbox = _BBOX.search(meta["crs_wkt"])
        if bbox is None:
            raise ValueError("cannot read the grid bounding box from meta.json")
        self._bbox = tuple(float(x) for x in bbox.groups())  # lat_min, lon_min, lat_max, lon_max
        self.levels = self._archived_levels(levels)
        self._first_chunk = self._first_chunk_index()
        self._shape: tuple[int, int] | None = None
        self.tile_deg = tile_deg
        self._cache: OrderedDict[tuple[Any, ...], NDArray[np.float32]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(cache_mb * 1e6)
        self._lock = threading.Lock()
        # omfiles' async readers run on the process-wide I/O loop (limits.run_async);
        # the pooled HTTP client below lives on that loop.
        self._fs: _AsyncHttp | None = None

    def _run(self, coro: Any) -> Any:
        return run_async(coro)

    def _filesystem(self) -> _AsyncHttp:
        if self._fs is None:  # only touched from the shared loop thread
            self._fs = _AsyncHttp(self.base_url, self._stats, self.max_connections)
        return self._fs

    def __enter__(self) -> OpenMeteoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the HTTP pool (the shared I/O loop keeps running)."""
        fs, self._fs = self._fs, None
        if fs is not None:
            run_async(fs.aclose())
        self._http.close()

    # -- metadata ------------------------------------------------------------------------------
    @property
    def stats(self) -> list[TransferStats]:
        return [self._stats]

    def _get_text(self, path: str, params: dict[str, str] | None = None) -> str:
        t0 = time.perf_counter()
        resp = self._http.get(f"{self.base_url}/{path}", params=params)
        resp.raise_for_status()
        self._stats.add(len(resp.content), time.perf_counter() - t0)
        return resp.text

    def _list(self, prefix: str, delimiter: str | None = None) -> str:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = delimiter
        return self._get_text("", params)

    def _archived_levels(self, allowed: Sequence[int] | None) -> list[int]:
        listing = self._list(f"data/{self.model}/", "/")
        per_var: dict[str, set[int]] = {}
        for prefix in re.findall(r"<Prefix>data/[^/]+/([^<]+)/</Prefix>", listing):
            m = _LEVEL.match(prefix)
            if m:
                per_var.setdefault(m["var"], set()).add(int(m["hpa"]))
        common: set[int] | None = None
        for name in self.fields.values():
            lv = per_var.get(OPENMETEO_VARIABLES[name][0], set())
            common = lv if common is None else common & lv
        levels = sorted(common or set())
        if allowed is not None:
            levels = [x for x in levels if x in set(allowed)]
        if len(levels) < 2:
            raise ValueError(f"{self.model}: fewer than 2 common isobaric levels")
        return levels

    def _first_chunk_index(self) -> int:
        var = OPENMETEO_VARIABLES[next(iter(self.fields.values()))][0]
        listing = self._list(f"data/{self.model}/{var}_{self.levels[-1]}hPa/")
        chunks = [int(x) for x in re.findall(r"chunk_(\d+)\.om</Key>", listing)]
        if not chunks:
            raise ValueError(f"{self.model}: no archived chunks")
        return min(chunks)

    def coverage(self) -> tuple[np.datetime64, np.datetime64]:
        """First and last archived hours (the first chunk may start partly empty)."""
        start = np.datetime64(self._first_chunk * self.chunk_hours, "h")
        return start, self._end

    def cube_keys(self, t0: np.datetime64) -> tuple[CubeKey, CubeKey]:
        h = np.datetime64(t0, "h")
        return CubeKey(h), CubeKey(h + self.time_step)

    # -- grid ------------------------------------------------------------------------------------
    def _grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Latitudes descending and longitudes ascending of the archive grid."""
        if self._shape is None:
            raise RuntimeError("grid shape unknown before the first read")
        nlat, nlon = self._shape
        lat_min, lon_min, lat_max, lon_max = self._bbox
        lat = lat_min + np.arange(nlat) * (lat_max - lat_min) / (nlat - 1)
        lon = lon_min + np.arange(nlon) * (lon_max - lon_min) / (nlon - 1)
        return lat[::-1].copy(), lon

    # -- cubes -----------------------------------------------------------------------------------
    def fetch(
        self,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        *,
        level_margin: int = 1,
        executor: Executor | None = None,
    ) -> Cube:
        start, end = self.coverage()
        if not start <= key.valid <= end:
            raise DataUnavailableError(f"{self.source_id}: {key.valid} outside {start}..{end}")
        hours = int(key.valid.astype("datetime64[h]").astype(np.int64))
        chunk, t_in = divmod(hours, self.chunk_hours)
        vertical = pressure_levels(self.levels)
        positions = vertical.select_bracketing(
            p_range[0],
            p_range[1],
            STANDARD_SURFACE_PRESSURE,
            STANDARD_SURFACE_PRESSURE,
            level_margin,
        )
        if self._shape is None:
            # the newest file always has the (lat, lon, time) layout
            last_chunk = int(self._end.astype(np.int64)) // self.chunk_hours
            self._shape = self._run(self._read_shape(last_chunk))
        lat, lon = self._grid()
        grid = GridIndex.locate(domain, lat, lon)
        nlat = len(lat)
        rows = nlat - 1 - np.arange(grid.rows.start, grid.rows.stop)  # archive rows (lat ascending)
        cols = grid.cols
        wanted = [
            (name, int(hpa)) for name in self.fields.values() for hpa in vertical.ids[positions]
        ]
        blocks = self._blocks(wanted, chunk, rows, cols)
        fields: dict[str, NDArray[np.float32]] = {}
        for out_name, name in self.fields.items():
            offset = OPENMETEO_VARIABLES[name][1]
            layers = [
                _gather(blocks[(name, int(hpa))], rows, cols, self._tile_pts(), t_in) + offset
                for hpa in vertical.ids[positions]
            ]
            fields[out_name] = np.stack(layers).astype(np.float32)
        return Cube(
            source=self.source_id,
            hour=key.valid,
            lat=grid.lat,
            lon=grid.lon,
            vertical=_subset(vertical, positions),
            fields=fields,
            p_range=(float(p_range[0]), float(p_range[1])),
            level_margin=level_margin,
            request_domain=domain,
        )

    def _tile_pts(self) -> int:
        lat_min, _, lat_max, _ = self._bbox
        assert self._shape is not None  # noqa: S101 - set before any tiling
        step = (lat_max - lat_min) / (self._shape[0] - 1)
        return max(round(self.tile_deg / step), 1)

    def _blocks(
        self,
        wanted: list[tuple[str, int]],
        chunk: int,
        rows: NDArray[np.int64],
        cols: NDArray[np.int64],
    ) -> dict[tuple[str, int], dict[tuple[int, int], NDArray[np.float32]]]:
        """Tiles of every (variable, level) covering ``rows x cols`` for one time chunk."""
        tp = self._tile_pts()
        tiles = [(int(tr), int(tc)) for tr in np.unique(rows // tp) for tc in np.unique(cols // tp)]
        out: dict[tuple[str, int], dict[tuple[int, int], NDArray[np.float32]]] = {
            w: {} for w in wanted
        }
        missing = []
        with self._lock:
            for name, hpa in wanted:
                for tile in tiles:
                    ck = (name, hpa, chunk, tile)
                    if ck in self._cache:
                        self._cache.move_to_end(ck)
                        out[(name, hpa)][tile] = self._cache[ck]
                    else:
                        missing.append(ck)
        if missing:
            read = self._run(self._read_tiles(missing, tp))
            with self._lock:
                for ck, arr in zip(missing, read, strict=True):
                    out[(ck[0], ck[1])][ck[3]] = arr
                    self._remember(ck, arr)
        return out

    def _remember(self, key: tuple[Any, ...], arr: NDArray[np.float32]) -> None:
        self._cache[key] = arr
        self._cache_bytes += arr.nbytes
        while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes

    def _path(self, name: str, hpa: int, chunk: int) -> str:
        return f"data/{self.model}/{OPENMETEO_VARIABLES[name][0]}_{hpa}hPa/chunk_{chunk}.om"

    async def _read_shape(self, chunk: int) -> tuple[int, int]:
        omfiles = _omfiles()
        name = next(iter(self.fields.values()))
        try:
            reader = await omfiles.OmFileReaderAsync.from_fsspec(
                self._filesystem(), self._path(name, self.levels[-1], chunk)
            )
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"{self.source_id}: chunk {chunk} not archived") from exc
        shape = tuple(reader.shape)
        reader.close()
        if len(shape) != 3:
            raise ValueError(f"{self.source_id}: unexpected layout {shape} in the newest file")
        return int(shape[0]), int(shape[1])

    async def _read_tiles(self, keys: list[tuple[Any, ...]], tp: int) -> list[NDArray[np.float32]]:
        """Read every (variable, level, chunk, tile) block concurrently."""
        omfiles = _omfiles()
        assert self._shape is not None  # noqa: S101
        nlat, nlon = self._shape
        fs = self._filesystem()
        paths = sorted({self._path(k[0], k[1], k[2]) for k in keys})
        try:
            opened = await asyncio.gather(
                *(omfiles.OmFileReaderAsync.from_fsspec(fs, p) for p in paths)
            )
        except FileNotFoundError as exc:
            raise DataUnavailableError(f"{self.source_id}: {exc} not archived") from exc
        readers = dict(zip(paths, opened, strict=True))

        async def one(k: tuple[Any, ...]) -> NDArray[np.float32]:
            r0, c0 = k[3][0] * tp, k[3][1] * tp
            r1, c1 = min(r0 + tp, nlat), min(c0 + tp, nlon)
            reader = readers[self._path(k[0], k[1], k[2])]
            if len(reader.shape) == 3:  # (lat, lon, time), files from 2025 on
                arr = await reader.read_array((slice(r0, r1), slice(c0, c1), slice(None)))
                return np.asarray(arr, dtype=np.float32)
            # older files: (lat * lon, time), row-major; read whole rows then crop
            if reader.shape[0] != nlat * nlon:
                raise ValueError(f"{self.source_id}: {reader.shape} does not match {nlat}x{nlon}")
            flat = await reader.read_array((slice(r0 * nlon, r1 * nlon), slice(None)))
            rows = np.asarray(flat, dtype=np.float32).reshape(r1 - r0, nlon, -1)
            return np.ascontiguousarray(rows[:, c0:c1])

        try:
            return list(await asyncio.gather(*(one(k) for k in keys)))
        finally:
            for r in readers.values():
                try:
                    r.close()
                except RuntimeError:  # reader still busy after a failed read
                    pass


def _gather(
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


def _subset(vertical: VerticalCoordinate, positions: NDArray[np.int64]) -> VerticalCoordinate:
    return vertical.subset(positions)


def _omfiles() -> Any:
    try:
        import omfiles
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading Open-Meteo archives needs: pip install 'supertraj-met[openmeteo]'"
        ) from exc
    return omfiles
