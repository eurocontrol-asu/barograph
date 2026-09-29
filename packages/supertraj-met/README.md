# supertraj-met

Wind (`u`, `v`) and temperature (`t`) at aircraft positions (time, latitude,
longitude, barometric altitude) from **ERA5** (137 native model levels or
pressure levels), **Météo-France ARPEGE** (live files and the Open-Meteo
archive) — fast, with bounded transfer, explicit coverage and no silent
extrapolation. Part of the [supertraj](../../README.md) workspace.

```python
import numpy as np
from supertraj_met import Enricher, arco_model_levels, track_components

enricher = Enricher(arco_model_levels(), cache_dir="~/data/supertraj-met-cache")
res = enricher.enrich(time_utc, lat, lon, altitude_ft=baro_alt_ft)
res["u"], res["v"], res["t"], res.flags, res.stats
along, cross = track_components(res["u"], res["v"], track_deg)
```

## Why

The public [ARCO-ERA5](https://github.com/google-research/arco-era5) zarr
stores keep a **global** latitude × longitude field in every chunk
(`[1 h, 18 levels, 721, 1440]` for model levels, `[1 h, 37 levels, 721, 1440]`
for pressure levels). Tools that read whole chunks (FastMeteo, pycontrails'
ARCO path) transfer ~1.2 GB per hour for u/v/t on model levels, whatever the
region.

Each chunk is, however, a Blosc frame made of independently compressed
~512 KiB blocks (≈ 90 latitude rows of one level), and the block offsets sit
in the frame header. `supertraj_met` reads the header with one range request, then
fetches **only the blocks covering the needed levels and latitude band**.

## Benchmarks

Same points for every tool, cold cache (first run), through a corporate HTTPS
proxy; warm = second run with the tool's own cache.

| Comparison | Cold time | Warm time | Downloaded | Cache on disk | Peak RAM | Values |
|---|---:|---:|---:|---:|---:|---|
| **ERA5 PL22** — FastMeteo 1.2.0 | 214 s | 1.1 s | 1296 MB | 778 MB | 1.5 GB | reference |
| **ERA5 PL22** — `supertraj-met` (same 22 levels, linear in p) | **12 s** | 0.3 s | **100 MB** | 0.4 MB | 0.16 GB | identical (≤ 1e-13), except FastMeteo's extrapolated points at the Greenwich seam |
| **ERA5 ML** — pycontrails 0.63.5 (ML → 61 PL → 4-D) | 322 s | 1.3 s | 4736 MB | 3040 MB | 7.0 GB | reference |
| **ERA5 ML137** — `supertraj-met` (direct, ln p, actual ps) | **31 s** | 0.9 s | **349 MB** | 1.3 MB | 0.22 GB | RMS 0.03 m/s (u, v), 0.016 K (T); max 0.5 m/s near the ground |
| **ARPEGE 0.1°** — FastMeteo 1.2.0 | 67 s | 2.0 s | 475 MB | — | — | reference |
| **ARPEGE 0.1°** — `supertraj-met` | **13 s** | 0.2 s | **54 MB** | — | — | identical east of Greenwich; FastMeteo extrapolates wildly west of it |

ERA5 rows: one 2 h flight from the surface to FL350 (1 764 points, 4 hourly
analyses). ARPEGE rows: a 3 h synthetic flight across France (400 points, run
2026-05-11 00 UTC). Network time dominates everywhere (> 98 % of the cold
time); decoding and interpolation take well under a second.

## Sources

| Source | Vertical | Grid / time | Access |
|---|---|---|---|
| `arco_model_levels()` | ERA5 L137 hybrid levels, actual surface pressure | 0.25 deg, hourly, 1940 → ~3 months ago | ARCO zarr, block-range reads |
| `arco_pressure_levels(levels=...)` | ERA5 37 pressure levels (or a subset) | 0.25 deg, hourly | ARCO zarr, block-range reads |
| `ArpegeSource(resolution="01"|"025")` | ARPEGE 24 (Europe 0.1 deg) / 34 (global 0.25 deg) pressure levels | hourly to 48 h; runs 00/06/12/18 UTC; **14-day retention, no open archive** | Météo-France open data (GRIB2), message-range reads with a cached per-file index; needs the `grib` extra |

| `OpenMeteoSource(model)` | Open-Meteo archive of a model (e.g. `meteofrance_arpege_europe` from 2023-12-01, `meteofrance_arpege_world025` from 2023-12-22): pressure levels, most recent run at each hour, values quantised (T in 0.5 K steps) | hourly | `s3://openmeteo` (CC-BY 4.0), needs the `openmeteo` extra |
| `SourceChain(a, b, ...)` | first source serving each interval wins, e.g. official ARPEGE (14 days) then the Open-Meteo archive; measured hand-over gap 0.035 m/s, 0.08 K RMS | — | — |

Every source declares its **coverage**; points outside it are flagged
`OUT_OF_COVERAGE` with a warning naming the covered period
(`on_uncovered="warn" | "raise" | "flag"`).

ARPEGE files are fetched from `files.data.gouv.fr/meteofrance-pnt/pnt` (the store
learns its redirect to Météo-France's object storage). A GRIB2 message cannot be
cropped, so only the variables, levels and steps needed are downloaded — e.g.
35 MB instead of a 476 MB file (0.1 deg) or 90 MB instead of 2.57 GB (0.25 deg)
for a 2.5 h flight once the file index is cached.

## FastMeteo-compatible façade

```python
from supertraj_met.compat import ArcoEra5, Arpege  # pip install 'supertraj-met[pandas,grib]'

ArcoEra5("cache/", model_levels=37).interpolate(df)  # FastMeteo's 22 levels, linear in p
ArcoEra5("cache/", model_levels=137).interpolate(df)  # all 137 levels + surface pressure
Arpege("cache/", resolution="01").interpolate(df)  # same columns as FastMeteo
ArcoEra5("cache/", mode="global").interpolate(df)  # whole fields: cache serves any flight
```

Same constructors, `interpolate(df)` contract (`timestamp`, `latitude`,
`longitude`, `altitude` in ft) and output columns as FastMeteo 1.2. Differences:
NaN (with a reason in `last_result.flags`) instead of extrapolation, longitudes
handled across Greenwich, and an immutable cube cache instead of a global zarr.

## What it does

* **Vertical coordinate from the actual surface pressure.** Full-level
  pressures `p_k = A_k + B_k · ps` (IFS L137 a/b, full level = mean of the
  adjacent half levels), computed per grid column; `ps` comes from the ARCO
  pressure-level store (`surface_pressure`), since the model-level store has
  none.
* **Barometric altitude → pressure with the ICAO ISA** (exact: ADS-B altitude
  is a pressure altitude).
* **Interpolation**: for each of the 4 horizontal corners, bracket the
  aircraft pressure in that column and interpolate in `ln p` (or `p`), then
  bilinear in latitude/longitude, then linear in time between hourly analyses.
* **Never extrapolates**: `Flag.OUTSIDE_DOMAIN`, `ABOVE_TOP`, `BELOW_BOTTOM`,
  `MISSING_DATA` (e.g. hour not published), `INVALID_INPUT`; values are NaN
  when a flag is set.
* **Hour-major planning**: points of *all* flights are grouped by hour, each
  hour is fetched once over the box and pressure range its points need, with
  bounded look-ahead and parallel range requests.
* **Immutable cube cache** (`cache_dir`): one atomically written `.npz` +
  manifest per (source, hour, domain, levels). Safe to share between processes;
  a cube is reused whenever it encloses a later request.
* **Ablation switches**: `surface_pressure="standard"` reproduces a fixed
  altitude→level table (FastMeteo's model-level assumption);
  `vertical="linear"` interpolates linearly in `p`; `arco_pressure_levels()`
  runs the same kernel on the 37 pressure levels.

For campaigns, pass a fixed `domain=Domain(...)` so every hour is cached over
the same box and reused by later runs and methods.

## Parallelism and politeness

* Within an enrichment: a thread pool of range requests (`io_workers`) and
  cubes fetched ahead (`prefetch`); Open-Meteo reads run on one process-wide
  asyncio loop, with identical page requests merged.
* **Per-host caps shared by the whole process** (`supertraj_met.limits`): Google
  Cloud Storage and AWS S3 32 concurrent requests, Météo-France 6 (it asks API
  users not to download in parallel), others 16; change with
  `set_host_limit(host, n)`. 429/5xx answers are retried with backoff
  (`Retry-After` honoured), logged on `supertraj_met.http` and counted in
  `stats["throttled"]`.
* Across processes: the cube cache is immutable and a per-slot `flock`
  claim makes concurrent workers wait for, then reuse, a cube another worker
  is downloading — each cube is fetched once per campaign.

## Prior art

This work builds on [FastMeteo](https://github.com/open-aviation/fastmeteo)
(Sun & Roosenbrand, JOAS 2023, doi:10.59490/joas.2023.7264), which pioneered
lightweight ERA5/ARPEGE enrichment of ADS-B trajectories from ARCO-ERA5, and on
the ARCO-ERA5 project (Carver & Merose, 2023). pycontrails' model-level
handling (Shapiro et al., Zenodo) was also a reference.

## Development

```bash
uv sync
uv run pytest            # unit + integration on synthetic ARCO-like stores
uv run pytest -m live    # opt-in: a few tens of MB from the public bucket
uv run ruff check src tests && uv run mypy
```

## Data licence

ERA5 is produced by the Copernicus Climate Change Service (C3S) at ECMWF and
distributed under the Copernicus licence; acknowledge C3S and cite
Hersbach et al. (2020), doi:10.1002/qj.3803, and ARCO-ERA5 (Carver & Merose, 2023).
