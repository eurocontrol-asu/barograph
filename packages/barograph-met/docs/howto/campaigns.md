# Campaigns: cache, fleets and parallelism

## One flight or a fleet: the cost is the area covered

`enrich` works on points, not flights. The points of each time slice are split
along the source's own storage tiling (latitude bands of Blosc blocks for
ARCO-ERA5, square tiles for the Open-Meteo archive): touching occupied tiles
form one cube, distant groups get their own. What is downloaded is the set of
tiles the traffic actually crosses, whatever the number of flights. Passing a
whole fleet in one call is simplest; splitting it into calls costs the same
with a cache (below).

## A shared, reusable cache

```python
from baro.met import Enricher, arco_model_levels

enricher = Enricher(arco_model_levels(), cache_dir="/shared/era5-cache")
```

- Every byte read from a remote object is kept on disk, in the source's own
  units (runs of Blosc blocks, GRIB messages, `.om` pages). A later call only
  downloads the ranges no earlier call has read: flight by flight, a fleet at
  once, in any order, from several processes.
- Only immutable data is kept: ERA5 hours older than about three months (the
  preliminary ERA5T may still be replaced), ARPEGE files of a published run,
  Open-Meteo chunks older than a week (recent hours are rewritten by newer
  runs). Store metadata is always re-read.
- Extents are written atomically and never modified; missing ranges are
  fetched under file locks on the 1 MiB windows they cover, so threads and
  processes sharing the cache never download the same bytes twice, while
  disjoint reads stay parallel.
- The cache only grows (nothing is ever evicted): size it for the campaign
  (for 1 000 worldwide flights of one day, about 3.6 GB for ERA5 on 22
  pressure levels and 15 GB on the 137 model levels) and delete the folder,
  or part of it, when no longer needed; nothing else refers to it.
- A fixed `domain=` still forces one box per time slice, and `mode="global"`
  reads whole fields on every level (FastMeteo-like cost).

## Parallelism and politeness

- `io_workers` (default 16) concurrent range requests, `prefetch` (default 2)
  cubes fetched ahead of interpolation.
- Concurrent requests are capped **per remote host for the whole process**:
  Google Cloud Storage and AWS S3 32, Météo-France 6 (it asks users not to
  download in parallel), others 16. Change a cap with
  `baro.met.set_host_limit(host, n)`.
- 429/5xx answers are retried with backoff (honouring `Retry-After`), logged
  on the `baro.met.http` logger and counted in `res.stats["throttled"]`.

## Reuse decoded weather between calls

`Enricher` releases decoded cubes after each call. For successive pressure-point
queries, a `WeatherSession` also keeps a bounded in-memory LRU:

```python
from baro.met import WeatherSession, arco_pressure_levels

source = arco_pressure_levels()
session = WeatherSession(
    source,
    fields=["u", "v"],
    vertical="linear",
    max_cache_bytes=128 * 1024 * 1024,
    cache_dir=".weather-cache",
)
result = session.sample(
    ["2020-01-01T12:30:00.123456789"],
    [48.0],
    [2.0],
    pressure_pa=[25_000.0],
)
assert result.valid.all()
print(result["u"], session.cache_info())
session.clear()
```

The session uses the same storage tiling, pressure bracketing, byte cache and
interpolation as `Enricher`. It reuses a decoded request when the validity time,
forecast run and requested region and pressure range are covered. Leaving that
region triggers a normal source fetch. It never extrapolates missing weather;
check result flags before interpreting a model state as physically expired.

Field selection uses `GridSource.select_fields`, an independent field mapping
sharing the original stores. Other sources can be configured with their own
fields before creating the session. Fields are fixed for the session's lifetime.
Do not mutate source configuration during use. Like `Enricher`, attaching an
on-disk cache changes the shared stores' byte-cache configuration.

The byte limit includes compact copies of fields, surface pressure, coordinates
and vertical coefficients retained by the LRU. It excludes Python overhead and
active interpolation/prefetch allocations; an oversized cube is sampled but
not retained. Cache counters in `result.stats` separate decoded hits and misses
from source transfer statistics. Calls on one session are serialized; each
call keeps the usual concurrent source reads. A session represents a snapshot:
use `clear()` when refreshing mutable forecasts or ERA5T. Clearing does not
remove the immutable on-disk byte cache or reset lifetime LRU counters.
