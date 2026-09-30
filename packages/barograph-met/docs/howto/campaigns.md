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
- Extents are written atomically and never modified; a lock per unit of work
  lets several processes share the cache without downloading the same bytes
  twice.
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
