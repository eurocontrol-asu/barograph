# Campaigns: cache, domains and parallelism

## Enrich many flights at once

Pass the points of **all** flights to one `enrich` call: each time slice is
read once for every flight that crosses it.

## A shared, reusable cache

```python
from supertraj_met import Domain, Enricher, arco_model_levels

europe = Domain(lat_min=30, lat_max=72, lon_min=-20, lon_max=45)
enricher = Enricher(arco_model_levels(), cache_dir="/shared/cubes", domain=europe)
```

- A fixed `domain` makes every cube cover the same box, so later runs and
  other experiments reuse them.
- `mode="global"` reads whole fields on every level (FastMeteo-like cost): one
  cache then serves any flight.
- The cache is immutable and written atomically; a per-slot lock lets several
  processes share it and never download the same cube twice.

## Parallelism and politeness

- `io_workers` (default 16) concurrent range requests, `prefetch` (default 2)
  cubes fetched ahead of interpolation.
- Concurrent requests are capped **per remote host for the whole process**:
  Google Cloud Storage and AWS S3 32, Météo-France 6 (it asks users not to
  download in parallel), others 16. Change a cap with
  `supertraj_met.set_host_limit(host, n)`.
- 429/5xx answers are retried with backoff (honouring `Retry-After`), logged
  on the `supertraj_met.http` logger and counted in `res.stats["throttled"]`.
