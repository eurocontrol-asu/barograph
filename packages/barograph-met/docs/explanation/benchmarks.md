# Benchmarks

Same points for every tool, cold cache (first run), through a corporate HTTPS
proxy; warm = second run with the tool's own cache.

| Comparison | Cold | Warm | Downloaded | Cache on disk | Peak RAM | Values |
|---|---:|---:|---:|---:|---:|---|
| ERA5 PL22 — FastMeteo 1.2.0 | 214 s | 1.1 s | 1296 MB | 778 MB | 1.5 GB | reference |
| ERA5 PL22 — barograph-met (same levels, linear in p) | **12 s** | 0.3 s | **100 MB** | 0.4 MB | 0.16 GB | identical (≤ 1e-13) except FastMeteo's extrapolated points at the Greenwich seam |
| ERA5 ML — pycontrails 0.63.5 (ML → 61 PL → 4-D) | 322 s | 1.3 s | 4736 MB | 3040 MB | 7.0 GB | reference |
| ERA5 ML137 — barograph-met (direct, ln p, actual ps) | **31 s** | 0.9 s | **349 MB** | 1.3 MB | 0.22 GB | RMS 0.03 m/s (u, v), 0.016 K (T) |
| ARPEGE 0.1° — FastMeteo 1.2.0 | 67 s | 2.0 s | 475 MB | — | — | reference |
| ARPEGE 0.1° — barograph-met | **13 s** | 0.2 s | **54 MB** | — | — | identical east of Greenwich; FastMeteo extrapolates west of it |

ERA5 rows: one 2 h flight trajectory from the surface to FL350 (1 764 points,
4 hourly analyses). ARPEGE rows: a 3 h synthetic flight across France (400
points, run 2026-05-11 00 UTC). Network time dominates (> 98 % of the cold
time); decoding and interpolation take well under a second.

> **Scope: one flight.** These figures compare a single trajectory. With many
> flights on the same day, tools that download whole global fields (FastMeteo,
> pycontrails) amortise them across flights, while barograph-met reads more
> bands as flights spread out, so the gap narrows. A multi-flight benchmark
> (1 to 1 000 flights on one day) is planned.
