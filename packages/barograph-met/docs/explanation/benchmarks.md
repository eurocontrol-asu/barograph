# Benchmarks

## Summary

Two samples of 1 000 real flights drawn from OpenSky across all ICAO regions
(one point every 10 s, surface to FL410): 18 June 2025 for ERA5, 28 September
2026 for ARPEGE (Météo-France keeps its files 14 days). Flights are added to
each tool's own cache in steps of 1, 10, 100 and 1 000; costs are cumulative
from a cold cache, through a corporate HTTPS proxy; values are compared on
every point both tools serve.

| Case | Tool | 1 flight: time · download | 1 000 flights: time · download | Disk | Peak RAM | Values vs reference (1 000 flights) |
|---|---|---:|---:|---:|---:|---|
| ERA5, 22 pressure levels | FastMeteo 1.2.0 | 330 s · 3.6 GB | 744 s · 8.1 GB | 4.9 GB | 5.8 GB | reference |
| | **barograph-met** | **15 s · 0.09 GB** | **302 s · 3.6 GB** | **3.6 GB** | **1.5 GB** | identical: 99th pct \|Δ\| < 2e-8 on 548 098 points ¹ |
| ERA5, model levels | pycontrails 0.63.5 | 753 s · 13 GB | 1 893 s · 29.6 GB | 23.7 GB | 39 GB | reference |
| | **barograph-met** (L137) | **34 s · 0.27 GB** | **900 s · 14.6 GB** | **14.6 GB** | **3.8 GB** | RMS 0.037 m/s (u, v), 0.021 K (T); 99th pct 0.15 m/s, 0.08 K on 545 466 points ² |
| ARPEGE 0.25°, official | FastMeteo ³ | 236 s · 2.6 GB | 504 s · 2.6 GB | 2.6 GB | 13 GB | reference |
| | **barograph-met** | **95 s · 0.20 GB** | **211 s · 1.25 GB** | **1.2 GB** | **2.1 GB** | identical with the same 22 levels: \|Δ\| ≤ 3e-8 on 574 673 points ⁴ |

At 1 000 flights barograph-met is 2.1–2.5× faster, downloads 2–2.3× less and
needs 4–10× less memory; for a single flight it downloads 13–48× less and is
2.5–22× faster (ARPEGE: indexing the files dominates the first call). FastMeteo and
pycontrails download whole global fields for every hour they need, so their
cost reaches a full day as soon as the flights cover it; barograph-met reads
only the latitude bands, levels and messages the flights cross, so its cost
grows with the area covered and levels off. Points it cannot interpolate
(below the lowest level, invalid altitudes: 0.4–0.8 %) are flagged, never
extrapolated.

1. Including the points where FastMeteo 1.2.0 extrapolates across the
   Greenwich seam, RMS 0.006 m/s and 0.002 K.
2. Different vertical methods: pycontrails interpolates model levels to
   pressure levels at a reference surface pressure, barograph-met interpolates
   the model levels directly with each column's actual surface pressure.
3. FastMeteo 1.2.0 cannot reach Météo-France's new data.gouv.fr host; it runs
   from the branch of [fastmeteo#5](https://github.com/open-aviation/fastmeteo/pull/5),
   in its only full-day mode (00 UTC run, one 000H024H file).
4. By default barograph-met also uses the 275 and 925 hPa levels that
   FastMeteo drops: 0.23 m/s and 0.12 K RMS from FastMeteo's values.

The sections below give the single-flight measurements and every step.

## One flight

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

## A worldwide fleet

1 000 flights of 18 June 2025 sampled from OpenSky across all ICAO regions
(550 000 points every 10 s, surface to FL410), added in steps of 1, 10, 100
and 1 000 flights to each tool's own cache. Cumulative cost from a cold cache:

| Flights | FastMeteo, ERA5 22 PL | barograph-met, same | pycontrails, ERA5 ML | barograph-met, ML137 |
|---:|---:|---:|---:|---:|
| 1 | 330 s, 3.6 GB | **15 s, 0.09 GB** | 753 s, 13.0 GB | **34 s, 0.27 GB** |
| 10 | 680 s, 7.5 GB | **85 s, 0.84 GB** | 1 642 s, 27.2 GB | **215 s, 2.8 GB** |
| 100 | 740 s, 8.1 GB | **225 s, 2.8 GB** | 1 852 s, 29.6 GB | **634 s, 9.9 GB** |
| 1 000 | 744 s, 8.1 GB | **302 s, 3.6 GB** | 1 893 s, 29.6 GB | **900 s, 14.6 GB** |
| Peak RAM | 5.8 GB | **1.5 GB** | 39 GB | **3.8 GB** |

FastMeteo and pycontrails download whole global fields for every hour they
need: their cost jumps to a full day as soon as the flights cover it, then
stays flat. barograph-met reads only the latitude bands and levels the
flights cross, so its cost grows with the area covered and levels off; with
1 000 flights spread over the globe it still downloads 2.3× (PL) and 2×
(ML) less. Values over the 545 000 comparable points: identical to
FastMeteo (99th percentile of |Δ| < 2e-8) except where FastMeteo extrapolates
at the Greenwich seam; versus pycontrails RMS 0.037 m/s (u, v) and 0.021 K (T),
99th percentile 0.15 m/s and 0.08 K. Points below the lowest level or with
invalid altitudes (0.8 %) are flagged rather than extrapolated.

ARPEGE 0.25° global, official Météo-France files: 1 000 flights of
28 September 2026 sampled the same way (577 000 points), 00 UTC run (FastMeteo's
only full-day mode, one 000H024H file; FastMeteo 1.2.0 cannot reach the new
data.gouv.fr host, so it runs from the branch of
[fastmeteo#5](https://github.com/open-aviation/fastmeteo/pull/5)):

| Flights | FastMeteo | barograph-met |
|---:|---:|---:|
| 1 | 236 s, 2.6 GB | **95 s, 0.20 GB** |
| 10 | 318 s, 2.6 GB | **128 s, 0.84 GB** |
| 100 | 411 s, 2.6 GB | **175 s, 1.2 GB** |
| 1 000 | 504 s, 2.6 GB | **211 s, 1.25 GB** |
| Peak RAM | 13 GB | **2.1 GB** |

barograph-met fetches only the GRIB messages (variable × level × step) the
flights need, and its first call includes indexing the files (cached after).
With FastMeteo's 22 levels the values are identical (|Δ| ≤ 3e-8 over 574 673
points); by default barograph-met also uses the 275 and 925 hPa levels
FastMeteo drops, which moves the values by 0.23 m/s and 0.12 K RMS.


Protocol: flights in a fixed random order (seed 0), each step enriching only
the flights added since the previous one with a persistent cache, as a
campaign grows; pycontrails' pressure levels are fixed once for the fleet's
altitude range (its cache key includes them). Sample and scripts:
`fetch_world_sample.py`, `run_fleet.py`, `report_fleet.py` (paper repository).
