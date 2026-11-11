# AROME pressure-level archive

Use the generic Open-Meteo source with `meteofrance_arome_france0025` for
hourly AROME wind components and temperature along a trajectory:

```python
from baro.met import Enricher
from baro.met.openmeteo import OpenMeteoSource

with OpenMeteoSource(
    "meteofrance_arome_france0025",
    tile_deg=0.25,
    merge_adjacent_tiles=False,
) as source:
    result = Enricher(
        source,
        cache_dir="weather-cache",
        margin_deg=0.025,
        level_margin=0,
    ).enrich(time_utc, latitude, longitude, altitude_ft=pressure_altitude_ft)
    u, v, temperature = result["u"], result["v"], result["t"]
```

Install the `openmeteo` extra. Coordinates and times must be one-dimensional
arrays of equal length; altitude is barometric altitude in feet. Output wind
is in m/s and temperature in kelvin. Each point uses horizontal, pressure
and time interpolation. Missing or outside points have flags and NaN values.
Pressure levels alone do not supply terrain or actual surface pressure.

**Other variables**: `fields=` also accepts the other names listed in
[Variables by source](../reference/variables.md); wind and temperature are only
the defaults.

`merge_adjacent_tiles=False` groups points by their own native tile, with
the requested margin and pressure range applied separately to each group.
This bounds the size of each request even when a long trajectory occupies
a continuous corridor. The default `True` joins touching tiles, which can
fill a large rectangular area around a narrow route. The tile memory budget
limits retained cache entries, not the temporary arrays of a fetch: each
tile decodes all samples of its native time chunk.

The archive is a stitched forecast series: the latest available runs are
combined at each valid time, without retained run identity or forecast lead.
The source discovers native grid, cadence, available levels and coverage from
the archive rather than applying AROME-specific constants. The
`meteofrance_arome_france_hd` archive has no common pressure levels for this
interface; it is not a substitute for the 0.025° pressure-level model.

## Live validation

A bounded validation on 7 October 2026 read real April 2026 trajectory points
through the public Open-Meteo bucket. Its metadata gave a 717 × 1121 grid,
latitude 37.5–55.4°N, longitude −12–16°E, spacing 0.025°, hourly samples and
108 samples per archive chunk. The 24 common u/v/t levels were 100, 125, 150,
175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
800, 850, 900, 925, 950 and 1000 hPa. Coverage reported a first chunk starting
23 December 2023 at 12 UTC; this is a chunk lower bound and its earliest
samples may be empty. Call `source.coverage()` for the current endpoints.

The sample comprised 22 cruise points between 48 and 49°N plus one point at
58.01°N, beyond the regional grid. All cruise points returned finite u/v/t;
the outside point returned `OUTSIDE_DOMAIN`. Values ranged from
19.35 to 21.36 m/s for u, 18.68 to 22.76 m/s for v, and 215.95 to 216.12 K
for temperature. This checks real archive decoding and interpolation, not
accuracy against independent observations.

Three tile sizes were tested on identical points with independent initially
empty disk caches, a 256 MB tile memory budget, 0.025° geographic margin,
zero extra pressure levels, four I/O workers and one prefetched cube.
Timings below cover enrichment with the default touching-tile grouping;
initial metadata discovery is excluded.

| Tile edge | Cold time | Download | Requests | Warm time | Warm download |
|---|---:|---:|---:|---:|---:|
| 0.25° | 9.36 s | 18.52 MB | 52 | 0.046 s | 0 bytes |
| 0.5° | 16.35 s | 21.67 MB | 58 | 0.033 s | 0 bytes |
| 1.0° | 10.86 s | 25.34 MB | 65 | 0.019 s | 0 bytes |

All three configurations produced identical values and flags, fetching four
cubes and only the 200, 225 and 250 hPa levels. Downloads include file headers
and ranged pages; reading a tile decodes all 108 native samples of its chunk.
Smaller tiles reduced bytes for this narrow sample. The timings are single
runs and depend on network conditions; this sample does not establish a
campaign-wide optimum. Choose tile size and margins for your trajectories;
the generic defaults are unchanged.

Local regression tests cover the 108-hour chunk boundary, selective pressure
levels, reuse, missing selected/unselected chunks, common-level intersection
and regional latitude clipping. Per-tile grouping also reproduces the legacy
interpolated values while bounding geographic and vertical requests; unit
tests cover overlapping margins and equivalent longitudes at the dateline.
They use synthetic mirrors without network
access or private trajectory files.

## Full-trajectory request size

A 7 294-point trajectory exposed a limitation of connected-tile grouping.
One hourly request covered a clipped 298 × 546 grid rectangle and 12 pressure
levels, including 1 705 tiles while only 82 tiles contained nearest trajectory
points. Decoding three variables across all 108 native timestamps required
about 2.65 GB of temporary tile arrays for a 23.43 MB single-time cube. The
run was stopped after its process reached about 4.7 GB RSS. This was not a
completed timing comparison.

With `merge_adjacent_tiles=False`, the complete trajectory returned in
78.72 s with 346.49 MiB peak process RSS and 222 small cubes (0.43 MB summed
cube arrays). It reused disk pages from the interrupted run: 37.58 MB of
additional network transfer and 179.29 MB of disk cache hits. These are warm
disk-cache measurements, not cold costs. All 5 278 valid points had finite
u/v/t; 1 821 points beyond the metadata grid were flagged `OUTSIDE_DOMAIN`,
two at 325 ft were `BELOW_BOTTOM`, and 193 along the northern edge of the
metadata grid were `MISSING_DATA`. The earlier narrow sample reproduced the
same flags and values within 4e-13. Per-tile grouping avoids filling the
empty rectangular area around a connected flight corridor and selects
pressure levels for each local group.
