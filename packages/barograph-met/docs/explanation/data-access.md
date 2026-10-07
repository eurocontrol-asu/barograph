# Data access

## ARCO-ERA5 (zarr v2, Blosc)

Every chunk holds a **global** latitude × longitude field
(`[1 h, 18 levels, 721, 1440]` for model levels), so a regional selection
never makes a chunk smaller. Each chunk is however a Blosc frame of
independently compressed ~512 KiB blocks (≈ 90 latitude rows of one level)
whose offsets are in the frame header. The reader fetches the header, then
only the block runs covering the requested levels and latitude band, and
re-frames them for the decoder. Blocks may be stored out of order
(multi-threaded compression); spans are computed from the physical order.

## ARPEGE (GRIB2 without inventory)

Messages (variable × level × step) are concatenated; each announces its
length. The index is built by reading the first bytes of every message, in
parallel segments whose starts are found by scanning for `GRIB` and verified
by chaining lengths (with a sequential fallback). A message cannot be
cropped, so the saving comes from selecting variables, levels and steps.

## Open-Meteo archive (`.om`)

Files are chunked as 1 latitude row × 14 longitudes × all native time steps of the file:
reading one step costs a whole compressed block, so tiles are read for all steps
and kept in memory. Requests are served from aligned 512 KiB pages, merging
the many tiny reads of the decoder. Two layouts coexist (flattened before
early 2025, `(lat, lon, time)` after); both are read.


`OpenMeteoSource("ecmwf_ifs025", fields={"u": "u", "v": "v", "t": "t"})`
reads IFS at 0.25 degrees and a native step of 3 hours. ARPEGE remains hourly.
The metadata field `chunk_time_length` counts **samples**, not hours: IFS has
104 samples per file, spanning 312 hours. For Unix time `t` in seconds,
`index = t // temporal_resolution_seconds`, and
`chunk, offset = divmod(index, chunk_time_length)`. The metadata's
`data_end_time` is exclusive; Barograph exposes the last native valid time
as its inclusive coverage endpoint. This follows Open-Meteo's
[metadata definition](https://github.com/open-meteo/open-meteo/blob/39a0b8c44b2ec2c86b26e89acaa2a7bb9340776f/Sources/App/Helper/File/ModelMetaJson.swift)
and [time indexing](https://github.com/open-meteo/open-meteo/blob/39a0b8c44b2ec2c86b26e89acaa2a7bb9340776f/Sources/OmTime/Time.swift).

The enricher groups points by native interval and interpolates linearly in
time between its two endpoints. It reads only the requested variables,
pressure levels bracketing the points (plus `level_margin`), and geographic
tiles intersecting each requested region. No global field is downloaded.
Compressed blocks and aligned HTTP pages impose some over-read; subsequent
points and flights reuse those bytes. Immutable chunks can also be cached
on disk via `Enricher(source, cache_dir=...)`.

This is an **assembled forecast archive, not an ERA5 reanalysis**. Original
run identities and lead times are unavailable; `CubeKey.run` and `.lead`
are `None`. Directory listings describe levels available at some time, not
necessarily every date. For 2026-04-15, direct reads verified `u/v/t` on 50, 100, 150, 200,
250, 300, 400, 500, 600, 700, 850, 925 and 1000 hPa. The 10 hPa files
were absent despite their listed directories. A missing file is reported
as unavailable rather than extrapolated. Coverage metadata is a broad bound, not a guarantee
that every field, level and timestamp is populated.


### Choosing tile size

The generic `tile_deg` option also applies to IFS. The shared default remains
5 degrees; a smaller tile reduces decoded cache memory but can increase
per-tile overhead. For a memory-conscious IFS workload, try:

```python
source = OpenMeteoSource("ecmwf_ifs025", tile_deg=1.0)
```

A cold-cache comparison on NSZ4280 then GAF287 (2026-04-15, 10,752 points)
produced the following totals. Each size used a fresh source without a disk
cache; the two flights shared its memory cache. Times are single-run network
measurements, not stable performance guarantees. Download counters exclude
source construction; memory refers only to decoded tiles, not total process
memory or the HTTP page cache.

| Tile edge | Download (MB) | HTTP requests | Decoded tiles (MB) | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| 5° | 191.1 | 429 | 66.4 | 17.2 |
| 2° | 193.1 | 433 | 56.2 | 17.1 |
| 1° | 179.0 | 406 | 45.2 | 18.9 |
| 0.5° | 169.0 | 387 | 39.9 | 36.2 |

All four sizes produced the same flags and `u/v/t` values to an absolute
tolerance of `1e-10`. GAF287 required no additional downloads at any size.
One-degree tiles saved about 6% of downloaded bytes and 32% of decoded tile
memory on these flights. Half-degree tiles doubled elapsed time despite
reading fewer bytes. Tile alignment, compressed blocks and 512 KiB HTTP
pages mean that reducing tile size does not guarantee fewer downloaded
bytes (as the 2-degree result illustrates).


A follow-up compared larger tiles using three cold-cache trials per size,
in the order 5, 7.5, 10, 10, 7.5, 5, 7.5, 5, 10 degrees. Inputs, cache
settings and correctness checks were the same as above. The table gives
median total enrichment time and the observed minimum–maximum; download
and decoded tile memory were identical across the three trials of each size.

| Tile edge | Download (MB) | Decoded tiles (MB) | Median time (s) | Time range (s) |
| --- | ---: | ---: | ---: | ---: |
| 5° | 191.1 | 66.4 | 15.75 | 15.62–16.85 |
| 7.5° | 250.3 | 164.0 | 17.30 | 17.10–18.43 |
| 10° | 223.6 | 145.8 | 16.15 | 15.47–16.21 |

Larger tiles did not demonstrate a speed advantage on these flights.
Ten-degree tiles downloaded 17% more bytes and more than doubled decoded
tile memory. Larger tiles also merged some trajectory regions: NSZ4280
required three larger cubes instead of five smaller cubes. Fewer cubes
therefore did not imply less data or faster enrichment. The generic
5-degree default remains unchanged; these two flights do not establish
an optimal tile size for every workload.
