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

Files are chunked as 1 latitude row × 14 longitudes × all hours of the file:
reading one hour costs a whole file block, so tiles are read for all hours
and kept in memory. Requests are served from aligned 512 KiB pages, merging
the many tiny reads of the decoder. Two layouts coexist (flattened before
early 2025, `(lat, lon, time)` after); both are read.
