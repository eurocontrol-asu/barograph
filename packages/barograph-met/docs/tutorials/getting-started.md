# Getting started

This tutorial enriches a short synthetic flight with ERA5 wind and
temperature on the 137 model levels.

## Install

```bash
pip install barograph-met            # ERA5 only
pip install 'barograph-met[grib,pandas,openmeteo]'   # ARPEGE, FastMeteo façade, Open-Meteo archive
```

## A first flight

```python
import numpy as np
from baro.met import Enricher, arco_model_levels, track_components

n = 60
time_utc = np.datetime64("2020-01-01T10:00") + np.arange(n) * np.timedelta64(30, "s")
lat = np.linspace(48.7, 47.5, n)
lon = np.linspace(2.4, 3.2, n)
altitude_ft = np.full(n, 35_000.0)          # barometric (pressure) altitude
track_deg = np.full(n, 150.0)

enricher = Enricher(arco_model_levels(), cache_dir="~/data/cubes")
res = enricher.enrich(time_utc, lat, lon, altitude_ft=altitude_ft)

print(res["t"][:3], res.flags[:3])          # temperature (K) and flags (0 = OK)
along, cross = track_components(res["u"], res["v"], track_deg)
print(res.stats)                            # bytes, requests, cubes, cache hits, seconds
```

What happened:

1. The points were grouped by hour; the two hourly analyses bracketing each
   point were read **once** for all points.
2. Only the compressed blocks covering the needed levels and latitude band
   were downloaded (a few MB instead of ~1.2 GB per hour).
3. Each value was interpolated between model levels using the column's own
   surface pressure, then in latitude/longitude and in time.
4. The cubes were written to `~/data/cubes`: running the script again
   downloads nothing (`res.stats["bytes"] == 0`).

## Reading the flags

A value is `NaN` whenever `res.flags` is non-zero; the flag says why (outside
the grid, above the top level, not published, before the archive starts...).
See [Flags and statistics](../reference/flags.md).
