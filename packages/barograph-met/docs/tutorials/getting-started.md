# Getting started

This tutorial enriches a short synthetic flight with ERA5 wind and
temperature on the 137 model levels.

## Install

Requires Python 3.12+ on a POSIX system (Linux or macOS). Native Windows is
not supported because the shared disk cache uses POSIX file locks.

Create an environment and install from PyPI with **uv**:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install barograph-met
```

Or use **pip** in your Python environment:

```bash
python -m pip install barograph-met
```

For ARPEGE, the FastMeteo DataFrame façade and Open-Meteo archives, install
`'barograph-met[grib,pandas,openmeteo]'` with either installer. In an existing
uv-managed project, `uv add barograph-met` records the dependency in your project.

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

enricher = Enricher(arco_model_levels(), cache_dir="~/data/era5-cache")
res = enricher.enrich(time_utc, lat, lon, altitude_ft=altitude_ft)

print(res["t"][:3], res.flags[:3])          # temperature (K) and flags (0 = OK)
along, cross = track_components(res["u"], res["v"], track_deg)
print(res.stats)                            # bytes, cache_bytes, requests, cubes, seconds
```

What happened:

1. The points were grouped by hour; the two hourly analyses bracketing each
   point were read **once** for all points.
2. Only the compressed blocks covering the needed levels and latitude band
   were downloaded (a few MB instead of ~1.2 GB per hour).
3. Each value was interpolated between model levels using the column's own
   surface pressure, then in latitude/longitude and in time.
4. The bytes read were kept in `~/data/era5-cache`: running the script again,
   or any flight over the same area and hours, downloads nothing
   (`res.stats["bytes"] == 0`) — up to the metadata of the store, and except
   for the last ~3 months (preliminary ERA5T), which are never cached.

## Reading the flags

A value is `NaN` whenever `res.flags` is non-zero; the flag says why (outside
the grid, above the top level, not published, before the archive starts...).
See [Flags and statistics](../reference/flags.md).
