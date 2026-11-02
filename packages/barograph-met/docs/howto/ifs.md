# IFS pressure-level wind and temperature

Use `OpenMeteoSource("ecmwf_ifs025")` to interpolate IFS wind and temperature
at aircraft positions from Open-Meteo's public archive. From the repository
root, install with `pip install './packages/barograph-met[openmeteo]'`;
no API key is required.

```python
import numpy as np
from baro.met import Enricher, OpenMeteoSource

with OpenMeteoSource(
    "ecmwf_ifs025",
    fields={"u": "u", "v": "v", "t": "t"},
    tile_deg=1.0,
    merge_adjacent_tiles=False,
) as source:
    result = Enricher(source, cache_dir="weather-cache").enrich(
        [np.datetime64("2026-04-15T12:30")],
        lat=[48.85],
        lon=[2.35],
        altitude_ft=[35000.0],
    )
    u, v, temperature = result["u"], result["v"], result["t"]
    flags = result.flags
```

Altitude is pressure altitude in feet. Wind components are in m/s and
temperature in kelvin. Interpolation uses the surrounding horizontal grid
points, pressure levels and native times. Missing fields, levels, dates or
positions outside coverage produce explicit flags and NaNs.

IFS has a native three-hour time step. The reader discovers cadence and
coverage from metadata; it does not assume hourly samples. For example,
`chunk_time_length=104` counts 104 samples, representing a 312-hour file at
that cadence. ARPEGE and AROME retain their hourly cadence through the same
generic reader. See [time indexing and the archive layout](../explanation/data-access.md#open-meteo-archive-om).

This archive combines forecasts at each valid time. Original initialization
times and leads are unavailable, so it cannot tell whether a value was
predicted 6, 24 or 48 hours in advance. It is not ERA5 reanalysis. Pressure
levels and populated dates depend on the published archive; metadata coverage
does not guarantee every variable and level exists at every date.

Reads select variables, bracketing pressure levels and geographic tiles with
the configured margins. Compressed storage blocks also contain other grid
points and timestamps; they determine the minimum acquisition granularity.
`tile_deg` controls the reader's grouping, not IFS model resolution; smaller
tiles can reduce empty area and decoded memory while increasing request
overhead. The shared byte cache reuses downloaded ranges. See the
[measured tile-size tradeoffs](../explanation/data-access.md#choosing-tile-size).
