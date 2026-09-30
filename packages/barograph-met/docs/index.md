# barograph-met

Wind (`u`, `v`) and temperature (`t`) at aircraft positions — time, latitude,
longitude and barometric altitude — from:

| Source | Class / function | Period | Vertical |
|---|---|---|---|
| ERA5, ARCO-ERA5 on Google Cloud | `arco_model_levels()` | 1940 → ~3 months ago | 137 model levels, actual surface pressure |
| ERA5, ARCO-ERA5 | `arco_pressure_levels(levels=...)` | 1940 → ~6 days ago | 37 pressure levels or a subset |
| Météo-France ARPEGE open data | `ArpegeSource(resolution="01" \| "025")` | last ~14 days (+ forecast) | 24 / 34 pressure levels |
| ARPEGE, Open-Meteo archive | `OpenMeteoSource("meteofrance_arpege_europe")` | from 2023-12-01 | 24 / 34 pressure levels |
| Any of the above, chained | `SourceChain(a, b, ...)` | union | — |

```python
from baro.met import Enricher, arco_model_levels

res = Enricher(arco_model_levels(), cache_dir="~/data/era5-cache").enrich(
    time_utc, lat, lon, altitude_ft=baro_alt_ft
)
res["u"], res["v"], res["t"], res.flags, res.stats
```

Start with the [tutorial](tutorials/getting-started.md), or jump to a
[how-to guide](howto/index.md).

## Why another tool

[FastMeteo](https://github.com/open-aviation/fastmeteo) pioneered lightweight
ERA5/ARPEGE enrichment of ADS-B trajectories, and this package builds on it.
The ARCO-ERA5 stores keep a *global* field in every chunk, so tools that read
whole chunks transfer the whole globe for any flight. `barograph-met` reads
only the compressed blocks covering the needed levels and latitude band,
interpolates on ERA5's native model levels with each column's surface
pressure, and never extrapolates silently. See the
[benchmarks](explanation/benchmarks.md).

Fixes found while comparing the two tools (Météo-France's new ARPEGE host,
longitudes around Greenwich) were contributed to FastMeteo
([open-aviation/fastmeteo#5](https://github.com/open-aviation/fastmeteo/pull/5)),
and this engine is intended to become available to FastMeteo as an optional
backend.
