# barograph-met

Wind (`u`, `v`) and temperature (`t`) at aircraft positions — time, latitude,
longitude and barometric altitude — from:

| Source | Class / function | Period | Vertical |
|---|---|---|---|
| ERA5, ARCO-ERA5 on Google Cloud | `arco_model_levels()` | 1940 → ~3 months ago | 137 model levels, actual surface pressure |
| ERA5, ARCO-ERA5 | `arco_pressure_levels(levels=...)` | 1940 → ~6 days ago | 37 pressure levels or a subset |
| Météo-France ARPEGE open data | `ArpegeSource(resolution="01" \| "025")` | last ~14 days (+ forecast) | 24 / 34 pressure levels |
| ARPEGE, Open-Meteo archive | `OpenMeteoSource("meteofrance_arpege_europe")` | from 2023-12-01 | 24 / 34 pressure levels |
| AROME, Open-Meteo archive | `OpenMeteoSource("meteofrance_arome_france0025")` | from late December 2023 | 24 pressure levels |
| IFS, Open-Meteo archive | `OpenMeteoSource("ecmwf_ifs025")` | from January 2024 | available pressure levels, 3-hourly |
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
The [IFS guide](howto/ifs.md) explains its three-hour archive and forecast
provenance, and [AROME-PI/PIAF access](howto/immediate-forecasts.md) shows how
to keep Météo-France API keys in private files outside the repository.

For surface weather, `SurfaceEnricher` accepts time, latitude and longitude
without altitude. `OpenMeteoSurfaceSource` reads archived 10 m wind, gusts,
precipitation and 2 m temperature; `AromeSurfaceSource` retains the initialization
and lead time of an explicit AROME forecast run. Stitched archives do not
retain that provenance. Period totals and maxima keep their native intervals
rather than being linearly interpolated in time.

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

## Data sources and licences

barograph-met redistributes no data: it reads the providers' public stores on
your behalf, and what you download stays under their terms. Credit them in
work that uses the values:

| Source | Licence and attribution |
|---|---|
| ERA5 (Copernicus Climate Change Service, ECMWF) | Copernicus licence: *"Contains modified Copernicus Climate Change Service information [year]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."* Cite Hersbach et al. (2020), [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803). |
| ARCO-ERA5 (Google Research, Google Cloud public datasets) | Same ERA5 terms; cite Carver & Merose (2023), [ARCO-ERA5](https://github.com/google-research/arco-era5). |
| Météo-France ARPEGE open data | [Licence Ouverte / Open Licence 2.0 (Etalab)](https://www.etalab.gouv.fr/licence-ouverte-open-licence/): credit *"Source: Météo-France"*. |
| Open-Meteo archive | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): credit *"Weather data by Open-Meteo.com"* and the model's producer (Météo-France for ARPEGE). |
| OpenSky Network (benchmark flights only) | Cite Schäfer et al. (2014), *Bringing up OpenSky*, IPSN; see the [OpenSky terms of use](https://opensky-network.org/about/terms-of-use). |
