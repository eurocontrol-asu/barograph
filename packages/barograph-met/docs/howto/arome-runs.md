# AROME surface forecasts by initialization

Use the native public Météo-France files when forecast initialization and lead
must be retained. The rolling server retains approximately fourteen days;
older runs require a local mirror of the same file layout. It provides the
0.025° AROME grid and hourly forecast leads from 0 to 51 hours.

```python
import numpy as np
from baro.met.arome_surface import AromeSurfaceSource
from baro.met.surface import SurfaceEnricher

valid_time = np.datetime64("now", "h")  # current UTC hour
source = AromeSurfaceSource.latest(valid_time)
result = SurfaceEnricher(source, cache_dir="/tmp/weather-cache").enrich(
    np.array([valid_time]),
    lat=[48.85], lon=[2.35],
)
print(result.fields, result.run, result.lead)
source.close()
```

The example selects a run that is available now for the current UTC hour.
For a reproducible historical run, pass an explicit `run=np.datetime64(...)`
to `AromeSurfaceSource` and use a local mirror if the run has left the rolling
server's retention window.

The canonical fields are `u10`, `v10` (m/s), `t2m` (K), `precipitation` (mm)
and `gusts` (m/s). Map output names with `fields={"temperature": "t2m"}`
to retrieve only selected variables. Heights are above the model ground;
queries require no altitude. `SurfaceEnricher` also returns `wind_speed` and
`wind_direction`, calculated from the interpolated horizontal components.
Direction is the meteorological wind-from direction, in degrees clockwise
from true north; it is undefined (NaN) for exactly calm wind.

Precipitation is the total water equivalent during the preceding hour, obtained
by subtracting cumulative values from the **same initialization**, including
when the previous lead occupies another GRIB file. Gusts retain the native
preceding-hour maximum. Instantaneous fields interpolate in time; period fields
select the full native interval containing the query. At initialization itself,
preceding-hour precipitation and gusts are unavailable. The final native lead
is queryable without requiring the next forecast hour.

`AromeSurfaceSource.latest(valid_time)` selects the newest run whose necessary
files can be found **now**. Its `observed_at` records when that check happened.
Initialization is distinct from publication: this method cannot reconstruct the
last forecast available at a historical cutoff. Use an explicit run when an
external publication record establishes that choice. A stitched Open-Meteo
archive cannot supply initialization or lead provenance.

The reader scans GRIB headers once to discover message offsets, then downloads
only selected field messages. Those full-grid compressed messages are cropped
after decoding. With a range cache, subsequent reads reuse immutable message
bytes and saved indexes. This is selective by variable and forecast step;
GRIB messages do not permit the regional byte selection of the Open-Meteo
archive's tiled format.

See the [official AROME specification](https://donneespubliques.meteofrance.fr/client/document/doc_arome_pour-portail_20250806_410.pdf)
for the grid, hourly horizon and run publication schedule, and the
[Open-Meteo downloader](https://github.com/open-meteo/open-meteo/blob/main/Sources/App/MeteoFrance/MeteoFranceDownloader.swift)
for native package naming and variable mapping.
