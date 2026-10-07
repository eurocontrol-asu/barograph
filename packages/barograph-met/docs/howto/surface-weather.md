# Query surface weather from an archived forecast

Surface weather is queried by UTC time, latitude and longitude. It uses a
separate interface from pressure-level fields: no altitude or pressure is
required. Fixed coordinates broadcast over a sequence of query times.

```python
import numpy as np
from baro.met import OpenMeteoSurfaceSource, SurfaceEnricher

when = np.array(["2026-04-15T14:00", "2026-04-15T14:30"], dtype="datetime64[ns]")

with OpenMeteoSurfaceSource("meteofrance_arome_france0025") as source:
    query = SurfaceEnricher(source, cache_dir="/data/weather-cache")
    weather = query.enrich(when, lat=48.8566, lon=2.3522)
    wind = weather["wind_speed"]
    rain = weather["rain"]
    rain_start = weather.interval_start["rain"]
    rain_end = weather.interval_end["rain"]
```

## Variables and units

| Variable | Unit | Height above model terrain | Native meaning |
| --- | --- | --- | --- |
| `u10`, `v10` | m/s | 10 m | Instantaneous eastward/northward wind |
| `t2m` | K | 2 m | Instantaneous temperature |
| `precipitation` | mm water equivalent | Surface | Total precipitation over the preceding native period |
| `rain` | mm | Surface | Precipitation minus snowfall water equivalent, clipped at zero |
| `gusts` | m/s | 10 m | Maximum gust over the preceding native period |
| `snowfall_water_equivalent` | mm water equivalent | Surface | Snow over the preceding native period; optional |
| `wind_speed` | m/s | 10 m | Speed derived from interpolated u/v |
| `wind_direction` | degrees | 10 m | Meteorological direction **from** which wind blows |

The default fields are `u10`, `v10`, `t2m`, `precipitation`, `rain` and `gusts`.
Speed and direction are computed after interpolating u/v, avoiding angular
wrap errors. At exactly zero wind speed, direction is NaN and the point remains
valid. The native wind is an instantaneous model value, not an hourly mean.
There is no correction for differences between local terrain and model terrain.

The raw archive has no rain folder. Requesting rain reads total precipitation
and snowfall water equivalent, then derives liquid precipitation at the native
grid points. Snow water equivalent is distinct from snowfall depth.
Open-Meteo has already deaccumulated the amounts before storing them; the
reader does not difference consecutive archive samples again.

## Time and space interpolation

Instant fields use bilinear spatial interpolation and linear time
interpolation between native timestamps. Period totals and maxima use
bilinear spatial interpolation and retain the full native period. For an
hourly archive, the value stamped 15:00 represents **(14:00, 15:00]**:

- A query at exactly 14:00 uses the period ending at 14:00.
- A query at 14:30 uses the period ending at 15:00.
- A query at exactly 15:00 also uses the period ending at 15:00.

`interval_start` and `interval_end` identify this support for every field and
point. Instant fields have both equal to the query time. A fractional query
returns neither a prorated rain amount nor a sub-hour maximum. Adding repeated
fractional queries would double-count the same native precipitation period.
For a source with a three-hour native step, the period is three hours rather
than one hour. Interpolating the gust grid does not guarantee the maximum
experienced at a particular location.

`flags` describes invalid inputs, horizontal coverage, temporal coverage and
missing data. As with `EnrichResult`, flags are aggregated across the requested
fields; a missing required field makes all outputs at that point NaN. The
first archived chunk may be partly empty. Queries before the first native
endpoint are flagged, even if a period stamped at that endpoint extends earlier.
The final native endpoint is available without reading a future sample.

## Read only the requested fields

```python
with OpenMeteoSurfaceSource(
    "meteofrance_arome_france0025",
    fields={"u10": "u10", "v10": "v10", "gusts": "gusts"},
    tile_deg=0.25,
) as source:
    weather = SurfaceEnricher(source).enrich(when, 48.8566, 2.3522)
```

Field mappings follow the existing pressure-source convention: output name to
variable name. If u/v outputs are renamed, pass `wind_components=("u", "v")`
to `SurfaceEnricher` to derive wind speed and direction from those names.
Set `wind_components=None` to omit derived wind outputs.

The same native OM byte-range reader, memory LRU and persistent page cache are
used for pressure and surface fields. Surface tiles default to 0.25 degrees,
with adjacent occupied tiles kept separate (`merge_adjacent_tiles=False`) to
avoid reading large empty rectangles between query points. Queries are grouped by native time and
occupied geographic tiles. Only selected fields and their dependencies are
read: at the left endpoint of a fractional query, only instant fields are
needed. OM compression imposes a minimum read block; a geographic request
cannot always download exactly one grid point or one hour. Repeated queries
reuse those blocks. Recent stitched chunks are periodically revalidated;
older chunks become immutable after the same stitching window as pressure
fields. No entire regional model grid is downloaded for a point query.

## Forecast provenance

This source reads a **stitched forecast archive**, not an ERA5 reanalysis.
It retains valid times and discards the original run identities and lead
times. Consequently `weather.run` and `weather.lead` are NaT. It cannot establish
which forecast was available before a historical cutoff. Native forecast
sources implementing `SurfaceSource` can retain `CubeKey.run`; the same
`SurfaceEnricher` then returns run and query lead for each point and refuses
to interpolate endpoints from different runs.

The supported time range comes from the current archive metadata; recently
available future valid times can be queried within that range. The reader
cannot recover older forecasts issued for the same valid time.

The surface definitions and rain derivation follow the
[official Open-Meteo Météo-France documentation](https://open-meteo.com/en/docs/meteofrance-api).
Credit Open-Meteo and Météo-France under CC-BY 4.0 when using the archive.

A small live comparison at one location (Paris, three queries on 15 April 2026,
all default fields, empty caches) gave the following results. Values were
identical for all tile sizes; a repeated query downloaded zero additional bytes.
These are single measurements, not a general latency guarantee.

| Tile edge | Query bytes downloaded | Decoded tiles in memory | Initial elapsed time, including catalog |
| --- | ---: | ---: | ---: |
| 0.25° | 7.05 MB | 0.26 MB | 31.9 s |
| 1° | 14.91 MB | 4.15 MB | 34.2 s |
| 5° | 56.33 MB | 103.68 MB | 83.2 s |

The surface default therefore uses 0.25°. Campaigns with dense spatial sampling
may benefit from larger explicitly configured tiles.
