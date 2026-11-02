# Query AROME-PI and PIAF forecasts

`AromePISource` and `PiafSource` retrieve geographically targeted surface
forecasts through Météo-France's authenticated WCS APIs. Install the GRIB
backend (`pip install './packages/barograph-met[grib]'` from the repository
root) and subscribe to each API in the
[Météo-France portal](https://portail-api.meteofrance.fr/).

| Source | Native grid | Output interval | Initializations | Maximum lead |
|---|---|---|---|---|
| AROME-PI | 0.025° or 0.01° | 15 minutes | Every hour | 6 hours |
| PIAF | 0.01° | 5 minutes | Every 5 minutes | 195 minutes |

These are forecasts, not reanalyses. The APIs expose about five days of runs.
The date of initialization is preserved alongside the forecast lead. A run's
initialization time is **not** its publication time.

## Credentials and fixed runs

Keep credentials outside the repository. The following example reads an API
key from a private local file; its contents are sent only in the `apikey`
header, never in URLs or cache keys. Paths are application configuration,
not hardcoded by the library.

For example, create separate files for the API keys authorized for AROME-PI
and PIAF. In Bash on Linux, this prompts without echoing either key or putting
its value into shell history:

```bash
install -d -m 700 "$HOME/.config/barograph"
(
    umask 077
    for api in arome-pi piaf; do
        key_file="$HOME/.config/barograph/$api.token"
        touch "$key_file"
        chmod 600 "$key_file"
        read -r -s -p "API key for $api: " barograph_key
        printf '\n'
        printf '%s\n' "$barograph_key" > "$key_file"
        unset barograph_key
    done
)
```

The files are `~/.config/barograph/arome-pi.token` and
`~/.config/barograph/piaf.token`, readable only by their owner. Keep them out of
Git and do not paste their values into examples or issue reports. The library
does not automatically search these paths or load a `.env` file: pass a token
or a callable that reads your chosen file, as below. Public Open-Meteo archives
and the native public AROME GRIB source do not require these API keys.

```python
from pathlib import Path
import numpy as np
from baro.met import AromePISource, SurfaceEnricher

key_file = Path("~/.config/barograph/arome-pi.token").expanduser()
run = np.datetime64("now", "h") - np.timedelta64(1, "h")

with AromePISource(
    run=run,
    token=lambda: key_file.read_text().strip(),
    resolution="0.025",
) as source:
    result = SurfaceEnricher(source, cache_dir="~/.cache/barograph/met").enrich(
        time=[run + np.timedelta64(30, "m")],
        lat=[48.8566],
        lon=[2.3522],
    )
    result["wind_speed"], result["wind_direction"]
    result["gusts"], result["precipitation"], result["t2m"]
    result.run, result.lead, result.flags
```

Choose a published run for reproducible results; the example's preceding-hour
run is a convenience, not a guarantee of publication. API keys use
`auth="apikey"` by default. For an OAuth access token use `auth="bearer"`.
A callable token supplier can handle renewal outside the library. HTTP 401/403
raises an explicit authentication error, rather than returning missing weather.

## Fields and resolution

On the 0.025° AROME-PI grid, the default fields are `u10`, `v10`, `t2m`,
`precipitation`, and `gusts`. The 0.01° grid defaults to `t2m`, `precipitation`,
and `gusts`: mean wind components are not published on that finer grid.
`rain` (liquid precipitation) is optional on either grid. All fields are
validated against the returned GRIB parameter, level, units, run, and interval.

Use `fields={"temperature": "t2m"}` to request only temperature, or any other
mapping of output names to the canonical fields. Wind speed and meteorological
wind-from direction are derived after interpolating `u10`/`v10`. If you alias
these components, set `SurfaceEnricher(..., wind_components=("east", "north"))`
to the chosen output names.

PIAF provides deterministic precipitation amounts:

```python
from baro.met import PiafSource

key_file = Path("~/.config/barograph/piaf.token").expanduser()
valid_time = np.datetime64("now", "s") + np.timedelta64(15, "m")

with PiafSource.latest(
    valid_time, token=lambda: key_file.read_text().strip()
) as source:
    result = SurfaceEnricher(source, cache_dir="~/.cache/barograph/met").enrich(
        [valid_time], [48.8566], [2.3522]
    )
    result["precipitation"]  # mm over the enclosing native five-minute interval
    result.interval_start["precipitation"], result.interval_end["precipitation"]
```

The PIAF WCS coverage name contains `TOTAL_PRECIPITATION_RATE`, but its GRIB
product is an accumulated amount in kg/m², equivalent to mm. It must not be
multiplied by the interval duration as though it were a rate. The reader
rejects incompatible statistical products rather than guessing their meaning.

## Time semantics and run selection

Instantaneous fields are interpolated between adjacent native times in the
same run. Precipitation retains the full amount over `(start, end]`; gusts
retain the maximum over that interval. At a fractional query time, these
period fields describe the enclosing interval, including its future portion.
They are not interpolated or prorated. At an exact boundary, the preceding
interval is returned. No period exists at lead zero.

`latest(valid_time, ...)` chooses the newest catalogued run with the necessary
field endpoints and heights. `source.observed_at` records when that selection
was made. This checks metadata observed **now**, not which forecast could have
been downloaded at a historical cutoff. A missing download never silently
switches the selected run. `published_runs(refresh=True)` refreshes the catalog.

The public APIs' short retention is distinct from an archive service: saving
requested responses in the cache does not create a complete historical archive.
Météo-France's PIAF documentation also mentions a three-year rolling archive
via BDAP, whose access is separate and is not implemented here.

## Download scope and caching

The reader requests one parameter, level, run, valid time and geographic box
per field. Bounds are snapped to the native grid centres required for bilinear
interpolation (at least 2×2). Exact timestamps avoid the unused next endpoint;
fractional timestamps fetch the left endpoint only for instantaneous fields.
Native interval products are preferred. A cumulative precipitation response
requires only its preceding same-run counter, which is differenced explicitly.

`tile_deg=0.25` groups nearby query points without merging an entire connected
trajectory into a large box. It controls request grouping, **not** model
resolution. Smaller values can reduce empty area while increasing request
count. This differs from Open-Meteo's storage tiles: WCS supplies a server-side
geographic subset, so an isolated site needs only its interpolation neighbours.

Validated GRIB responses use the shared `RangeCache`. Repeating identical
queries with the same source and cache causes zero network transfers. A new
source discovers the current catalog and field metadata again; these can be
larger than the requested weather data. Cached field responses are reusable
across source instances, processes and token renewal. Overlapping but different
geographic subsets have different cache keys.
The cache is not an offline archive: a newly created source still needs the
run's catalog entry and metadata, so expired runs cannot be reopened through
these APIs merely because their GRIB responses remain on disk.

## Provider documentation

- [WCS API operations and retention](https://confluence-meteofrance.atlassian.net/wiki/spaces/OpenDataMeteoFrance/pages/854032416)
- [Model descriptions and grid availability](https://confluence-meteofrance.atlassian.net/wiki/spaces/OpenDataMeteoFrance/pages/621019138)
- [AROME-PI API](https://portail-api.meteofrance.fr/web/fr/api/AROME-PI)
- [PIAF API](https://portail-api.meteofrance.fr/web/fr/api/PrevisionImmediatePrecipitations)
