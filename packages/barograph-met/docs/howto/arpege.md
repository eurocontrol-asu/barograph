# ARPEGE, live and archive

Météo-France publishes ARPEGE as open data for about **14 days**; older runs
are available from **Open-Meteo's archive** (stitched from the most recent run
at each hour, values stored as scaled integers: we measured 0.035 m/s and
0.08 K RMS against the official files).

## Live open data

```python
from baro.met import ArpegeSource, Enricher

src = ArpegeSource(resolution="01", index_dir="~/data/grib-index")   # 0.1°, Europe
res = Enricher(src).enrich(time_utc, lat, lon, altitude_ft=alt_ft)
```

- `resolution="025"` is the global 0.25° grid.
- `run="latest"` (default) takes the most recent published run (00/06/12/18
  UTC) not after each interval, falling back to older runs; `run="daily"`
  uses the 00 UTC run of the day (FastMeteo's behaviour); a date fixes the run.
- Files are GRIB2 without an inventory: an index of message offsets is built
  once per file (a few seconds) and cached in `index_dir`. Only the messages
  of the needed variables, levels and steps are then downloaded.
- Decoding needs ecCodes: `pip install 'barograph-met[grib]'`.

## Archive (from December 2023)

```python
from baro.met import OpenMeteoSource

with OpenMeteoSource("meteofrance_arpege_europe") as archive:   # 0.1°, from 2023-12-01
    res = Enricher(archive).enrich(time_utc, lat, lon, altitude_ft=alt_ft)
```

`"meteofrance_arpege_world025"` is the global grid (from 2023-12-22). Needs
`pip install 'barograph-met[openmeteo]'`.

## Both, with a seamless hand-over

```python
from baro.met import SourceChain

arpege = SourceChain(ArpegeSource(resolution="01"), OpenMeteoSource("meteofrance_arpege_europe"))
```

For each time interval the first source that serves it is used; both cubes
bracketing a point always come from the same source. On a flight covered by
both, the two differ by 0.035 m/s (wind) and 0.08 K (temperature) RMS — the
archive's quantisation. Points before the archive starts are flagged
`OUT_OF_COVERAGE` and reported in a warning.

**Other variables**: `ArpegeSource(fields={"u": "u", "v": "v", "t": "t", "r": "r"})`
selects among the names listed in [Variables by source](../reference/variables.md);
the Open-Meteo archive takes the same kind of mapping. An unknown name raises
`ValueError`.
