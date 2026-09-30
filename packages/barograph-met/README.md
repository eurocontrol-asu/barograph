<p align="center">
  <img src="https://raw.githubusercontent.com/eurocontrol-asu/barograph/main/assets/logo.svg" alt="barograph" width="100" />
</p>

<p align="center">
  <strong>barograph-met — wind and temperature at aircraft positions, fast and explicit</strong>
</p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/barograph-met/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/barograph-met/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/barograph-met/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

`barograph-met` interpolates wind (`u`, `v`) and temperature (`t`) at
aircraft positions (time, latitude, longitude, barometric altitude).

```python
from baro.met import Enricher, arco_model_levels, track_components

res = Enricher(arco_model_levels(), cache_dir="~/data/cubes").enrich(
    time_utc, lat, lon, altitude_ft=baro_alt_ft
)
res["u"], res["v"], res["t"], res.flags, res.stats
along, cross = track_components(res["u"], res["v"], track_deg)
```

## Features

- **Sources**: ERA5 from ARCO-ERA5 on its 137 model levels
  (`arco_model_levels`) or pressure levels (`arco_pressure_levels`);
  Météo-France ARPEGE open data, 0.1° Europe or 0.25° global
  (`ArpegeSource`); the Open-Meteo archive of ARPEGE from December 2023
  (`OpenMeteoSource`); any of them chained (`SourceChain`).
- **Only the bytes a flight needs**: partial reads of the compressed blocks of
  ARCO-ERA5 chunks (levels × latitude band), of GRIB2 messages (variables ×
  levels × steps), and of Open-Meteo tiles; every time slice is read once for
  all flights.
- **Exact vertical coordinate**: level pressures from each grid column's
  actual surface pressure, interpolation in ln p.
- **Explicit**: no silent extrapolation — every missing value has a flag, and
  points outside a source's period are reported.
- **Campaign-ready**: immutable cube cache shared by processes, per-host
  concurrency caps, retry with backoff.
- **FastMeteo-compatible façade**: `baro.met.compat.ArcoEra5` and
  `Arpege`, same constructors and `interpolate(df)`.

## Installation

```bash
pip install barograph-met                              # ERA5
pip install 'barograph-met[grib,pandas,openmeteo]'     # + ARPEGE, FastMeteo façade, archive
```

## Benchmarks

Same points for every tool, cold cache (first run), through a corporate HTTPS
proxy. Gains are reference ÷ barograph-met.

| Case | Reference | Time | Downloaded | Peak RAM | Values vs reference |
|---|---|---:|---:|---:|---|
| ERA5, 22 pressure levels | FastMeteo 1.2.0 | 214 s → **12 s** (×18) | 1 296 → **100 MB** (×13) | 1.5 → **0.16 GB** | identical (≤ 1e-13), except FastMeteo's extrapolated points at the Greenwich seam |
| ERA5, 137 model levels | pycontrails 0.63.5 | 322 s → **31 s** (×10) | 4 736 → **349 MB** (×14) | 7.0 → **0.22 GB** | RMS 0.03 m/s (u, v), 0.016 K (T) |
| ARPEGE 0.1°, official (14 days) | FastMeteo 1.2.0 | 67 s → **13 s** (×5) | 475 → **54 MB** (×9) | — | identical east of Greenwich; FastMeteo extrapolates west of it |

Warm runs (second call, tool's own cache) take 0.2–2 s for every tool.
ERA5 cases: one 2 h flight from the surface to FL350 (1 764 points, 4 hourly
analyses). ARPEGE case: a 3 h flight across France (400 points). Network time
is over 98 % of the cold time: the gain comes from reading only the levels and
latitude bands the flight crosses instead of whole global fields.

| Coverage | FastMeteo | pycontrails | barograph-met |
|---|:---:|:---:|:---:|
| ERA5 pressure levels | 22 of 37 | 37 | 37 |
| ERA5 model levels (L137) | — | via 61 PL | direct, actual surface pressure |
| ARPEGE official (last 14 days) | ✓ | — | ✓ |
| ARPEGE archive (since Dec 2023, Open-Meteo) | — | — | ✓, chained with the official feed |
| Points outside the data | extrapolated | NaN | flagged |

Details in the [benchmark notes](https://github.com/eurocontrol-asu/barograph/blob/main/packages/barograph-met/docs/explanation/benchmarks.md).

## FastMeteo

This package builds on [FastMeteo](https://github.com/open-aviation/fastmeteo)
(Sun & Roosenbrand, JOAS 2023, doi:10.59490/joas.2023.7264). Fixes found while
comparing the two tools were contributed upstream
([open-aviation/fastmeteo#5](https://github.com/open-aviation/fastmeteo/pull/5)),
and this engine is intended to become available to FastMeteo as an optional
backend. ERA5 data: Copernicus Climate Change Service (Hersbach et al. 2020,
doi:10.1002/qj.3803), via ARCO-ERA5 (Carver & Merose, 2023).

## Development

```bash
uv sync --all-packages --all-groups
uv run --package barograph-met --directory packages/barograph-met pytest   # offline suite
uv run --package barograph-met --directory packages/barograph-met pytest -m live   # public buckets
make lint
```

## License

[EUPL-1.2](LICENSE).
