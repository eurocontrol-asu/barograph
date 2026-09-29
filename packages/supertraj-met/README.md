<p align="center">
  <img src="https://raw.githubusercontent.com/asu-eurocontrol/supertraj/main/assets/logo.svg" alt="supertraj" width="100" />
</p>

<p align="center">
  <strong>supertraj-met — wind and temperature at aircraft positions, fast and explicit</strong>
</p>

<p align="center">
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/ci.yml"><img src="https://github.com/asu-eurocontrol/supertraj/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/supertraj-met/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/supertraj-met/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/supertraj-met/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

`supertraj-met` interpolates wind (`u`, `v`) and temperature (`t`) at
aircraft positions (time, latitude, longitude, barometric altitude).

```python
from supertraj_met import Enricher, arco_model_levels, track_components

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
- **FastMeteo-compatible façade**: `supertraj_met.compat.ArcoEra5` and
  `Arpege`, same constructors and `interpolate(df)`.

## Installation

```bash
pip install supertraj-met                              # ERA5
pip install 'supertraj-met[grib,pandas,openmeteo]'     # + ARPEGE, FastMeteo façade, archive
```

## Benchmarks

Same points for every tool, cold cache, through a corporate HTTPS proxy.

| Comparison | Cold | Downloaded | Peak RAM | Values |
|---|---:|---:|---:|---|
| ERA5 PL22 — FastMeteo 1.2.0 | 214 s | 1296 MB | 1.5 GB | reference |
| ERA5 PL22 — supertraj-met | **12 s** | **100 MB** | 0.16 GB | identical (≤ 1e-13) except FastMeteo's extrapolated points at the Greenwich seam |
| ERA5 ML — pycontrails 0.63.5 | 322 s | 4736 MB | 7.0 GB | reference |
| ERA5 ML137 — supertraj-met | **31 s** | **349 MB** | 0.22 GB | RMS 0.03 m/s (u, v), 0.016 K (T) |
| ARPEGE 0.1° — FastMeteo 1.2.0 | 67 s | 475 MB | — | reference |
| ARPEGE 0.1° — supertraj-met | **13 s** | **54 MB** | — | identical east of Greenwich; FastMeteo extrapolates west of it |

Details in the [documentation](https://asu-eurocontrol.github.io/supertraj/met/explanation/benchmarks/).

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
uv run --package supertraj-met --directory packages/supertraj-met pytest   # offline suite
uv run --package supertraj-met --directory packages/supertraj-met pytest -m live   # public buckets
make lint
```

## License

[EUPL-1.2](LICENSE).
