<p align="center">
  <img src="assets/logo.svg" alt="barograph" width="120" />
</p>

<h1 align="center">barograph</h1>
<p align="center"><strong>Enrich aircraft trajectories: weather, gap filling, flight-data views.</strong></p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

barograph is a uv workspace of Python packages that attach meteorological
information to aircraft trajectories (ADS-B, flight data).

## Features

| Package | Role |
|---|---|
| [`barograph-met`](packages/barograph-met) | Wind and temperature at aircraft positions from ERA5 (137 model levels or pressure levels) and Météo-France ARPEGE (live open data and the Open-Meteo archive), with partial reads of only the bytes a flight needs, explicit coverage, and a FastMeteo-compatible façade |

## Installation

```bash
pip install 'barograph-met[grib,pandas,openmeteo]'
```

## Benchmarks

Two samples of 1 000 real flights drawn from OpenSky across all ICAO regions
(one point every 10 s, surface to FL410): 18 June 2025 for ERA5, 28 September
2026 for ARPEGE (Météo-France keeps its files 14 days). Flights are added to
each tool's own cache in steps of 1, 10, 100 and 1 000; costs are cumulative
from a cold cache, through a corporate HTTPS proxy; values are compared on
every point both tools serve.

| Case | Tool | 1 flight: time · download | 1 000 flights: time · download | Disk | Peak RAM | Values vs reference (1 000 flights) |
|---|---|---:|---:|---:|---:|---|
| ERA5, 22 pressure levels | FastMeteo 1.2.0 | 330 s · 3.6 GB | 744 s · 8.1 GB | 4.9 GB | 5.8 GB | reference |
| | **barograph-met** | **15 s · 0.09 GB** | **302 s · 3.6 GB** | **3.6 GB** | **1.5 GB** | identical: 99th pct \|Δ\| < 2e-8 on 548 098 points ¹ |
| ERA5, model levels | pycontrails 0.63.5 | 753 s · 13 GB | 1 893 s · 29.6 GB | 23.7 GB | 39 GB | reference |
| | **barograph-met** (L137) | **34 s · 0.27 GB** | **900 s · 14.6 GB** | **14.6 GB** | **3.8 GB** | RMS 0.037 m/s (u, v), 0.021 K (T); 99th pct 0.15 m/s, 0.08 K on 545 466 points ² |
| ARPEGE 0.25°, official | FastMeteo ³ | 236 s · 2.6 GB | 504 s · 2.6 GB | 2.6 GB | 13 GB | reference |
| | **barograph-met** | **95 s · 0.20 GB** | **211 s · 1.25 GB** | **1.2 GB** | **2.1 GB** | identical with the same 22 levels: \|Δ\| ≤ 3e-8 on 574 673 points ⁴ |

At 1 000 flights barograph-met is 2.1–2.5× faster, downloads 2–2.3× less and
needs 4–10× less memory; for a single flight it downloads 13–48× less and is
2.5–22× faster (ARPEGE: indexing the files dominates the first call). FastMeteo and
pycontrails download whole global fields for every hour they need, so their
cost reaches a full day as soon as the flights cover it; barograph-met reads
only the latitude bands, levels and messages the flights cross, so its cost
grows with the area covered and levels off. Points it cannot interpolate
(below the lowest level, invalid altitudes: 0.4–0.8 %) are flagged, never
extrapolated.

1. Including the points where FastMeteo 1.2.0 extrapolates across the
   Greenwich seam, RMS 0.006 m/s and 0.002 K.
2. Different vertical methods: pycontrails interpolates model levels to
   pressure levels at a reference surface pressure, barograph-met interpolates
   the model levels directly with each column's actual surface pressure.
3. FastMeteo 1.2.0 cannot reach Météo-France's new data.gouv.fr host; it runs
   from the branch of [fastmeteo#5](https://github.com/open-aviation/fastmeteo/pull/5),
   in its only full-day mode (00 UTC run, one 000H024H file).
4. By default barograph-met also uses the 275 and 925 hPa levels that
   FastMeteo drops: 0.23 m/s and 0.12 K RMS from FastMeteo's values.

### Coverage

| Coverage | FastMeteo | pycontrails | barograph-met |
|---|:---:|:---:|:---:|
| ERA5 pressure levels | 22 of 37 | 37 | 37 |
| ERA5 model levels (L137) | — | via 61 PL | direct, actual surface pressure |
| ARPEGE official (last 14 days) | ✓ | — | ✓ |
| ARPEGE archive (since Dec 2023, Open-Meteo) | — | — | ✓, chained with the official feed |
| Points outside the data | extrapolated | NaN | flagged |

Details in the [benchmark notes](https://github.com/eurocontrol-asu/barograph/blob/main/packages/barograph-met/docs/explanation/benchmarks.md).

## Development

```bash
uv sync --all-packages --all-groups
uv run prek install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
make check      # lint + types + tests
make quality    # axm audit + axm init_check
make docs-serve
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[EUPL-1.2](LICENSE).
