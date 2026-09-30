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

Same points for every tool, cold cache (first run), through a corporate HTTPS
proxy. Gains are reference ÷ barograph-met.

| Case | Reference | Time | Downloaded | Peak RAM | Values vs reference |
|---|---|---:|---:|---:|---|
| ERA5, 22 pressure levels | FastMeteo 1.2.0 | 214 s → **12 s** (×18) | 1 296 → **100 MB** (×13) | 1.5 → **0.16 GB** | identical (≤ 1e-13), except FastMeteo's extrapolated points at the Greenwich seam |
| ERA5, 137 model levels | pycontrails 0.63.5 | 322 s → **31 s** (×10) | 4 736 → **349 MB** (×14) | 7.0 → **0.22 GB** | RMS 0.03 m/s (u, v), 0.016 K (T) |
| ARPEGE 0.1°, official (14 days) | FastMeteo 1.2.0 | 67 s → **13 s** (×5) | 475 → **54 MB** (×9) | — | identical east of Greenwich; FastMeteo extrapolates west of it |

> **Scope: one flight.** These figures compare a single trajectory. With many
> flights on the same day, tools that download whole global fields (FastMeteo,
> pycontrails) amortise them across flights, while barograph-met reads more
> bands as flights spread out, so the gap narrows. A multi-flight benchmark
> (1 to 1 000 flights on one day) is planned.

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
