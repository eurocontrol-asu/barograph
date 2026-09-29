<p align="center">
  <img src="assets/logo.svg" alt="supertraj" width="120" />
</p>

<h1 align="center">supertraj</h1>
<p align="center"><strong>Weather along aircraft trajectories.</strong></p>

<p align="center">
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/ci.yml"><img src="https://github.com/asu-eurocontrol/supertraj/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/asu-eurocontrol/supertraj/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asu-eurocontrol/supertraj/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

supertraj is a uv workspace of Python packages that attach meteorological
information to aircraft trajectories (ADS-B, flight data).

## Features

| Package | Role |
|---|---|
| [`supertraj-met`](packages/supertraj-met) | Wind and temperature at aircraft positions from ERA5 (137 model levels or pressure levels) and Météo-France ARPEGE (live open data and the Open-Meteo archive), with partial reads of only the bytes a flight needs, explicit coverage, and a FastMeteo-compatible façade |

## Installation

```bash
pip install 'supertraj-met[grib,pandas,openmeteo]'
```

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
