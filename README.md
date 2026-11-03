<p align="center">
  <img src="assets/logo.svg" alt="barograph" width="120" />
</p>

<h1 align="center">barograph</h1>
<p align="center"><strong>Targeted weather for aircraft trajectories, fixed sites and contrail models.</strong></p>

<p align="center">
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/barograph/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://forge.axm-protocols.io/audit/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://forge.axm-protocols.io/init/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/eurocontrol-asu/barograph/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/barograph/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://eurocontrol-asu.github.io/barograph/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="EUPL-1.2">
</p>

---

barograph is a uv workspace of Python packages that retrieve meteorology along
aircraft trajectories (ADS-B, flight data) or at fixed sites. Reads select the
needed fields, times, levels and storage blocks, with explicit coverage and
shared caching.

## Features

| Package | Role |
|---|---|
| [`barograph-met`](packages/barograph-met) | Wind and temperature aloft; surface wind, gusts, precipitation and temperature; targeted ERA5 humidity, vertical motion, cirrus and radiation for optional contrail consumers |

| Source | Interface | Data and access |
|---|---|---|
| ERA5 | `arco_model_levels()`, `arco_pressure_levels()` | Reanalysis on model or pressure levels; public ARCO archive |
| ARPEGE | `ArpegeSource`, `OpenMeteoSource` | Public native forecasts or assembled Open-Meteo archive |
| IFS | `OpenMeteoSource("ecmwf_ifs025")` | Assembled forecast archive, native 3-hourly pressure-level wind/temperature |
| AROME aloft | `OpenMeteoSource("meteofrance_arome_france0025")` | Public hourly assembled forecast archive |
| AROME surface | `AromeSurfaceSource` | Public native forecasts, preserving initialization and lead |
| AROME-PI | `AromePISource` | Surface forecasts every 15 minutes; Météo-France API subscription/key |
| PIAF | `PiafSource` | Precipitation every 5 minutes; Météo-France API subscription/key |

IFS and the Open-Meteo ARPEGE/AROME archives combine forecasts at valid times;
their original run identities and leads are unavailable. They are distinct from
ERA5 reanalysis. Fields and coverage depend on the selected source.

- [IFS wind and temperature](packages/barograph-met/docs/howto/ifs.md).
- [AROME aloft](packages/barograph-met/docs/howto/arome-pressure.md) and
  [AROME surface forecasts](packages/barograph-met/docs/howto/arome-runs.md).
- [AROME-PI/PIAF access and private key files](packages/barograph-met/docs/howto/immediate-forecasts.md#credentials-and-fixed-runs).
- [Lazy native pycontrails weather](packages/barograph-met/docs/howto/lazy-pycontrails-weather.md), with targeted field reads and optional bounded decoded Blosc reuse. This supplies meteorology to the engine; it does not change contrail physics.

## Installation

Requires Python 3.12+ on a POSIX system (Linux or macOS). Native Windows is
not supported because the shared disk cache uses POSIX file locks.

With **uv**:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'barograph-met[grib,pandas,openmeteo]'
# Optional native pycontrails adapter and cirrus calculation:
uv pip install 'barograph-met[contrails]'
```

With **pip**, in your Python environment:

```bash
python -m pip install 'barograph-met[grib,pandas,openmeteo]'
python -m pip install 'barograph-met[contrails]'
```

For ERA5 alone, install `barograph-met` without extras. In a uv-managed
project, use `uv add 'barograph-met[grib,pandas,openmeteo]'` instead.
Start with the [first-flight tutorial](https://eurocontrol-asu.github.io/barograph/met/tutorials/getting-started/).
For development from source, use the workspace setup below.

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

In these acquisition benchmarks, at 1 000 flights barograph-met is 2.1–2.5×
faster, downloads 2–2.3× less and
needs 4–10× less memory; for a single flight it downloads 13–48× less and is
2.5–22× faster (ARPEGE: indexing the files dominates the first call). FastMeteo and
the compared pycontrails configuration download whole global fields for every
hour they need, so their
cost reaches a full day as soon as the flights cover it; barograph-met reads
only the latitude bands, levels and messages the flights cross, so its cost
grows with the area covered and levels off. Points it cannot interpolate
(below the lowest level, invalid altitudes: 0.4–0.8 %) are flagged, never
extrapolated. These measurements concern weather retrieval and interpolation;
they do not compare CoCiP physics engines or establish the same ratios for
other sources, retrieval configurations or cache states.

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

### Benchmarked coverage

| Coverage | FastMeteo | pycontrails | barograph-met |
|---|:---:|:---:|:---:|
| ERA5 pressure levels | 22 of 37 | 37 | 37 |
| ERA5 model levels (L137) | — | via 61 PL | direct, actual surface pressure |
| ARPEGE official (last 14 days) | ✓ | — | ✓ |
| ARPEGE archive (since Dec 2023, Open-Meteo) | — | — | ✓, chained with the official feed |
| Points outside the data | extrapolated | NaN | flagged |

Details in the [benchmark notes](https://eurocontrol-asu.github.io/barograph/met/explanation/benchmarks/).

## Development

```bash
uv sync --all-packages --all-groups --all-extras --locked
uv run prek install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
make check      # lint + types + tests
make quality    # axm audit + axm init_check
make docs-serve
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Legal and usage

> [!IMPORTANT]
> **Research use only.** This software is provided for research purposes and
> is not a regulatory or operational tool; EUROCONTROL disclaims any
> responsibility for misuse or operational application.

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

## License

[EUPL-1.2](LICENSE) — © 2026 Eurocontrol
