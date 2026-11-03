<p align="center">
  <img src="assets/logo.svg" alt="barograph" width="120" />
</p>

<h1 align="center">barograph</h1>
<p align="center"><strong>Enrich aircraft trajectories — starting with weather at full model resolution.</strong></p>

barograph is a uv workspace of Python packages that attach meteorological
information to aircraft trajectories (ADS-B, flight data). Its first package,
[barograph-met](met/index.md), retrieves wind and temperature at aircraft
positions from ERA5 and Météo-France ARPEGE:

- **fast** — only the bytes a flight needs are downloaded (partial reads of
  compressed ARCO-ERA5 blocks and of GRIB2 messages), for every flight of a
  time slice at once;
- **exact on the vertical** — ERA5's 137 native model levels, with each grid
  column's actual surface pressure;
- **explicit** — no silent extrapolation: every missing value carries a
  reason (outside the grid, above/below the levels, not published, before the
  archive starts);
- **compatible** — a façade with the same API as
  [FastMeteo](https://github.com/open-aviation/fastmeteo).

| Package | Role |
|---|---|
| [barograph-met](met/index.md) | Retrieval and interpolation of ERA5, ARPEGE and Open-Meteo archives at aircraft positions |

## Install barograph-met

Requires Python 3.12+ on a POSIX system (Linux or macOS).

With uv, in an activated virtual environment:

```bash
uv pip install 'barograph-met[grib,pandas,openmeteo]'
```

Or with pip:

```bash
python -m pip install 'barograph-met[grib,pandas,openmeteo]'
```

For a uv-managed project, use `uv add barograph-met` and select extras as needed.
Start with the [first-flight tutorial](met/tutorials/getting-started.md).

!!! danger "Research use only"
    This software is provided for research purposes and is not a regulatory
    or operational tool; EUROCONTROL disclaims any responsibility for misuse or
    operational application.

Licensed under the [EUPL-1.2](https://eupl.eu/1.2/en/) — © 2026 Eurocontrol.
The data read by barograph-met stay under their providers' licences; see
[data sources and licences](met/index.md#data-sources-and-licences).
