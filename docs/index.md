<p align="center">
  <img src="assets/logo.svg" alt="supertraj" width="120" />
</p>

<h1 align="center">supertraj</h1>
<p align="center"><strong>Weather along aircraft trajectories.</strong></p>

supertraj is a uv workspace of Python packages that attach meteorological
information to aircraft trajectories (ADS-B, flight data). Its first package,
[supertraj-met](met/index.md), retrieves wind and temperature at aircraft
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
| [supertraj-met](met/index.md) | Retrieval and interpolation of ERA5, ARPEGE and Open-Meteo archives at aircraft positions |

Licensed under the [EUPL-1.2](https://eupl.eu/1.2/en/).
