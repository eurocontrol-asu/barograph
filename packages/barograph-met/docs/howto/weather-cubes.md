# Export weather grids to contrail engines

`baro.met.weather.WeatherCube` and `RadiationCube` validate already retrieved
weather and export it without introducing a mandatory contrail-engine dependency.
They do not fetch weather or compute optical depth. A provider offering wind and
temperature alone does not thereby become a complete CoCiP weather source.

Construct a pressure cube with keyword arguments `lon`, `lat`, `level_hpa`,
`time_ns`, `fields`, `units`, and `provenance`. Coordinates must be strictly
ascending; time is signed integer Unix nanoseconds in UTC. Every field has shape
`(longitude, latitude, pressure, time)`. Descending source coordinates must be
reordered together with the data before construction. Pressure is in hPa.

| Field | Unit |
| --- | --- |
| `u`, `v` | `m s-1` |
| `t` | `K` |
| `q`, `ciwc` | `kg kg-1` |
| `omega` | `Pa s-1` |
| `tau_cirrus` | `1` |
| `geopotential` | `m2 s-2` |

CoCiP exports require `u`, `v`, `t`, `q`, `omega`, and `tau_cirrus`. The latter
must be calculated from an appropriate atmospheric column before cropping to
flight levels. Missing or nonfinite values are rejected; missing weather must
never silently become physical contrail extinction.

A radiation cube has the same arguments except `level_hpa`, with field shape
`(longitude, latitude, time)`. It requires an explicit `time_semantics`:
`valid` for instantaneous flux, or `interval_midpoint` for flux averaged over
an interval and assigned to that interval's midpoint. Fields are
`top_net_solar_radiation` and `top_net_thermal_radiation`, both in `W m-2`.
Accumulated `J m-2` and interval-end timestamps must be resolved upstream.
Neither export applies a time shift. For example, an ERA5 hourly flux for the
interval ending at 12:00 belongs at 11:30 after conversion to a mean flux.

```python
met_arrays = weather.to_diffcocip()
rad_arrays = radiation.to_diffcocip()

# Optional dependencies: pycontrails and xarray
met = weather.to_pycontrails()
rad = radiation.to_pycontrails()
```

The diffcocip export creates plain NumPy dictionaries, including seconds since
2000 and the solar time bridge. It requires at least two points on every axis.
It does not import JAX. The pycontrails export maps units to engine conventions
and marks radiation timestamps as already resolved to prevent a second shift.
`to_pycontrails(require_cocip=False)` permits partial grids for preprocessing.

`coverage` describes inclusive coordinate bounds; it is not permission to
extrapolate. `provenance` is a caller-supplied string mapping retained by the
container and pycontrails export. Keep the container alongside diffcocip arrays
when provenance is needed, since the engine dict contains numerical arrays only.
For example, distinguish ERA5 reanalysis from assembled forecast archives whose
original run times and forecast leads are unavailable.

## Retrieve a native window

`materialize_weather` and `materialize_radiation` connect these containers to
the targeted ARCO readers. This path supports pressure-level ERA5; it does not
implicitly convert model levels or invent missing fields for other providers.

```python
import numpy as np
from baro.met import (
    ArcoRadiationSource, Domain, RangeCache, WeatherRequest,
    arco_pressure_levels, materialize_radiation, materialize_weather,
)

source = arco_pressure_levels(fields={
    "u": "u_component_of_wind", "v": "v_component_of_wind",
    "t": "temperature", "q": "specific_humidity",
    "omega": "vertical_velocity", "ciwc": "specific_cloud_ice_water_content",
})
radiation_source = ArcoRadiationSource()
cache = RangeCache(".weather-cache")
source.use_cache(cache)
radiation_source.use_cache(cache)
request = WeatherRequest(
    Domain(48, 49, 1, 2),
    np.datetime64("2017-05-28T09:30"), np.datetime64("2017-05-28T10:00"),
    (25000, 30000),  # Pa
)
weather = materialize_weather(source, request, tau_cirrus=True)
radiation = materialize_radiation(radiation_source, request)
```

Without `tau_cirrus=True`, only the requested local variables and pressure
neighbours are read. Deriving cirrus optical depth additionally reads temperature
and cloud ice from the top of the native archive down to a lower guard level.
It uses optional pycontrails for the grid calculation, before cropping. If
`geopotential` is configured it supplies the height field; otherwise pycontrails'
ISA height approximation is used and recorded in provenance. Keep the full
native pressure axis on the source for this calculation. Native variable units
are checked when present; with missing unit metadata, the caller is responsible
for supplying fields in the declared units.

Radiation reads are independent of pressure levels. ERA5 hourly `J m**-2`
values are divided by 3600 and placed 30 minutes before their archive timestamp.
The negative sign of net thermal flux is retained. These are top-of-atmosphere
fluxes, not irradiance at ground level; no finite height above ground applies.
`SurfaceEnricher(ArcoRadiationSource(), wind_components=None)` can also sample
them directly at geographic points without altitude.

Windows reject requests outside their source's coverage. They include the
neighbours required by interpolation, but provide no guarantee about future
advection. A pressure margin must account for lower probes and sedimentation.
Materialization holds the requested window in memory; for repeated point
sampling use `WeatherSession` with its bounded decoded-cube cache instead.
