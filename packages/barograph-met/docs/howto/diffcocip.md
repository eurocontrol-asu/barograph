# Evolve contrails with weather fetched on demand

`baro.met.diffcocip.run_diffcocip` runs the forward formation and evolution
functions from the optional `diffcocip` package. The weather provider is a
callback, so data retrieval and caching remain independent of the engine.
Installing or importing Barograph does not import JAX or diffcocip.
Install `barograph-met[contrails]` for pycontrails exports and cirrus preprocessing,
and install diffcocip separately from a repository/version you can access. The
adapter was tested against the upstream revision named below; Barograph does
not bundle that repository or configure credentials for it.

```python
from baro.met.diffcocip import run_diffcocip

# flight and params follow diffcocip's own documented input contract.
result = run_diffcocip(flight, provider, params, max_age_s=6 * 3600)
print(result.energy_forcing_j, result.stop_reason)
```

For targeted ERA5 retrieval, configure the pressure/radiation sources shown in
[Weather cubes](weather-cubes.md), then connect the built-in provider:

```python
import numpy as np
from baro.met import WeatherWindowProvider, run_diffcocip

weather = WeatherWindowProvider(
    source, radiation_source,
    cache_dir=".weather-cache",
    max_cache_bytes=256 * 1024 * 1024,
    margin_deg=0.25,
    pressure_margin_pa=1000,
    prefetch=np.timedelta64(5, "m"),
)
result = run_diffcocip(flight, weather.for_diffcocip, params, max_age_s=6 * 3600)
print(result.stop_reason, weather.cache_info())
```

These margins and the six-hour cap are example choices, not physical bounds.
The checked driver requests additional support whenever necessary. Defaults
apply no speculative margins or prefetch. For multiple trajectories, invoke
the driver separately for each flight while reusing the same provider/cache;
do not concatenate unrelated flights into a single physical contrail.
Flight performance/emissions and engine parameters remain explicit inputs.

The current provider supplies all required meteorological and radiative fields
at each requested window, including formation. It does not yet defer the first
cirrus/radiation reads until after Schmidt–Appleman filtering. Retrieval is
targeted by native blocks and window bounds, not a theoretical minimum of
individual scalar values. Performance results for earlier point-interception
prototypes do not establish the speed of this native-window implementation.

The decoded budget bounds retained arrays, not peak process memory: active
retrievals, engine exports and JAX state allocate additional memory. For wind
samples alone use `WeatherSession`; it does not read radiation or derive cirrus.
Pycontrails currently consumes exported native windows; the checked adaptive
evolution driver is specific to diffcocip.

The maximum age is explicit. `extinct` means there are no active lanes and no
future formations. `age_limit_reached` conservatively reports that a lane reached
the configured age, or survived through the final model timestep. It must not be
interpreted as natural extinction. Diagnostics include executed timesteps only;
trailing empty timesteps of the engine's dense scan are omitted.

## Provider contract

`provider(demand)` returns `(met_arrays, rad_arrays)` in diffcocip's native dict
format. `demand.points` contains exact probe coordinates as columns: longitude
in degrees, latitude in degrees, pressure in hPa, and seconds since 2000-01-01.
`demand.bounds` contains `(minimum, maximum)` tuples under `lon`, `lat`, `level`
and `time_s`. Fetch original grid nodes bracketing these coordinates, with at
least two nodes on each axis. Do not interpolate to an artificial grid before
handing the arrays to the engine.

The meteorological dict has axes `lon`, `lat`, `level`, `time_s`, and fields
`air_temperature`, `specific_humidity`, `eastward_wind`, `northward_wind`,
`lagrangian_tendency_of_air_pressure`, `tau_cirrus`. The radiation dict has axes
`lon`, `lat`, `time_s`, and fields `top_net_solar_radiation` and
`top_net_thermal_radiation`. Field dimensions follow this axis order.
All axes increase strictly. Temperature is K, humidity kg/kg, wind m/s, pressure
velocity Pa/s, and radiation W/m² at its valid midpoint time. `tau_cirrus` is
computed on the native grid before interpolation, using the required upper
column. Provider responses must come from one consistent dataset: enlarging a
window must preserve already supplied values.

Each demand covers probes needed for one current phase, not the entire future
integration horizon. Keep a bounded decoded cache and a shared byte cache in
the provider. Returning extra coverage can reduce repeated requests. Longitudes
use [-180, 180]; crossings of the antimeridian currently request a potentially
wide minimum/maximum interval rather than an optimized periodic window.

## Coverage and engine compatibility

Before accepting formation, the adapter checks original, lower-gradient and
wake-downwash probes. Before accepting each evolution step, it checks start,
end, lower-gradient and radiation probes. If a trial leaves the available
weather, it fetches the missing support and replays the unchanged previous
state. An uncovered request or nonfinite meteorology raises
`DataUnavailableError`; it is never interpreted as contrail disappearance.
A latitude beyond ±80° raises, matching the upstream forward model's supported
scope rather than silently accepting its polar-exit mask.

This adapter targets diffcocip 1.0.0's formation/evolution APIs, audited at
revision `a485c1c99d460317423fdc8704ff61b2774092b1`. Its explicit probe checks must
be reviewed if upstream adds interpolation sites. Synthetic extension tests and
an offline realistic ERA5 fixture check its parity with the same engine's dense
scan. This is not a validation of atmospheric truth or equivalence to every
pycontrails version.

The host loop is forward-only: it supports neither JAX tracing/gradients nor
soft-persistence smoothing. For gradients, first prepare a fixed complete
weather domain and call diffcocip's differentiable API directly. No global
monkeypatch or upstream code change is required.
