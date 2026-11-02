# Lazy weather for native pycontrails

`PycontrailsWeather` supplies native `MetDataset` objects whose fields are read
only when pycontrails evaluates spatial, pressure and time slices. It uses
Barograph's normal selective range reader and optional shared byte cache; it
does not replace or patch CoCiP physics. Install the optional pycontrails stack
before calling `datasets`; importing Barograph does not import that stack.

```python
import numpy as np
from baro.met import ArcoRadiationSource, PycontrailsWeather, arco_pressure_levels
from pycontrails.models.cocip import Cocip

# Keep the complete native pressure column for optical-depth dependencies.
source = arco_pressure_levels(fields={
    "u": "u_component_of_wind", "v": "v_component_of_wind",
    "t": "temperature", "q": "specific_humidity", "omega": "vertical_velocity",
    "ciwc": "specific_cloud_ice_water_content",
})
radiation = ArcoRadiationSource()
weather = PycontrailsWeather(source, radiation, cache_dir="/path/to/cache")
met, rad = weather.datasets(
    np.datetime64("2017-01-01T00"), np.datetime64("2017-01-01T06")
)
model = Cocip(met, rad, params={
    "interpolation_bounds_error": True,
    "compute_tau_cirrus_in_model_init": False,
})
result = model.eval(flight)
print(weather.cache_info())
```

The caller supplies a horizon covering formation and all requested evolution.
This is an explicit cap, not a claim that contrails disappear by that time.
Missing, out-of-coverage or nonfinite data raise errors. Use strict native
interpolation bounds; disabling them can turn missing weather into apparent
physical extinction. No radiation timestamps are silently shifted by this
bridge. The source supplies effective interpolation timestamps and normalized
fluxes; pycontrails receives `W m**-2` and the already-resolved timestamp marker.

The default pressure fields are `u`, `v`, `t`, `q`, `omega`, `tau_cirrus`.
For partial consumers, pass `fields=("u", "v")`; radiation may be omitted.
Cirrus optical depth requires a complete native pressure column of temperature
and cloud ice when first queried, unless the source supplies optical depth.
Its computation remains the optional engine's native calculation. Not every
source supplies the full CoCiP weather stack. Metadata is inherited or supplied
explicitly with `provenance`; unknown provider/dataset/product stays unknown.

The decoded LRU is thread-safe and bounded by array payload bytes (256 MiB by
default). `clear()` drops its retained arrays. The budget excludes coordinate
keys, source buffers, temporary full columns, caller-held arrays and native
consumer caches; it is not a process memory limit. Counters remain bounded in
size and count successful field reads, cache hits and materialization seconds.
Spatial/pressure coordinates stay native. Global longitudes include their
periodic closing endpoint; regional grids crossing the antimeridian need an
explicit regional adapter. Reads preserve float64 to avoid reducing engine
parity through an extra float32 conversion.

Decoded Blosc run reuse is a separate, reader-level option, disabled by default:

```python
source.configure_decoded_cache(512 * 1024**2)
radiation.grid.configure_decoded_cache(64 * 1024**2)
print(source.store.decoded_cache_info())
source.store.clear_decoded_cache()
```

Each reader owns its budget, shared by its field views. Configure after choosing
source fields. Only final field chunks qualify, using the same `final_after`
policy as the on-disk RangeCache. Recent ERA5T never qualifies; bridge arrays
also bypass retention for recent data, conservatively allowing a complete time
chunk beyond the request. No disk cache policy is changed. Separate model-level
surface-pressure readers require their own configuration.

The cache retains exact decoded Blosc block runs keyed by array/chunk/run,
including level offsets. Overlapping selections reuse runs only when their
block boundaries match; it does not prefetch a larger region. Full-chunk
fallbacks, oversized runs and disabled caches bypass retention. Independent
loads keep parallel I/O; simultaneous misses for one run may decode twice but
retain one entry. Clear/reconfigure invalidates retention from in-flight loads.
Counters survive clearing and configuration. A cache cap bounds retained byte
payload, not process memory, and makes no unconditional speedup guarantee.

For direct `ZarrStore` use, `configure_decoded_cache(max_bytes,
immutable=predicate)` requires an explicit immutable-chunk predicate. Never
mark mutable forecast chunks immutable. Reader metadata and backing-store
identity are fixed for its lifetime: create a new reader after changing a
store, forecast snapshot or metadata. `clear_decoded_cache()` releases decoded
payload; it does not refresh reader metadata or invalidate consumer copies.
