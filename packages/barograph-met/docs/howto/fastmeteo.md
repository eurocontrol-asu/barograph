# The FastMeteo-compatible façade

```python
from baro.met.compat import ArcoEra5, Arpege   # pip install 'barograph-met[pandas,grib]'

ArcoEra5("cache/", model_levels=37).interpolate(df)    # FastMeteo's 22 levels, linear in p
ArcoEra5("cache/", model_levels=137).interpolate(df)   # all 137 levels + surface pressure
Arpege("cache/", resolution="01").interpolate(df)      # same columns as FastMeteo
ArcoEra5("cache/", mode="global").interpolate(df)      # whole fields: the cache serves any flight
```

Same constructors, same `interpolate(df)` contract (columns `timestamp`,
`latitude`, `longitude`, `altitude` in ft; one output column per feature).
Differences, all deliberate:

- points that cannot be interpolated are `NaN` (reason in
  `last_result.flags`, or in a column with `flags_column="flags"`) instead of
  extrapolated values;
- longitudes are handled across Greenwich;
- `local_store` is a cache of the byte ranges read (immutable data only), not a global zarr copy;
- ARPEGE lead-time files are found automatically (`time_range` is ignored).

With `model_levels=37` the values equal FastMeteo's to float round-off
wherever FastMeteo does not extrapolate.
