# ERA5 on model or pressure levels

```python
from baro.met import Enricher, arco_model_levels, arco_pressure_levels

ml = Enricher(arco_model_levels())                        # 137 hybrid levels
pl = Enricher(arco_pressure_levels())                     # 37 pressure levels
pl22 = Enricher(arco_pressure_levels(levels=[100, 125, 150, 175, 200, 225, 250,
                                             300, 350, 400, 450, 500, 550, 600,
                                             650, 700, 750, 800, 850, 900, 950, 1000]))
```

- **Altitude**: pass `altitude_ft=` (barometric/pressure altitude, converted
  with the ICAO standard atmosphere — exact for ADS-B) or `pressure_pa=`.
- **Other variables**: `fields={"q": "specific_humidity"}` maps output names to
  ARCO array names. A non-empty `fields` replaces the default `u`, `v`, `t`; see
  [Variables by source](../reference/variables.md).
- **Vertical interpolation**: `Enricher(source, vertical="linear")` interpolates
  linearly in pressure instead of in ln p;
  `surface_pressure="standard"` ignores the actual surface pressure (ablation:
  this is FastMeteo's fixed altitude→level table).
- **Coverage**: the model-level store lags real time by about three months,
  the pressure-level store by about six days. Points after the last published
  hour are flagged `OUT_OF_COVERAGE` with a warning.

A local mirror of the ARCO stores (same directory layout) can be used with
`arco_model_levels(base_url="/data/arco")`.
