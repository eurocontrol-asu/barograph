# Flags and statistics

## `Flag`

Values are `NaN` whenever the flag is not `OK`. The most fundamental cause is
reported.

| Flag | Value | Meaning |
|---|---|---|
| `OK` | 0 | interpolated |
| `INVALID_INPUT` | 1 | non-finite time, position or pressure |
| `OUTSIDE_DOMAIN` | 2 | outside the horizontal grid (e.g. regional ARPEGE grid) |
| `ABOVE_TOP` | 4 | above the highest level read |
| `BELOW_BOTTOM` | 8 | below the lowest level (near or below the surface) |
| `MISSING_DATA` | 16 | NaN in the source, or cube not published |
| `OUT_OF_COVERAGE` | 32 | time outside the period the source covers |

`Enricher(on_uncovered=...)` controls points outside the source's period:
`"warn"` (default) flags them and emits an `UncoveredPointsWarning` naming the
covered period, `"raise"` raises `DataUnavailableError`, `"flag"` only flags.

## `EnrichResult.stats`

| Key | Meaning |
|---|---|
| `cubes` | cubes needed (time slices × groups of occupied tiles) |
| `fetched`, `unavailable` | how many were read, how many do not exist |
| `cube_bytes` | memory of the cubes read |
| `bytes`, `requests` | network transfer of this call |
| `cache_bytes` | bytes served from the on-disk cache instead |
| `throttled` | 429/5xx answers retried |
| `seconds` | wall time |
