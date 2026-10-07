# Vertical interpolation

Every vertical coordinate is expressed as `p_k = A_k + B_k · ps`:

- ERA5 model levels (hybrid sigma-pressure, IFS L137): `A_k`, `B_k` are the
  means of the half-level coefficients (`p_half = a + b · ps`, full level =
  mean of the two adjacent half levels);
- pressure levels: `A_k = p_k`, `B_k = 0`.

For each of the four grid columns around a point, level pressures are
computed from **that column's surface pressure**, the two levels bracketing
the aircraft pressure are found, and the field is interpolated linearly in
`ln p` (or `p`). The four column values are combined bilinearly, then the two
native time-step values linearly in time (1 h for ERA5/ARPEGE, 3 h for
the Open-Meteo IFS archive).

Barometric altitude is a pressure altitude (standard setting), so converting
it with the ICAO standard atmosphere is exact. A fixed altitude→level table
(standard surface pressure) is off by about 90 m at cruise in a 950 hPa low and
by hundreds of metres over high terrain; `surface_pressure="standard"`
reproduces it for comparison.

Nothing is extrapolated: a point above the top or below the bottom level of
any of its columns is flagged instead.
