# Architecture

```mermaid
flowchart LR
    P[points of all flights] --> E[Enricher: plan per time interval]
    E --> C{cube cache}
    C -- hit --> K[kernel: vertical, horizontal, time]
    C -- miss --> S[Source.fetch]
    S --> ST[stores: partial reads]
    ST --> S --> C
    K --> R[values + flags + stats]
```

- **Sources** (`GridSource` for ARCO-ERA5, `ArpegeSource`, `OpenMeteoSource`,
  `SourceChain`) implement one contract: a time step, the two cube keys
  bracketing an interval (with a forecast run when relevant), a coverage
  period, and `fetch` returning a regional *cube* (levels × lat × lon).
- **Stores** read bytes: `HttpStore` (range requests, retries, per-host caps,
  redirect learning), `FileStore` for local mirrors, and format readers on
  top (zarr-v2 with Blosc block reads, GRIB2 message index, Open-Meteo `.om`
  tiles).
- **Enricher** groups points by interval, fetches each cube once with bounded
  look-ahead, and frees it after its last use.
- **Kernel** interpolates a cube at points (see
  [vertical interpolation](vertical.md)).
- **Cache** stores cubes immutably with a manifest; a cube serves any later
  request it encloses.
