# Public Python API

Everything exported by `baro.met`. The first group is what most users need;
the second is for writing a source of one's own or working on cubes directly.

::: baro.met
    options:
      show_submodules: false
      members:
        - Enricher
        - EnrichResult
        - WeatherSession
        - WeatherWindowProvider
        - ArcoRadiationSource
        - materialize_weather
        - materialize_radiation
        - run_diffcocip
        - EvolutionResult
        - arco_model_levels
        - arco_pressure_levels
        - ArpegeSource
        - OpenMeteoSource
        - OpenMeteoSurfaceSource
        - AromeSurfaceSource
        - AromePISource
        - PiafSource
        - SurfaceEnricher
        - SurfaceResult
        - SourceChain
        - Domain
        - Flag
        - DataUnavailableError
        - UncoveredPointsWarning
        - SampleOptions
        - track_components
        - pressure_from_altitude_ft
        - set_host_limit

## Building blocks

::: baro.met
    options:
      show_submodules: false
      members:
        - Source
        - SurfaceSource
        - SurfaceCube
        - SurfaceVariable
        - CubeKey
        - Cube
        - WeatherCube
        - RadiationCube
        - WeatherRequest
        - WeatherDemand
        - PycontrailsWeather
        - GridSource
        - Tiling
        - RangeCache
        - VerticalCoordinate
        - model_levels
        - pressure_levels
        - sample
