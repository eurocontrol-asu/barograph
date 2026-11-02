"""Fast weather retrieval along aircraft trajectories.

ERA5 on model or pressure levels and forecast fields from Météo-France and
Open-Meteo, interpolated with explicit coverage and no silent extrapolation.
Surface queries accept fixed sites or trajectories without altitude.
"""

from __future__ import annotations

from baro.met.arco_surface import ArcoRadiationSource
from baro.met.arome_surface import AromeSurfaceSource
from baro.met.arpege import ArpegeSource
from baro.met.base import CubeKey, DataUnavailableError, Source
from baro.met.chain import SourceChain
from baro.met.diffcocip import EvolutionResult, WeatherDemand, run_diffcocip
from baro.met.domain import Cube, Domain
from baro.met.enrich import Enricher, EnrichResult, UncoveredPointsWarning
from baro.met.immediate import AromePISource, PiafSource
from baro.met.isa import pressure_from_altitude_ft
from baro.met.kernel import Flag, SampleOptions, sample, track_components
from baro.met.limits import set_host_limit
from baro.met.materialize import WeatherRequest, materialize_radiation, materialize_weather
from baro.met.openmeteo import OpenMeteoSource
from baro.met.openmeteo_surface import OpenMeteoSurfaceSource
from baro.met.pycontrails import PycontrailsWeather
from baro.met.rangecache import RangeCache
from baro.met.session import WeatherSession
from baro.met.sources import GridSource, arco_model_levels, arco_pressure_levels
from baro.met.surface import (
    SurfaceCube,
    SurfaceEnricher,
    SurfaceResult,
    SurfaceSource,
    SurfaceVariable,
)
from baro.met.tiling import Tiling
from baro.met.vertical import VerticalCoordinate, model_levels, pressure_levels
from baro.met.weather import RadiationCube, WeatherCube
from baro.met.windows import WeatherWindowProvider

__all__ = [
    "ArcoRadiationSource",
    "AromePISource",
    "AromeSurfaceSource",
    "ArpegeSource",
    "Cube",
    "CubeKey",
    "DataUnavailableError",
    "Domain",
    "EnrichResult",
    "Enricher",
    "EvolutionResult",
    "Flag",
    "GridSource",
    "OpenMeteoSource",
    "OpenMeteoSurfaceSource",
    "PiafSource",
    "PycontrailsWeather",
    "RadiationCube",
    "RangeCache",
    "SampleOptions",
    "Source",
    "SourceChain",
    "SurfaceCube",
    "SurfaceEnricher",
    "SurfaceResult",
    "SurfaceSource",
    "SurfaceVariable",
    "Tiling",
    "UncoveredPointsWarning",
    "VerticalCoordinate",
    "WeatherCube",
    "WeatherDemand",
    "WeatherRequest",
    "WeatherSession",
    "WeatherWindowProvider",
    "arco_model_levels",
    "arco_pressure_levels",
    "materialize_radiation",
    "materialize_weather",
    "model_levels",
    "pressure_from_altitude_ft",
    "pressure_levels",
    "run_diffcocip",
    "sample",
    "set_host_limit",
    "track_components",
]
