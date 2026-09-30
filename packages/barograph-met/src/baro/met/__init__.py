"""Fast weather retrieval along aircraft trajectories.

ERA5 on its 137 model levels (or pressure levels), Météo-France ARPEGE (live
open data and the Open-Meteo archive), interpolated at aircraft positions with
explicit coverage and no silent extrapolation.
"""

from __future__ import annotations

from baro.met.arpege import ArpegeSource
from baro.met.base import CubeKey, DataUnavailableError, Source
from baro.met.cache import CubeCache, CubeQuery
from baro.met.chain import SourceChain
from baro.met.domain import Cube, Domain
from baro.met.enrich import Enricher, EnrichResult, UncoveredPointsWarning
from baro.met.isa import pressure_from_altitude_ft
from baro.met.kernel import Flag, SampleOptions, sample, track_components
from baro.met.limits import set_host_limit
from baro.met.openmeteo import OpenMeteoSource
from baro.met.sources import GridSource, arco_model_levels, arco_pressure_levels
from baro.met.vertical import VerticalCoordinate, model_levels, pressure_levels

__all__ = [
    "ArpegeSource",
    "Cube",
    "CubeCache",
    "CubeKey",
    "CubeQuery",
    "DataUnavailableError",
    "Domain",
    "EnrichResult",
    "Enricher",
    "Flag",
    "GridSource",
    "OpenMeteoSource",
    "SampleOptions",
    "Source",
    "SourceChain",
    "UncoveredPointsWarning",
    "VerticalCoordinate",
    "arco_model_levels",
    "arco_pressure_levels",
    "model_levels",
    "pressure_from_altitude_ft",
    "pressure_levels",
    "sample",
    "set_host_limit",
    "track_components",
]
