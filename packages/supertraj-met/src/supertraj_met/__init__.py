"""Fast weather retrieval along aircraft trajectories.

ERA5 on its 137 model levels (or pressure levels), Météo-France ARPEGE (live
open data and the Open-Meteo archive), interpolated at aircraft positions with
explicit coverage and no silent extrapolation.
"""

from __future__ import annotations

from supertraj_met.arpege import ArpegeSource
from supertraj_met.base import CubeKey, DataUnavailableError, Source
from supertraj_met.cache import CubeCache, CubeQuery
from supertraj_met.chain import SourceChain
from supertraj_met.domain import Cube, Domain
from supertraj_met.enrich import Enricher, EnrichResult, UncoveredPointsWarning
from supertraj_met.isa import pressure_from_altitude_ft
from supertraj_met.kernel import Flag, SampleOptions, sample, track_components
from supertraj_met.limits import set_host_limit
from supertraj_met.openmeteo import OpenMeteoSource
from supertraj_met.sources import GridSource, arco_model_levels, arco_pressure_levels
from supertraj_met.vertical import VerticalCoordinate, model_levels, pressure_levels

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
