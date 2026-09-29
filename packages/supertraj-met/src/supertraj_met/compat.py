"""FastMeteo-compatible façade (``pip install 'supertraj-met[pandas]'``).

Same class names, constructor arguments and ``interpolate(df)`` contract as
FastMeteo 1.2 (input columns ``timestamp``, ``latitude``, ``longitude``,
``altitude`` in ft; output = input + one column per feature), backed by this
library's engine. Deliberate differences, all documented:

* points that cannot be interpolated get NaN instead of an extrapolated value;
  the reason is available in ``last_result.flags`` (or ``flags_column``);
* ``local_store`` is an immutable regional cube cache, not a global zarr copy;
  ``mode="global"`` reads whole fields (FastMeteo-like cost, reusable cache);
* model levels use all 137 levels with the actual surface pressure;
* ARPEGE lead-time files are resolved automatically (``time_range`` is ignored).

With default options, pressure-level results equal FastMeteo's to float
round-off wherever FastMeteo does not extrapolate.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from supertraj_met.arpege import ARPEGE_BASE_URL, ArpegeSource, RunPolicy
from supertraj_met.base import Source
from supertraj_met.enrich import Enricher, EnrichResult, Mode
from supertraj_met.kernel import VerticalMethod
from supertraj_met.sources import (
    ARCO_BASE_URL,
    FASTMETEO_LEVELS_22,
    arco_model_levels,
    arco_pressure_levels,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ArcoEra5", "Arpege"]

ERA5_FEATURES = ["u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"]
ARPEGE_FEATURES = ["u", "v", "t", "r"]


class _FrameEnricher:
    features: list[str]
    flags_column: str | None
    last_result: EnrichResult | None

    def _setup(self, source: Source, local_store: str | None, **options: Any) -> None:
        self._enricher = Enricher(source, cache_dir=local_store, **options)
        self.last_result = None

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with one column per feature (NaN where not interpolable)."""
        import pandas as pd

        times = pd.to_datetime(df["timestamp"])
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        res = self._enricher.enrich(
            times.to_numpy("datetime64[ns]"),
            df["latitude"].to_numpy(np.float64),
            df["longitude"].to_numpy(np.float64),
            altitude_ft=df["altitude"].to_numpy(np.float64),
        )
        self.last_result = res
        out = df.drop(columns=self.features, errors="ignore").copy()
        for name in self.features:
            out[name] = res[name]
        if self.flags_column is not None:
            out[self.flags_column] = res.flags
        return out


class ArcoEra5(_FrameEnricher):
    """ERA5 from ARCO: ``model_levels=37`` (pressure levels) or ``137`` (model levels)."""

    def __init__(
        self,
        local_store: str | None = None,
        model_levels: Literal[37, 137] = 37,
        features: list[str] | None = None,
        *,
        levels: Sequence[int] | None = None,
        mode: Mode = "flight",
        vertical: VerticalMethod | None = None,
        flags_column: str | None = None,
        base_url: str = ARCO_BASE_URL,
        **options: Any,
    ) -> None:
        self.features = list(features or ERA5_FEATURES)
        self.flags_column = flags_column
        fields = {f: f for f in self.features}
        if model_levels == 37:
            source: Source = arco_pressure_levels(
                levels=FASTMETEO_LEVELS_22 if levels is None else levels,
                fields=fields,
                base_url=base_url,
            )
            default_vertical: VerticalMethod = "linear"  # FastMeteo interpolates linearly in p
        elif model_levels == 137:
            source = arco_model_levels(levels=levels, fields=fields, base_url=base_url)
            default_vertical = "log"
        else:
            raise ValueError("model_levels must be 37 or 137")
        self._setup(
            source, local_store, mode=mode, vertical=vertical or default_vertical, **options
        )


class Arpege(_FrameEnricher):
    """Météo-France ARPEGE forecasts (0.25 deg global 3-hourly, or 0.1 deg Europe hourly)."""

    def __init__(
        self,
        local_store: str | None = None,
        levels: Sequence[int] | None = None,
        features: list[str] | None = None,
        resolution: Literal["025", "01"] = "025",
        package: str = "IP1",
        time_range: str | None = None,
        *,
        run: RunPolicy = "daily",
        mode: Mode = "flight",
        vertical: VerticalMethod = "linear",
        index_dir: str | None = None,
        flags_column: str | None = None,
        base_url: str = ARPEGE_BASE_URL,
        **options: Any,
    ) -> None:
        if time_range is not None:
            warnings.warn(
                "time_range is resolved automatically from each point's lead time and is ignored",
                stacklevel=2,
            )
        self.features = list(features or ARPEGE_FEATURES)
        self.flags_column = flags_column
        if index_dir is None and local_store is not None:
            index_dir = str(Path(local_store) / "grib-index")
        source = ArpegeSource(
            resolution=resolution,
            package=package,
            fields={f: f for f in self.features},
            levels=FASTMETEO_LEVELS_22 if levels is None else levels,
            run=run,
            index_dir=index_dir,
            base_url=base_url,
        )
        self._setup(source, local_store, mode=mode, vertical=vertical, **options)

    def local_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias kept for FastMeteo compatibility (there is no global zarr to build)."""
        return self.interpolate(df)
