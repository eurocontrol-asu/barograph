"""Opt-in tiny authenticated requests; credentials are read from external files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from baro.met import AromePISource, PiafSource, SurfaceEnricher

pytestmark = pytest.mark.live


@pytest.mark.parametrize("model", ["arome-pi", "arome-pi-hd", "piaf"])
def test_live_native_surface_and_cache(model: str, tmp_path: Path) -> None:
    variable = "BAROGRAPH_PIAF_TOKEN_FILE" if model == "piaf" else "BAROGRAPH_AROME_PI_TOKEN_FILE"
    filename = os.environ.get(variable)
    if not filename:
        pytest.skip(f"set {variable} to an external credential file")
    credential_file = Path(filename).expanduser()
    valid = np.datetime64("now", "s") + np.timedelta64(15, "m")
    kwargs = {"resolution": "0.01"} if model == "arome-pi-hd" else {}
    kind = PiafSource if model == "piaf" else AromePISource
    with kind.latest(valid, token=lambda: credential_file.read_text().strip(), **kwargs) as source:
        sampler = SurfaceEnricher(source, cache_dir=tmp_path)
        first = sampler.enrich(valid, 48.8566, 2.3522)
        repeated = sampler.enrich(valid, 48.8566, 2.3522)
        assert first.valid.all()
        assert all(np.isfinite(v).all() for k, v in first.fields.items() if k != "wind_direction")
        assert repeated.stats["bytes"] == 0
        np.testing.assert_array_equal(first.run, source.run)
        for name, variable_meta in first.variables.items():
            np.testing.assert_array_equal(first[name], repeated[name])
            if variable_meta.temporal != "instant":
                np.testing.assert_array_equal(
                    first.interval_end[name] - first.interval_start[name], source.time_step
                )
