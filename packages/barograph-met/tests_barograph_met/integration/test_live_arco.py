"""Opt-in checks against the public ARCO-ERA5 bucket (run with ``-m live``).

They transfer a few tens of MB: one surface-pressure chunk (2.4 MB) and the
cruise-level bands of one short flight segment on both stores.
"""

from __future__ import annotations

import numpy as np
import pytest

from baro.met import Enricher, Flag, arco_model_levels, arco_pressure_levels
from baro.met.sources import ARCO_BASE_URL, ARCO_PRESSURE_LEVEL
from baro.met.store import HttpStore, ZarrStore

pytestmark = [pytest.mark.integration, pytest.mark.live]

HOUR = np.datetime64("2020-01-01T10", "h")


def test_partial_read_equals_full_chunk_on_the_real_store() -> None:
    z = ZarrStore(HttpStore(f"{ARCO_BASE_URL}/{ARCO_PRESSURE_LEVEL}"))
    t = int((HOUR - np.datetime64("1900-01-01T00", "h")) / np.timedelta64(1, "h"))
    band = z.read_rows("surface_pressure", [(t,)], slice(150, 250))[0]
    full = z._decode_full(z.array("surface_pressure"), (t, 0, 0))
    np.testing.assert_array_equal(band, full[0, 150:250])


def test_model_and_pressure_levels_agree_at_cruise() -> None:
    minutes = np.arange(0, 30, 0.5)
    time = HOUR + (minutes * 60).astype("timedelta64[s]")
    lat = np.linspace(48.7, 45.0, minutes.size)
    lon = np.linspace(2.4, 4.5, minutes.size)
    alt = np.full(minutes.size, 35000.0)
    ml = Enricher(arco_model_levels()).enrich(time, lat, lon, altitude_ft=alt)
    pl = Enricher(arco_pressure_levels()).enrich(time, lat, lon, altitude_ft=alt)
    print("ML stats", ml.stats, "PL stats", pl.stats)  # noqa: T201
    assert np.all(ml.flags == Flag.OK) and np.all(pl.flags == Flag.OK)
    assert np.all((ml["t"] > 200) & (ml["t"] < 250))
    assert np.abs(ml["t"] - pl["t"]).max() < 2.0
    assert np.hypot(ml["u"] - pl["u"], ml["v"] - pl["v"]).max() < 8.0
