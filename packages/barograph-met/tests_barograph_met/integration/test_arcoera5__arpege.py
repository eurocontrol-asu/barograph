"""FastMeteo-compatible façade on local mirrors of both sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baro.met.compat import ArcoEra5, Arpege
from baro.met.sources import ARCO_MODEL_LEVEL, ARCO_PRESSURE_LEVEL
from synthetic import ARPEGE_RUN, EPOCH, arpege_truth, truth, write_stores

pytestmark = pytest.mark.integration
FEATURES = ["u_component_of_wind", "v_component_of_wind", "temperature"]


@pytest.fixture(scope="module")
def arco(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("arco")
    write_stores(root, n_hours=3, names=(ARCO_MODEL_LEVEL, ARCO_PRESSURE_LEVEL))
    return root


def _frame(start: np.datetime64, lat0: float, lon0: float) -> pd.DataFrame:
    n = 50
    t = start + (np.linspace(5, 95, n) * 60).astype("timedelta64[s]")
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(t).tz_localize("UTC"),
            "latitude": np.linspace(lat0, lat0 + 2.0, n),
            "longitude": np.linspace(lon0, lon0 + 3.0, n),
            "altitude": np.linspace(20000.0, 36000.0, n),
            "icao24": "abc123",
        },
        index=pd.RangeIndex(100, 100 + n),
    )


def test_arcoera5_interpolate_keeps_the_fastmeteo_contract(arco: Path) -> None:
    df = _frame(EPOCH, 44.0, 22.0)
    out = ArcoEra5(model_levels=137, features=FEATURES, base_url=str(arco)).interpolate(df)
    assert list(out.columns) == [*df.columns, *FEATURES]
    assert out.index.equals(df.index)
    from baro.met.isa import pressure_from_altitude_ft

    p = pressure_from_altitude_ft(df.altitude)
    hours = (df.timestamp.dt.tz_localize(None).to_numpy() - EPOCH) / np.timedelta64(1, "h")
    expected = truth("t", df.latitude.to_numpy(), df.longitude.to_numpy(), p, hours.astype(float))
    np.testing.assert_allclose(out.temperature, expected, atol=2e-3)


def test_global_mode_cache_serves_any_later_flight(arco: Path, tmp_path: Path) -> None:
    def engine() -> ArcoEra5:
        return ArcoEra5(
            str(tmp_path), model_levels=137, features=FEATURES, base_url=str(arco), mode="global"
        )

    engine().interpolate(_frame(EPOCH, 44.0, 22.0))
    second = engine()
    out = second.interpolate(_frame(EPOCH, 10.0, 100.0))  # elsewhere on the globe
    assert second.last_result is not None
    assert second.last_result.stats["bytes"] == 0 and second.last_result.stats["cache_bytes"] > 0
    assert out.temperature.notna().all()


def test_arpege_compat_on_a_mirror(tmp_path_factory: pytest.TempPathFactory) -> None:
    pytest.importorskip("eccodes")
    from synthetic import write_arpege_mirror

    mirror = write_arpege_mirror(tmp_path_factory.mktemp("arpege"))
    df = _frame(ARPEGE_RUN + np.timedelta64(1, "h"), 45.0, -1.0)
    with pytest.warns(UserWarning, match="time_range"):
        fm = Arpege(
            features=["u", "v", "t"], resolution="01", time_range="000H012H", base_url=str(mirror)
        )
    out = fm.interpolate(df)
    from baro.met.isa import pressure_from_altitude_ft

    p = pressure_from_altitude_ft(df.altitude)
    hours = (
        (df.timestamp.dt.tz_localize(None).to_numpy() - EPOCH) / np.timedelta64(1, "h")
    ).astype(float)
    ok = (p >= 20000) & (p <= 50000)
    expected = arpege_truth("u", df.latitude.to_numpy(), df.longitude.to_numpy(), p, hours)
    np.testing.assert_allclose(out.u[ok], expected[ok], atol=0.2)  # linear in p (FastMeteo default)
