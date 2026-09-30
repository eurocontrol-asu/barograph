from __future__ import annotations

import numpy as np
import pytest

from baro.met.vertical import STANDARD_SURFACE_PRESSURE, model_levels, pressure_levels


def test_l137_full_levels_match_ecmwf_table() -> None:
    ml = model_levels()
    pf = ml.pressure(STANDARD_SURFACE_PRESSURE)
    # ECMWF L137 table: pf(137) = 1012.0494 hPa, pf(67) ~ 140.8 hPa, pf(1) = 0.01 hPa
    assert pf[136] == pytest.approx(101204.94, abs=0.05)
    assert pf[66] == pytest.approx(14080.0, rel=2e-3)
    assert pf[0] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(pf) > 0)


def test_level_pressure_follows_surface_pressure_near_the_ground_only() -> None:
    ml = model_levels([79, 137])
    shift = ml.pressure(95000.0) - ml.pressure(STANDARD_SURFACE_PRESSURE)
    assert shift[1] == pytest.approx(0.997 * -6325.0, rel=1e-2)
    assert -600.0 < shift[0] < -200.0  # ~ -3.5 hPa at ~250 hPa (b ~ 0.056)


def test_select_bracketing_keeps_levels_around_the_range() -> None:
    ml = model_levels()
    pos = ml.select_bracketing(20000.0, 30000.0, 95000.0, 103000.0, margin=1)
    lo = ml.pressure(95000.0)[pos]
    hi = ml.pressure(103000.0)[pos]
    assert hi[0] < 20000.0 and hi[1] < 20000.0  # a spare level above
    assert lo[-1] > 30000.0 and lo[-2] > 30000.0  # a spare level below
    assert np.all(np.diff(pos) == 1)


def test_select_bracketing_clamps_at_the_model_bottom() -> None:
    ml = model_levels()
    pos = ml.select_bracketing(90000.0, 110000.0, 100000.0, 101000.0)
    assert pos[-1] == 136


def test_pressure_levels_have_no_surface_dependency() -> None:
    pl = pressure_levels([200, 250, 300])
    assert not pl.needs_surface_pressure
    np.testing.assert_array_equal(pl.pressure(70000.0), [20000.0, 25000.0, 30000.0])


@pytest.mark.parametrize("bad", [[250, 200], [0, 5], [137, 138]])
def test_invalid_level_lists_are_rejected(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="increasing"):
        if bad == [250, 200]:
            pressure_levels(bad)
        else:
            model_levels(bad)
