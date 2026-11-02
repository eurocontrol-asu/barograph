"""Optional xarray backend shape validation and vectorized selection."""

import numpy as np
import pytest


def test_backend_shape_error_and_vectorized_indexing() -> None:
    xr = pytest.importorskip("xarray")
    from baro.met._lazy_backend import lazy_variable

    coords = {"x": np.arange(4), "y": np.arange(3)}
    values = np.arange(12).reshape(4, 3)
    variable = lazy_variable(coords, lambda axes: values[np.ix_(*axes)], "1")
    array = xr.DataArray(variable, coords=coords)
    np.testing.assert_array_equal(
        array.isel(x=xr.DataArray([3, 1], dims="point"), y=xr.DataArray([0, 2], dims="point")),
        [9, 5],
    )
    broken = xr.DataArray(lazy_variable(coords, lambda axes: np.ones((1,)), "1"), coords=coords)
    with pytest.raises(ValueError, match="expected"):
        _ = broken.values
