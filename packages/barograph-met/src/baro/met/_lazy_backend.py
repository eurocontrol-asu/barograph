"""Optional xarray backend loaded only when lazy datasets are constructed."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from baro.met.base import DataUnavailableError

type Axes = tuple[NDArray[Any], ...]
type Reader = Callable[[Axes], NDArray[np.float64]]


@cache
def _backend_type() -> Any:
    backend = import_module("xarray.backends").BackendArray
    indexing = import_module("xarray.core.indexing")

    class LazyField(backend):  # type: ignore[misc, valid-type]
        """Delegate basic slices; xarray handles outer/vectorized indexing."""

        def __init__(self, coordinates: Sequence[NDArray[Any]], reader: Reader) -> None:
            self.coordinates = tuple(np.array(a, copy=True) for a in coordinates)
            for axis in self.coordinates:
                axis.setflags(write=False)
            self.shape = tuple(len(a) for a in self.coordinates)
            self.dtype = np.dtype("float64")
            self.reader = reader

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            # Backend is an immutable read handle; retain its shared bounded cache.
            return self

        def __getitem__(self, key: Any) -> Any:
            return indexing.explicit_indexing_adapter(
                key, self.shape, indexing.IndexingSupport.BASIC, self._read
            )

        def _read(self, key: Any) -> NDArray[np.float64]:
            ids = [np.atleast_1d(np.arange(n)[k]) for n, k in zip(self.shape, key, strict=True)]
            shape = tuple(len(a) for a in ids)
            if any(n == 0 for n in shape):
                out = np.empty(shape, dtype=self.dtype)
            else:
                axes = tuple(a[i] for a, i in zip(self.coordinates, ids, strict=True))
                out = self.reader(axes)
            if out.shape != shape:
                raise ValueError(f"Lazy weather returned shape {out.shape}, expected {shape}")
            if not np.isfinite(out).all():
                raise DataUnavailableError("Nonfinite meteorology; refusing silent extinction")
            squeeze = tuple(i for i, k in enumerate(key) if isinstance(k, int | np.integer))
            return np.squeeze(out, axis=squeeze)

    return LazyField


def lazy_variable(coordinates: dict[str, NDArray[Any]], reader: Reader, units: str) -> Any:
    """Create an xarray variable without evaluating or caching its field data."""
    xr = import_module("xarray")
    indexing = import_module("xarray.core.indexing")
    backend = _backend_type()(tuple(coordinates.values()), reader)
    return xr.Variable(
        tuple(coordinates), indexing.LazilyIndexedArray(backend), attrs={"units": units}
    )
