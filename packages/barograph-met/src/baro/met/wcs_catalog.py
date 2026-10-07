"""Strict geographic and discrete-axis metadata for Météo-France WCS."""

from dataclasses import dataclass
from xml.etree.ElementTree import Element

import numpy as np
from numpy.typing import NDArray

_MAX_AXIS_SIZE = 1_000_000


@dataclass(frozen=True)
class CoverageGrid:
    """Advertised grid centres and exact available forecast leads in seconds."""

    lat: NDArray[np.float64]
    lon: NDArray[np.float64]
    lead_seconds: tuple[int, ...]
    heights: tuple[float, ...]


def _one(root: Element, name: str) -> Element:
    nodes = list(root.iterfind(f".//{{*}}{name}"))
    if len(nodes) != 1:
        raise ValueError(f"WCS description requires exactly one {name}")
    return nodes[0]


def _numbers(text: str | None) -> NDArray[np.float64]:
    values = np.array((text or "").split(), dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("WCS coordinates must be nonempty and finite")
    return values


def _spatial(lower: float, upper: float, step: float) -> NDArray[np.float64]:
    step = abs(step)
    if step == 0 or upper <= lower:
        raise ValueError("WCS geographic grid must have positive extent and resolution")
    count = (upper - lower) / step
    if not np.isclose(count, round(count), atol=1e-6, rtol=0) or count > _MAX_AXIS_SIZE:
        raise ValueError("WCS geographic bounds do not describe a regular grid")
    return np.linspace(lower, upper, round(count) + 1)


def _vector(vector: Element) -> tuple[list[str], NDArray[np.float64]]:
    labels = vector.attrib.get("axisLabels", "").split()
    values = _numbers(vector.text)
    # Live Météo-France ground products retain an empty height coordinate slot
    # in their numeric vectors while advertising a three-dimensional CRS.
    if labels == ["long", "lat", "time"] and len(values) == 4 and values[2] == 0:  # noqa: PLR2004
        values = np.delete(values, 2)
    return labels, values


def _axis_step(axis: Element, name: str) -> NDArray[np.float64]:
    """Validate the coordinate system and isolate one nonzero axis component."""
    vector = _one(axis, "offsetVector")
    labels, values = _vector(vector)
    units = vector.attrib.get("uomLabels", "").split()
    if len(set(labels)) != len(labels) or set(labels) not in (
        {"long", "lat", "time"},
        {"long", "lat", "height", "time"},
    ):
        raise ValueError("Unsupported WCS coordinate dimensions")
    expected = {"long": "deg", "lat": "deg", "time": "s", "height": "m"}
    if units != [expected[label] for label in labels] or len(values) != len(labels):
        raise ValueError("Unsupported WCS axis units or offset vector")
    index = labels.index(name)
    if np.any(np.delete(values, index) != 0) or values[index] == 0:
        raise ValueError("WCS rotated or degenerate grids are unsupported")
    return values[index : index + 1]


def _axes(root: Element) -> dict[str, tuple[Element, NDArray[np.float64]]]:
    result: dict[str, tuple[Element, NDArray[np.float64]]] = {}
    for axis in root.iterfind(".//{*}GeneralGridAxis"):
        name = (_one(axis, "gridAxesSpanned").text or "").strip()
        if name not in {"long", "lat", "time", "height"} or name in result:
            raise ValueError(f"Unsupported or repeated WCS axis {name!r}")
        result[name] = (axis, _axis_step(axis, name))
    if set(result) not in ({"long", "lat", "time"}, {"long", "lat", "time", "height"}):
        raise ValueError("WCS description is missing required axes")
    return result


def _discrete(axis: Element) -> NDArray[np.float64]:
    values = _numbers(_one(axis, "coefficients").text)
    if np.any(np.diff(values) <= 0):
        raise ValueError("WCS discrete coordinates must be strictly increasing")
    return values


def parse_description(root: Element) -> CoverageGrid:
    """Parse a single regular lon/lat surface coverage, rejecting unknown axes.

    Météo-France documents time coefficients as forecast lead seconds and height
    coefficients as metres. No interpolation of missing leads is implied here.
    """
    axes = _axes(root)
    envelope = _one(root, "EnvelopeWithTimePeriod")
    labels = envelope.attrib.get("axisLabels", "").split()
    if labels not in (["long", "lat", "time"], ["long", "lat", "height", "time"]):
        raise ValueError("Unsupported WCS envelope coordinate order")
    lower = _numbers(_one(envelope, "lowerCorner").text)
    upper = _numbers(_one(envelope, "upperCorner").text)
    if len(lower) != len(labels) - 1 or len(upper) != len(lower):
        raise ValueError("WCS envelope coordinate dimensions do not match")
    lon = _spatial(lower[0], upper[0], axes["long"][1][0])
    lat = _spatial(lower[1], upper[1], axes["lat"][1][0])[::-1].copy()
    leads = _discrete(axes["time"][0])
    if np.any(leads < 0) or np.any(leads != np.floor(leads)):
        raise ValueError("WCS forecast leads must be nonnegative integer seconds")
    heights = tuple(float(v) for v in _discrete(axes["height"][0])) if "height" in axes else ()
    return CoverageGrid(lat, lon, tuple(int(v) for v in leads), heights)


_FIELD_NAMES = {
    "u10": ("U_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",),
    "v10": ("V_COMPONENT_OF_WIND__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",),
    "t2m": ("TEMPERATURE__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",),
    "gusts": (
        "WIND_SPEED_GUST_15MIN__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
        "WIND_SPEED_GUST__SPECIFIC_HEIGHT_LEVEL_ABOVE_GROUND",
    ),
    "precipitation": (
        "TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE",
        "TOTAL_PRECIPITATION_RATE__GROUND_OR_WATER_SURFACE",
    ),
    "rain": ("TOTAL_WATER_PRECIPITATION__GROUND_OR_WATER_SURFACE",),
}


def _coverage_candidates(field: str, stamp: str, step_seconds: int) -> list[str]:
    """Order exact field identifiers by native interval, then cumulative fallback."""
    if field not in _FIELD_NAMES or step_seconds <= 0:
        raise ValueError("Unsupported WCS field or temporal step")
    suffix = f"_PT{step_seconds // 60}M" if step_seconds % 60 == 0 else f"_PT{step_seconds}S"
    names = [
        name for name in _FIELD_NAMES[field] if "_15MIN__" not in name or step_seconds == 15 * 60
    ]
    return [name + "___" + stamp + ending for ending in (suffix, "") for name in names]


def select_coverage(
    ids: tuple[str, ...],
    field: str,
    run: np.datetime64,
    step_seconds: int,
    override: str | None = None,
) -> str:
    """Select exact semantic names and run, preferring native interval products.

    Plain names may represent cumulative precipitation; callers must inspect GRIB
    statistical metadata. No meaning or interval is inferred from substrings.
    Explicit overrides still require the requested run and catalog membership.
    """
    stamp = str(np.datetime_as_string(run, unit="s")).replace(":", ".") + "Z"
    if override is not None:
        if override not in ids or override.split("___")[-1].split("_")[0] != stamp:
            raise ValueError("WCS coverage override must exist and match the requested run")
        return override
    for candidate in _coverage_candidates(field, stamp, step_seconds):
        if candidate in ids:
            return candidate
    raise ValueError(f"No supported WCS coverage for {field} at {stamp}")
