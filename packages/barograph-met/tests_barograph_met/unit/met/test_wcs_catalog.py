"""Metadata contract derived from the official WCS DescribeCoverage example."""

from xml.etree.ElementTree import fromstring

import numpy as np
import pytest

from baro.met.wcs_catalog import parse_description, select_coverage


def description(height: bool = True) -> str:
    labels = "long lat height time" if height else "long lat time"
    units = "deg deg m s" if height else "deg deg s"
    names = labels.split()
    axes = []
    for index, name in enumerate(names):
        vector = ["0"] * len(names)
        vector[index] = "0.025" if name in ("long", "lat") else "1"
        coeff = "10" if name == "height" else "300 600 900"
        axes.append(
            f'<g:GeneralGridAxis><g:offsetVector axisLabels="{labels}" '
            f'uomLabels="{units}">{" ".join(vector)}</g:offsetVector>'
            f"<g:gridAxesSpanned>{name}</g:gridAxesSpanned>"
            f"<g:coefficients>{coeff}</g:coefficients></g:GeneralGridAxis>"
        )
    extra = " 10" if height else ""
    return (
        '<root xmlns:g="urn:g">'
        + "".join(axes)
        + (
            f'<g:EnvelopeWithTimePeriod axisLabels="{labels}">'
            f"<g:lowerCorner>-12 37.5{extra}</g:lowerCorner>"
            f"<g:upperCorner>16 55.4{extra}</g:upperCorner>"
            "</g:EnvelopeWithTimePeriod></root>"
        )
    )


@pytest.mark.parametrize("height", [True, False])
def test_regular_grid_and_exact_leads(height: bool) -> None:
    grid = parse_description(fromstring(description(height)))  # noqa: S314 - local fixture
    assert grid.lon.shape == (1121,)
    assert grid.lat.shape == (717,)
    np.testing.assert_allclose(grid.lon[[0, -1]], [-12, 16])
    np.testing.assert_allclose(grid.lat[[0, -1]], [55.4, 37.5])
    assert grid.lead_seconds == (300, 600, 900)
    assert grid.heights == ((10,) if height else ())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("300 600 900", "300 300 900"),
        ("300 600 900", "300 nan 900"),
        ("300 600 900", "-300 600 900"),
        ("300 600 900", "300.5 600 900"),
        ("height", "probability"),
        ("deg deg m s", "deg deg m h"),
        ("0.025", "0.026"),
        ("0.025", "0"),
        ("16 55.4", "-12 55.4"),
        ("0.025 0 0 0", "0.025 0.025 0 0"),
    ],
)
def test_reject_invalid_metadata(old: str, new: str) -> None:
    with pytest.raises(ValueError):
        parse_description(fromstring(description().replace(old, new)))  # noqa: S314 - local fixture


def test_coverage_selection_preserves_semantics_run_and_interval() -> None:
    run = np.datetime64("2026-10-07T12:00")
    base = "TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE___2026-10-07T12.00.00Z"
    rain = base.replace("TOTAL_PRECIPITATION", "TOTAL_WATER_PRECIPITATION")
    ids = (base + "_PT1H", rain, base, base + "_PT15M")
    assert select_coverage(ids, "precipitation", run, 900) == base + "_PT15M"
    assert select_coverage(ids, "rain", run, 900) == rain
    with pytest.raises(ValueError):
        select_coverage((rain, base + "_PT1H"), "precipitation", run, 900)
    with pytest.raises(ValueError):
        select_coverage(ids, "precipitation", run + np.timedelta64(1, "h"), 900)
    assert select_coverage(ids, "precipitation", run, 900, rain) == rain
    with pytest.raises(ValueError):
        select_coverage(ids, "precipitation", run, 900, "absent")


def test_live_ground_vector_empty_height_slot() -> None:
    # The ground CRS advertises long/lat/time, but real vectors retain a zero
    # height slot. Only this exact server representation is accepted.
    xml = description(False).replace("0.025 0 0<", "0.025 0 0 0<")
    xml = xml.replace("0 0.025 0<", "0 0.025 0 0<").replace("0 0 1<", "0 0 0 1<")
    grid = parse_description(fromstring(xml))  # noqa: S314 - local fixture
    assert grid.lead_seconds == (300, 600, 900)
    assert grid.heights == ()
    with pytest.raises(ValueError):
        parse_description(fromstring(xml.replace("0 0 0 1<", "0 0 2 1<")))  # noqa: S314


def test_piaf_named_rate_is_available_as_native_accumulation() -> None:
    run = np.datetime64("2026-10-07T13:55")
    cid = "TOTAL_PRECIPITATION_RATE__GROUND_OR_WATER_SURFACE___2026-10-07T13.55.00Z_PT5M"
    assert select_coverage((cid,), "precipitation", run, 300) == cid
    with pytest.raises(ValueError):
        select_coverage((cid,), "precipitation", run, 900)
