"""Generate ``reference/variables.md`` from the variable tables of the code.

The mkdocs-gen-files plugin runs this script during the package build; the root
``docs/gen_ref_pages.py`` runs it during the workspace build, because the
monorepo plugin does not run package generators. The accepted-name tables are
read from the constants the sources validate against (``ARPEGE_PARAMS``,
``OPENMETEO_VARIABLES``, ``OPENMETEO_SURFACE_VARIABLES``, ``DEFAULT_FIELDS``),
so the page cannot drift from the code.
"""

from collections.abc import Iterable

import mkdocs_gen_files

from baro.met.arpege import ARPEGE_PARAMS
from baro.met.openmeteo import OPENMETEO_VARIABLES
from baro.met.openmeteo_surface import _DEFAULT_FIELDS as SURFACE_DEFAULTS
from baro.met.openmeteo_surface import OPENMETEO_SURFACE_VARIABLES
from baro.met.sources import DEFAULT_FIELDS


def _names(names: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


era5_defaults = _table(
    ("Output name", "ARCO-ERA5 array"),
    [(f"`{out}`", f"`{array}`") for out, array in DEFAULT_FIELDS.items()],
)
arpege = _table(
    ("Name", "GRIB2 discipline, category, number"),
    [(f"`{name}`", ", ".join(map(str, key))) for name, key in ARPEGE_PARAMS.items()],
)
openmeteo = _table(
    ("Name", "Open-Meteo variable"),
    [(f"`{name}`", f"`{variable}`") for name, (variable, _) in OPENMETEO_VARIABLES.items()],
)
surface = _table(
    ("Name", "Open-Meteo native variables", "Unit"),
    [
        (f"`{name}`", _names(native), meta.unit)
        for name, (native, _, meta) in OPENMETEO_SURFACE_VARIABLES.items()
    ],
)

PAGE = f"""# Variables by source

Every upper-air source takes a `fields=` mapping from **output name** to
**provider variable name**. The output name is the key you read from the result
(`res["q"]`); the provider name selects what is read. Wind (`u`, `v`) and
temperature (`t`) are only the defaults.

```python
from baro.met import Enricher, arco_model_levels

res = Enricher(arco_model_levels(fields={{"q": "specific_humidity"}})).enrich(
    time_utc, lat, lon, altitude_ft=alt_ft
)
res["q"]
```

A non-empty `fields` **replaces** the defaults: the example above returns `q`
only. List `u`, `v` and `t` as well to keep them. An empty mapping or `None`
selects the defaults. `SourceChain` requires its sources to have identical
output fields.

## ERA5 (ARCO-ERA5)

`arco_model_levels(fields=...)` and `arco_pressure_levels(fields=...)` use
these defaults:

{era5_defaults}

The provider names are array names of the ARCO-ERA5 Zarr store being read, and
the code holds no list of them: pass names from the
[ARCO-ERA5 documentation](https://github.com/google-research/arco-era5). The
model-level and pressure-level stores are different stores, so check the name
against the one you use. Names are not validated when the source is created;
a name missing from the store raises `KeyError` (`array '...' not in store`)
when data is read. See the [ERA5 guide](../howto/era5.md).

## ARPEGE open data

`ArpegeSource(fields=...)` accepts these names (default `u`, `v`, `t`):

{arpege}

```python
from baro.met import ArpegeSource

ArpegeSource(fields={{"u": "u", "v": "v", "t": "t", "r": "r"}})
```

An unknown name raises `ValueError` when the source is created. See the
[ARPEGE guide](../howto/arpege.md).

## Open-Meteo archives (ARPEGE, AROME, IFS)

`OpenMeteoSource(model, fields=...)` accepts these names (default `u`, `v`,
`t`):

{openmeteo}

```python
from baro.met import OpenMeteoSource

OpenMeteoSource("ecmwf_ifs025", fields={{"u": "u", "v": "v", "t": "t", "rh": "r"}})
```

An unknown name raises `ValueError` when the source is created. Guides:
[IFS](../howto/ifs.md), [AROME](../howto/arome-pressure.md),
[ARPEGE](../howto/arpege.md).

## Open-Meteo surface archive

`OpenMeteoSurfaceSource(fields=...)` takes the same kind of mapping with these
names. The defaults are {_names(SURFACE_DEFAULTS)};
an unknown name raises `ValueError`, as does an empty mapping.

{surface}

See the [surface weather guide](../howto/surface-weather.md).
"""

# The workspace site runs this script on behalf of the package (see the root
# ``docs/gen_ref_pages.py``) and injects the page path under the package prefix.
output = globals().get("OUTPUT", "reference/variables.md")
with mkdocs_gen_files.open(output, "w") as fd:
    fd.write(PAGE)
