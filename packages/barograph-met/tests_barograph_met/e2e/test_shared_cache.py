"""Two processes enrich the same flight with one shared byte cache."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthetic import write_stores

pytestmark = pytest.mark.e2e

SCRIPT = """
import json, sys
import numpy as np
from baro.met import Enricher, GridSource, model_levels
from baro.met.sources import SurfacePressure
from baro.met.store import FileStore, ZarrStore

ml, pl, cache = sys.argv[1:4]
source = GridSource(
    "e2e", ZarrStore(FileStore(ml)), model_levels(), level_array="hybrid",
    surface_pressure=SurfacePressure(ZarrStore(FileStore(pl))),
)
n = 200
t = np.datetime64("2020-01-01T00:10") + (np.linspace(0, 100, n) * 60).astype("timedelta64[s]")
res = Enricher(source, cache_dir=cache).enrich(
    t, np.linspace(42, 48, n), np.linspace(21, 33, n), pressure_pa=np.linspace(85000, 22000, n)
)
print(json.dumps({"bytes": res.stats["bytes"], "ok": int((res.flags == 0).sum())}))
"""


def _run(ml: Path, pl: Path, cache: Path, n: int) -> list[dict[str, int]]:
    procs = [
        subprocess.Popen(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-c", SCRIPT, str(ml), str(pl), str(cache)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(n)
    ]
    outs = [json.loads(p.communicate(timeout=120)[0]) for p in procs]
    assert all(p.returncode == 0 for p in procs)
    return outs


def test_processes_share_the_cache_without_duplicate_downloads(tmp_path: Path) -> None:
    ml, pl = write_stores(tmp_path / "stores", n_hours=3)
    (alone,) = _run(ml, pl, tmp_path / "alone", 1)
    pair = _run(ml, pl, tmp_path / "shared", 2)
    assert all(o["ok"] == 200 for o in [alone, *pair])
    assert sum(o["bytes"] for o in pair) == alone["bytes"]  # every byte once in total
    (later,) = _run(ml, pl, tmp_path / "shared", 1)
    assert later["bytes"] == 0
