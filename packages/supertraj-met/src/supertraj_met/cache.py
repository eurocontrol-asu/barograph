"""Immutable on-disk cache of regional cubes.

Layout: ``root/<source>/<YYYY>/<MM>/<DD>/<HH>[/run-<YYYYMMDDHH>]/<digest>.npz``
plus a ``.json`` manifest written last (commit marker). Files are written
atomically and never modified, so concurrent processes can share a cache
safely; a per-slot lock (:meth:`CubeCache.claim`) additionally prevents two
processes from downloading the same cube at the same time.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from supertraj_met.base import CubeKey
from supertraj_met.domain import Cube, Domain

__all__ = ["CubeCache"]


def _stamp(t: np.datetime64) -> str:
    return str(np.datetime64(t, "h")).replace("-", "").replace("T", "")


class CubeCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def _key_dir(self, source: str, key: CubeKey) -> Path:
        v = _stamp(key.valid)
        folder = self.root / source / v[:4] / v[4:6] / v[6:8] / v[8:10]
        return folder if key.run is None else folder / f"run-{_stamp(key.run)}"

    def find(
        self,
        source: str,
        key: CubeKey,
        domain: Domain,
        p_range: tuple[float, float],
        fields: list[str],
        level_margin: int = 1,
    ) -> Cube | None:
        """A cached cube answering the request (see :meth:`Cube.serves`)."""
        folder = self._key_dir(source, key)
        if not folder.is_dir():
            return None
        for manifest_path in sorted(folder.glob("*.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not _may_serve(manifest, p_range, fields, level_margin):
                continue
            cube = Cube.load(manifest_path.with_suffix(".npz"))
            if cube.serves(domain, p_range, level_margin):
                return cube
        return None

    @contextmanager
    def claim(self, source: str, key: CubeKey) -> Iterator[None]:
        """Exclusive claim on a cube slot, across threads *and* processes.

        While a process fetches a cube, others asking for the same slot wait,
        then find it in the cache instead of downloading it a second time.
        Implemented with ``flock`` on a per-slot lock file (POSIX filesystems).
        """
        folder = self._key_dir(source, key)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / ".lock", "a+b") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def put(self, cube: Cube) -> Path:
        digest = hashlib.sha256(json.dumps(cube.manifest(), sort_keys=True).encode()).hexdigest()
        key = CubeKey(cube.key.valid, cube.key.run)
        path = self._key_dir(cube.source, key) / f"{digest[:20]}.npz"
        if not path.with_suffix(".json").exists():
            cube.save(path)
        return path


def _may_serve(
    manifest: dict[str, Any], p_range: tuple[float, float], fields: list[str], margin: int
) -> bool:
    """Cheap screening on the manifest before loading arrays."""
    lo, hi = manifest.get("p_range", [0.0, float("inf")])
    return (
        set(fields) <= set(manifest.get("fields", []))
        and float(lo) <= p_range[0]
        and p_range[1] <= float(hi)
        and int(manifest.get("level_margin", 0)) >= margin
    )
