"""Process-wide politeness: concurrent requests are capped per remote host.

Every thread and every source of a process share one limit per host, so a
campaign that opens several sources (or several enrichers) does not multiply
the load on a server. Defaults follow each provider's capacity: object stores
built for parallel reads (Google Cloud Storage, AWS S3) get a generous cap;
Météo-France's distribution, which asks its API users not to download in
parallel, gets a small one.

Asynchronous readers (Open-Meteo) run on one shared event loop per process so
that the same caps apply to them.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

__all__ = ["DEFAULT_HOST_LIMITS", "host_limit", "run_async", "set_host_limit", "slot"]

# host (exact or suffix after a dot) -> max concurrent requests
DEFAULT_HOST_LIMITS: dict[str, int] = {
    "storage.googleapis.com": 32,  # ARCO-ERA5
    "amazonaws.com": 32,  # Open-Meteo archive (and other S3 open data)
    "public-api.meteofrance.fr": 1,  # authenticated WCS, 100 requests/minute
    "api.meteofrance.fr": 1,
    "files.data.gouv.fr": 6,  # Météo-France open data entry point
    "object.data.gouv.fr": 6,
    "meteofrance-pnt.s3.rbx.io.cloud.ovh.net": 6,  # Météo-France object storage
}
_FALLBACK = 16

_lock = threading.Lock()
_overrides: dict[str, int] = {}
_semaphores: dict[str, threading.BoundedSemaphore] = {}
_async_semaphores: dict[str, asyncio.Semaphore] = {}
_loop: asyncio.AbstractEventLoop | None = None


def _host(url: str) -> str:
    return (urlsplit(url).hostname or url).lower()


def host_limit(url_or_host: str) -> int:
    """Concurrent requests allowed to the host of ``url_or_host``."""
    host = _host(url_or_host) if "://" in url_or_host else url_or_host.lower()
    table = {**DEFAULT_HOST_LIMITS, **_overrides}
    if host in table:
        return table[host]
    for suffix, value in table.items():
        if host.endswith("." + suffix):
            return value
    return _FALLBACK


def set_host_limit(host: str, limit: int) -> None:
    """Change the cap for ``host`` (takes effect for hosts not contacted yet)."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    with _lock:
        _overrides[host.lower()] = limit
        _semaphores.pop(host.lower(), None)
        _async_semaphores.pop(host.lower(), None)


@contextmanager
def slot(url: str) -> Iterator[None]:
    """Hold one of the host's request slots (blocking, thread-safe)."""
    host = _host(url)
    with _lock:
        sem = _semaphores.get(host)
        if sem is None:
            sem = _semaphores[host] = threading.BoundedSemaphore(host_limit(host))
    with sem:
        yield


def async_slot(url: str) -> asyncio.Semaphore:
    """The host's semaphore on the shared event loop (call from that loop only)."""
    host = _host(url)
    sem = _async_semaphores.get(host)
    if sem is None:
        sem = _async_semaphores[host] = asyncio.Semaphore(host_limit(host))
    return sem


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` on the process-wide I/O event loop and wait for its result.

    Libraries whose async readers must always be driven from the same loop
    (omfiles) are safe, and per-host async caps are shared process-wide.
    """
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="barograph-met-aio", daemon=True).start()
            _loop = loop
        loop = _loop
    return asyncio.run_coroutine_threadsafe(coro, loop).result()
