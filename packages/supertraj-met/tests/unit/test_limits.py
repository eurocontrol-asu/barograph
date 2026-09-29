from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from supertraj_met.limits import host_limit, set_host_limit, slot
from supertraj_met.store import HttpStore


def test_known_hosts_get_polite_defaults() -> None:
    assert host_limit("https://meteofrance-pnt.s3.rbx.io.cloud.ovh.net/pnt/x") == 6
    assert host_limit("https://openmeteo.s3.amazonaws.com/data") == 32  # suffix match
    assert host_limit("https://unknown.example/x") == 16


def test_slot_caps_concurrency_across_threads() -> None:
    set_host_limit("cap.example", 3)
    active, peak = 0, 0
    guard = threading.Lock()

    def work(_: int) -> None:
        nonlocal active, peak
        with slot("https://cap.example/obj"):
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with guard:
                active -= 1

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(work, range(60)))
    assert peak == 3


def test_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        set_host_limit("x.example", 0)


def test_throttling_is_counted_and_retry_after_honoured() -> None:
    answers = iter(
        [httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, content=b"ok")]
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: next(answers)))
    store = HttpStore("https://busy.example", client=client, backoff=0.0)
    assert store.get("k") == b"ok"
    assert store.stats.snapshot()["throttled"] == 1
