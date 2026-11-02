from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import httpx
import numpy as np
import pytest

from baro.met import store as module
from baro.met.blosc import decode_run
from baro.met.store import USER_AGENT, HttpStore, MissingChunkError, ZarrStore, list_s3
from synthetic import ArraySpec, memory_zarr

TARGET = "https://bucket.example/pnt"


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://entry.example/pnt/"):
            key = url.removeprefix("https://entry.example/pnt/")
            return httpx.Response(302, headers={"location": f"{TARGET}/{key.replace(':', '%3A')}"})
        if url.startswith(TARGET) and "missing" not in url:
            body = b"0123456789"
            if "range" in request.headers:
                a, b = request.headers["range"].removeprefix("bytes=").split("-")
                return httpx.Response(206, content=body[int(a) : int(b) + 1])
            if request.method == "HEAD":
                return httpx.Response(200, headers={"content-length": str(len(body))})
            return httpx.Response(200, content=body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_permanent_redirect_is_learnt_once_even_with_encoded_paths() -> None:
    client = httpx.Client(transport=_transport(), follow_redirects=True)
    store = HttpStore("https://entry.example/pnt", client=client)
    key = "2026-09-29T00:00:00Z/arpege/f.grib2"
    assert store.get(key, 2, 5) == b"234"
    assert store._base == TARGET
    assert store.get(key, 0, 2) == b"01"  # served directly by the target
    assert store.size(key) == 10


def test_missing_objects_raise() -> None:
    store = HttpStore(TARGET, client=httpx.Client(transport=_transport(), follow_redirects=True))
    with pytest.raises(MissingChunkError):
        store.get("missing/object")


def test_server_errors_are_retried_and_counted() -> None:
    answers = iter([httpx.Response(503), httpx.Response(200, content=b"ok")])
    client = httpx.Client(transport=httpx.MockTransport(lambda _r: next(answers)))
    store = HttpStore("https://busy.example", client=client, backoff=0.0)
    assert store.get("k") == b"ok" and store.stats.snapshot()["throttled"] == 1


def test_transport_errors_are_retried_then_raised() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("down", request=request)

    store = HttpStore(
        "https://down.example",
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
        retries=2,
        backoff=0.0,
    )
    with pytest.raises(httpx.ConnectError):
        store.get("k")
    assert calls["n"] == 3


def test_ignored_range_request_is_an_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"x"))
    )
    with pytest.raises(RuntimeError, match="ignored the range"):
        HttpStore("https://x.example", client=client).get("k", 0, 1)


def _answers(pages: dict[str, httpx.Response], seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return pages[str(request.url)]

    return httpx.MockTransport(handler)


def test_a_signed_redirect_is_not_learnt_and_a_stale_one_is_forgotten() -> None:
    seen: list[httpx.Request] = []
    signed = "https://cdn.example/x/f?sig=abc"
    pages = {
        "https://entry.example/x/f": httpx.Response(302, headers={"location": signed}),
        signed: httpx.Response(200, content=b"data"),
    }
    store = HttpStore(
        "https://entry.example/x",
        client=httpx.Client(transport=_answers(pages, seen), follow_redirects=True),
    )
    assert store.get("f") == b"data" and store._base == "https://entry.example/x"
    store._base = "https://gone.example/x"  # a learnt host that no longer serves the files
    pages["https://gone.example/x/f"] = httpx.Response(404)
    assert store.get("f") == b"data" and store._base == "https://entry.example/x"


def test_requests_identify_the_library() -> None:
    assert HttpStore("https://h.example").client.headers["user-agent"] == USER_AGENT
    assert USER_AGENT.startswith("barograph-met/")


def test_s3_listings_follow_continuation_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = (
        "<ListBucketResult><IsTruncated>true</IsTruncated>"
        "<NextContinuationToken>t2</NextContinuationToken>"
        "<CommonPrefixes><Prefix>data/a/</Prefix></CommonPrefixes></ListBucketResult>"
    )
    page2 = (
        "<ListBucketResult><Prefix>data/</Prefix><IsTruncated>false</IsTruncated>"
        "<CommonPrefixes><Prefix>data/b/</Prefix></CommonPrefixes></ListBucketResult>"
    )
    calls: list[dict[str, str]] = []

    def get(url: str, *, params: dict[str, str], **_: object) -> httpx.Response:
        calls.append(dict(params))
        body = page2 if "continuation-token" in params else page1
        return httpx.Response(200, text=body, request=httpx.Request("GET", url))

    monkeypatch.setattr("baro.met.store.httpx.get", get)
    assert list_s3("https://b.example", "data/", dirs=True) == ["data/a/", "data/b/"]
    assert calls[1]["continuation-token"] == "t2" and calls[0]["delimiter"] == "/"


BLOCK_BYTES = 65536


def reader(budget: int = BLOCK_BYTES) -> tuple[ZarrStore, Any]:
    data = np.arange(2 * 2 * 96 * 256, dtype="<f4").reshape(2, 2, 96, 256)
    zarr = ZarrStore(
        memory_zarr([ArraySpec("t", data, (1, 2, 96, 256), ("t", "l", "y", "x"))]),
        full_read_ratio=1.1,
    )
    zarr.configure_decoded_cache(budget, immutable=lambda key: True)
    return zarr, data


def test_overlap_exact_run_and_level_chunk_isolation() -> None:
    zarr, data = reader(BLOCK_BYTES * 4)
    np.testing.assert_array_equal(zarr.read_rows("t", [(0, 0)], slice(0, 2))[0], data[0, 0, :2])
    before = zarr.store.stats.bytes
    np.testing.assert_array_equal(zarr.read_rows("t", [(0, 0)], slice(1, 3))[0], data[0, 0, 1:3])
    assert zarr.store.stats.bytes == before
    assert zarr.decoded_cache_info()["hits"] == 1
    for lead in ((0, 1), (1, 0)):
        np.testing.assert_array_equal(zarr.read_rows("t", [lead], slice(0, 2))[0], data[*lead, :2])
    assert zarr.decoded_cache_info()["entries"] == 3


def test_eviction_oversize_disabled_and_clear() -> None:
    zarr, _ = reader()
    zarr.read_rows("t", [(0, 0)], slice(0, 2))
    zarr.read_rows("t", [(0, 1)], slice(0, 2))
    assert zarr.decoded_cache_info()["evictions"] == 1
    assert zarr.decoded_cache_info()["bytes"] == BLOCK_BYTES
    zarr.clear_decoded_cache()
    assert zarr.decoded_cache_info()["entries"] == 0
    for budget in (0, BLOCK_BYTES - 1):
        zarr.configure_decoded_cache(budget, immutable=lambda key: True)
        zarr.read_rows("t", [(0, 0)], slice(0, 2))
        assert zarr.decoded_cache_info()["bytes"] == 0
    with pytest.raises(ValueError, match="nonnegative"):
        zarr.configure_decoded_cache(-1, immutable=lambda key: True)


def test_mutable_keys_are_not_retained() -> None:
    zarr, _ = reader()
    zarr.configure_decoded_cache(BLOCK_BYTES, immutable=lambda key: False)
    for _ in range(2):
        zarr.read_rows("t", [(0, 0)], slice(0, 2))
    assert zarr.decoded_cache_info()["entries"] == zarr.decoded_cache_info()["hits"] == 0


@pytest.mark.parametrize("same_key", [False, True])
def test_concurrent_loads_do_not_serialize_or_double_account(
    monkeypatch: pytest.MonkeyPatch, same_key: bool
) -> None:
    zarr, data = reader(BLOCK_BYTES * 2)
    barrier = Barrier(2)
    decode = decode_run

    def synchronized(*args: Any) -> bytes:
        barrier.wait(timeout=5)
        return decode(*args)

    monkeypatch.setattr(module, "decode_run", synchronized)
    leads = [(0, 0), (0, 0) if same_key else (1, 0)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda lead: zarr.read_rows("t", [lead], slice(0, 2)), leads))
    for lead, result in zip(leads, results, strict=True):
        np.testing.assert_array_equal(result[0], data[*lead, :2])
    count = 1 if same_key else 2
    assert zarr.decoded_cache_info()["entries"] == count
    assert zarr.decoded_cache_info()["bytes"] == count * BLOCK_BYTES


def test_clear_during_fill_discards_inflight_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    zarr, _ = reader()
    decode = decode_run

    def clearing(*args: Any) -> bytes:
        zarr.clear_decoded_cache()
        return decode(*args)

    monkeypatch.setattr(module, "decode_run", clearing)
    zarr.read_rows("t", [(0, 0)], slice(0, 2))
    assert zarr.decoded_cache_info()["entries"] == 0


def test_full_chunk_fallback_does_not_retain_decoded_runs() -> None:
    zarr, data = reader(BLOCK_BYTES * 8)
    zarr.full_read_ratio = 0.0
    np.testing.assert_array_equal(zarr.read_rows("t", [(0, 0)], slice(0, 2))[0], data[0, 0, :2])
    assert zarr.decoded_cache_info()["entries"] == 0


def test_failed_decoded_run_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    zarr, data = reader()

    def broken(*args: Any) -> bytes:
        raise ValueError("damaged payload")

    monkeypatch.setattr(module, "decode_run", broken)
    with pytest.raises(ValueError, match="damaged payload"):
        zarr.read_rows("t", [(0, 0)], slice(0, 2))
    assert zarr.decoded_cache_info()["entries"] == 0
    monkeypatch.setattr(module, "decode_run", decode_run)
    np.testing.assert_array_equal(zarr.read_rows("t", [(0, 0)], slice(0, 2))[0], data[0, 0, :2])
    assert zarr.decoded_cache_info()["entries"] == 1
