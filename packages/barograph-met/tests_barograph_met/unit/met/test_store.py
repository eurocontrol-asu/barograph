from __future__ import annotations

import httpx
import pytest

from baro.met.store import USER_AGENT, HttpStore, MissingChunkError, list_s3

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
