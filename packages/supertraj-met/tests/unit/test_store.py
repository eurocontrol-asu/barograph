from __future__ import annotations

import httpx
import pytest

from supertraj_met.store import HttpStore, MissingChunkError

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
