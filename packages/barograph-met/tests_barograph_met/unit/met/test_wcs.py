"""WCS wire contract, response validation and immutable cache isolation."""

from pathlib import Path

import httpx
import numpy as np
import pytest

from baro.met.rangecache import RangeCache
from baro.met.wcs import WCSAuthenticationError, WCSClient, WCSRequestError, coverage_run

ENDPOINT = "https://weather.example/wcs/model"
COVERAGE = "WIND_SPEED__HEIGHT___2026-10-07T12.00.00Z_PT15M"
GRIB = b"GRIB\x00\x00\x00\x02" + (20).to_bytes(8, "big") + b"7777"


def test_run_and_catalogue() -> None:
    xml = (
        f'<w:Capabilities xmlns:w="urn:wcs"><w:CoverageId>{COVERAGE}</w:CoverageId>'
        "</w:Capabilities>"
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=xml)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with WCSClient(ENDPOINT, lambda: "secret", client=client) as source:
        assert source.capabilities() == (COVERAGE,)
        assert source.describe(COVERAGE).tag.endswith("Capabilities")
    assert not client.is_closed
    assert requests[0].headers["Authorization"] == "Bearer secret"
    assert requests[0].url.path.endswith("/GetCapabilities")
    assert coverage_run(COVERAGE) == np.datetime64("2026-10-07T12:00", "ns")
    assert coverage_run("latest") is None
    client.close()


def test_subsets_cache_and_credentials(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=GRIB)

    cache = RangeCache(tmp_path)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = WCSClient(ENDPOINT, "secret", client=client, cache=cache)
        subsets = ["lat(45,46)", "long(2,3)", "time(900)", "height(10)"]
        assert source.coverage(COVERAGE, subsets=subsets) == GRIB
        assert source.coverage(COVERAGE, subsets=subsets[::-1]) == GRIB
        assert len(requests) == 1
        assert sorted(requests[0].url.params.get_list("subset")) == sorted(subsets)
        assert source.stats.bytes == len(GRIB)
        source.coverage(COVERAGE, subsets=["lat(46,47)"])
        assert len(requests) == 2
        other = WCSClient(ENDPOINT, "different", client=client, cache=cache, auth="apikey")
        other.coverage(COVERAGE, subsets=subsets)
        assert other.stats.requests == 0
        other.coverage("latest", subsets=subsets)
        other.coverage("latest", subsets=subsets)
        assert requests[-1].headers["apikey"] == "different"
        assert len(requests) == 4
    assert all(b"secret" not in path.read_bytes() for path in tmp_path.rglob("key.txt"))


@pytest.mark.parametrize(
    "status,error",
    [
        (401, WCSAuthenticationError),
        (403, WCSAuthenticationError),
        (429, WCSRequestError),
        (500, WCSRequestError),
        (302, WCSRequestError),
    ],
)
def test_http_errors(status: int, error: type[WCSRequestError]) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        source = WCSClient(ENDPOINT, "secret", client=client, retries=0)
        with pytest.raises(error, match="HTTP"):
            source.capabilities()


@pytest.mark.parametrize("data", [b"", b"<ExceptionReport/>", GRIB[:-1], b"GRIB" + b"\0" * 20])
def test_invalid_grib_never_cached(tmp_path: Path, data: bytes) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
    ) as client:
        source = WCSClient(ENDPOINT, "secret", client=client, cache=RangeCache(tmp_path))
        for _ in range(2):
            with pytest.raises(WCSRequestError):
                source.coverage(COVERAGE, subsets=[])
        assert source.stats.requests == 2
    assert not list(tmp_path.rglob("*.bin"))


def test_validation_before_cache(tmp_path: Path) -> None:
    def reject(data: bytes) -> None:
        assert data == GRIB
        raise ValueError("incorrect run")

    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=GRIB))
    ) as client:
        source = WCSClient(ENDPOINT, "secret", client=client, cache=RangeCache(tmp_path))
        with pytest.raises(ValueError, match="incorrect run"):
            source.coverage(COVERAGE, subsets=[], validate=reject)
    assert not list(tmp_path.rglob("*.bin"))


@pytest.mark.parametrize(
    "xml", ["bad", "<!DOCTYPE x><x/>", "<ExceptionReport><Exception/></ExceptionReport>"]
)
def test_invalid_xml(xml: str) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=xml))
    ) as client:
        with pytest.raises(WCSRequestError):
            WCSClient(ENDPOINT, "secret", client=client).capabilities()


def test_bad_config_and_empty_token() -> None:
    with pytest.raises(ValueError):
        WCSClient("http://example.org", "secret")
    with pytest.raises(ValueError):
        WCSClient(ENDPOINT, "secret", auth="query")
    with WCSClient(ENDPOINT, "") as source, pytest.raises(WCSAuthenticationError):
        source.capabilities()


def test_transport_error_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("possibly sensitive response", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(WCSRequestError, match="transport request failed"):
            WCSClient(ENDPOINT, "secret", client=client).capabilities()


def test_redirect_never_forwards_credentials() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example/collect"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        with pytest.raises(WCSRequestError, match="302"):
            WCSClient(ENDPOINT, "secret", client=client, auth="apikey").capabilities()
    assert len(seen) == 1
    assert seen[0].url.host == "weather.example"


@pytest.mark.parametrize("token", ["secret\nvalue", "secret value", "clé"])
def test_invalid_token_never_exposed(token: str) -> None:
    with WCSClient(ENDPOINT, token) as source:
        with pytest.raises(WCSAuthenticationError) as error:
            source.capabilities()
    assert token not in str(error.value)


def test_retry_after_and_token_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    seen: list[httpx.Request] = []
    monkeypatch.setattr("baro.met.wcs.time.sleep", delays.append)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 3:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, content=GRIB)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = WCSClient(ENDPOINT, lambda: f"token{len(seen)}", client=client)
        assert source.coverage(COVERAGE, subsets=[]) == GRIB
        assert source.stats.throttled == 2
    assert delays == [0.25, 0.25]
    assert [r.headers["Authorization"] for r in seen] == [
        "Bearer token0",
        "Bearer token1",
        "Bearer token2",
    ]


def test_retry_after_long_does_not_retry_early(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("baro.met.wcs.time.sleep", delays.append)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(429, headers={"Retry-After": "120"}))
    ) as client:
        source = WCSClient(ENDPOINT, "secret", client=client)
        with pytest.raises(WCSRequestError, match="429"):
            source.capabilities()
        assert source.stats.requests == 1
    assert not delays


@pytest.mark.parametrize(
    "header,delay", [("bad", 1.0), ("NaN", 1.0), ("Wed, 07 Oct 2026 12:00:05 GMT", 5.0)]
)
def test_retry_after_date_or_fallback(
    header: str, delay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("baro.met.wcs.time.time", lambda: 1791374400.0)
    monkeypatch.setattr("baro.met.wcs.time.sleep", delays.append)

    def handler(request: httpx.Request) -> httpx.Response:
        if not delays:
            return httpx.Response(503, headers={"Retry-After": header})
        return httpx.Response(200, content=GRIB)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = WCSClient(ENDPOINT, "secret", client=client)
        assert source.coverage(COVERAGE, subsets=[]) == GRIB
        assert source.stats.requests == 2
        assert source.stats.throttled == 1
    assert delays == [delay]


def test_coverage_validation_once_per_call(tmp_path: Path) -> None:
    validated: list[bytes] = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=GRIB))
    ) as client:
        source = WCSClient(ENDPOINT, "secret", client=client, cache=RangeCache(tmp_path))
        for count in (1, 2):
            assert source.coverage(COVERAGE, subsets=[], validate=validated.append) == GRIB
            assert validated == [GRIB] * count
        assert source.stats.requests == 1
