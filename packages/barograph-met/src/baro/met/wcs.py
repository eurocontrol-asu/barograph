"""Authenticated, geographically targeted Météo-France WCS 2.0.1 requests.

Credentials stay in headers and never enter cache keys. Only explicit, run-stamped
GRIB coverages are persisted; the changing capabilities catalogue is not cached.
Tokens may be callables so applications can manage OAuth renewal themselves.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx
import numpy as np

from baro.met.rangecache import RangeCache, namespace_for
from baro.met.store import TransferStats

_RUN = re.compile(r"___(\d{4}-\d{2}-\d{2}T\d{2})\.(\d{2})\.(\d{2})Z(?:_|$)")


class WCSRequestError(RuntimeError):
    """An HTTP or OGC service error, excluding missing local credentials."""


class WCSAuthenticationError(WCSRequestError):
    """Credentials are absent, expired or not authorized for this API."""


def coverage_run(coverage_id: str) -> np.datetime64 | None:
    """UTC initialization encoded in a Météo-France coverage identifier."""
    match = _RUN.search(coverage_id)
    if match is None:
        return None
    return np.datetime64(":".join(match.groups()), "ns")


def _xml(data: bytes) -> ET.Element:
    # ElementTree does not fetch external entities; disallow DTDs altogether.
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise WCSRequestError("WCS response contains a prohibited XML declaration")
    try:
        root = ET.fromstring(data)  # noqa: S314
    except ET.ParseError:
        raise WCSRequestError("WCS returned invalid XML") from None
    if any(
        node.tag.rsplit("}", 1)[-1] in {"Exception", "ServiceException"} for node in root.iter()
    ):
        raise WCSRequestError("WCS returned an OGC exception report")
    return root


def _grib(data: bytes) -> None:
    """Reject XML/error pages and incomplete concatenated GRIB2 messages."""
    header_size, edition, minimum_size = 16, 2, 20
    offset = 0
    while offset < len(data):
        header = data[offset : offset + 16]
        if len(header) < header_size or header[:4] != b"GRIB" or header[7] != edition:
            raise WCSRequestError("WCS did not return complete GRIB2 data")
        size = int.from_bytes(header[8:16], "big")
        if (
            size < minimum_size
            or offset + size > len(data)
            or data[offset + size - 4 : offset + size] != b"7777"
        ):
            raise WCSRequestError("WCS returned truncated GRIB2 data")
        offset += size
    if not data:
        raise WCSRequestError("WCS returned an empty coverage")


class WCSClient:
    """Small WCS client with credential-independent immutable response caching.

    ``endpoint`` ends at the service name, before ``/GetCapabilities``.
    ``auth='apikey'`` sends the token in the ``apikey`` header instead of Bearer.
    An injected HTTP client remains owned by its caller. No automatic retry is
    made: rate limiting and unavailable services are reported without hiding a
    potentially large sequence of requests.
    """

    def __init__(
        self,
        endpoint: str,
        token: str | Callable[[], str],
        *,
        client: httpx.Client | None = None,
        cache: RangeCache | None = None,
        auth: str = "bearer",
    ) -> None:
        url = httpx.URL(endpoint)
        if url.scheme != "https" or url.username or url.password or url.query or url.fragment:
            raise ValueError("WCS endpoint must be HTTPS without credentials, query or fragment")
        if auth not in {"bearer", "apikey"}:
            raise ValueError("auth must be 'bearer' or 'apikey'")
        self.endpoint = endpoint.rstrip("/")
        self._token = token
        self._auth = auth
        self._client = client or httpx.Client(timeout=120.0, follow_redirects=False)
        self._owned = client is None
        self.cache = cache
        self.stats = TransferStats()
        self._namespace = namespace_for(self.endpoint)

    def _request(self, operation: str, params: list[tuple[str, str]]) -> bytes:
        token = self._token() if callable(self._token) else self._token
        if not token or not token.strip():
            raise WCSAuthenticationError("A nonempty Météo-France API token is required")
        headers = (
            {"Authorization": f"Bearer {token}"} if self._auth == "bearer" else {"apikey": token}
        )
        started = time.monotonic()
        try:
            response = self._client.get(
                f"{self.endpoint}/{operation}",
                params=tuple(params),
                headers=headers,
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise WCSRequestError("WCS transport request failed") from None
        self.stats.add(len(response.content), time.monotonic() - started)
        if response.status_code in {401, 403}:
            raise WCSAuthenticationError(f"WCS authentication failed (HTTP {response.status_code})")
        if not response.is_success:
            raise WCSRequestError(f"WCS request failed (HTTP {response.status_code})")
        return response.content

    @staticmethod
    def _params() -> list[tuple[str, str]]:
        return [("service", "WCS"), ("version", "2.0.1")]

    def capabilities(self) -> tuple[str, ...]:
        """Read the current catalogue without persisting its changing contents."""
        root = _xml(self._request("GetCapabilities", self._params()))
        return tuple(
            dict.fromkeys(
                node.text.strip()
                for node in root.iter()
                if node.tag.rsplit("}", 1)[-1] == "CoverageId" and node.text and node.text.strip()
            )
        )

    def describe(self, coverage_id: str) -> ET.Element:
        """Return the coverage metadata XML, including axis names and values."""
        return _xml(
            self._request("DescribeCoverage", [*self._params(), ("coverageid", coverage_id)])
        )

    def coverage(
        self,
        coverage_id: str,
        *,
        subsets: Sequence[str],
        validate: Callable[[bytes], None] | None = None,
    ) -> bytes:
        """Fetch one GRIB subset, retaining repeated ``subset`` query parameters.

        Optional ``validate`` runs before writing a new cache entry and on cache
        hits, allowing the caller to enforce field/run/interval metadata.
        """
        params = [*self._params(), ("coverageid", coverage_id), ("format", "application/wmo-grib")]
        params.extend(("subset", subset) for subset in sorted(subsets))

        def fetch() -> bytes:
            data = self._request("GetCoverage", params)
            _grib(data)
            if validate is not None:
                validate(data)
            return data

        if self.cache is None or coverage_run(coverage_id) is None:
            return fetch()
        data = self.cache.get_full(self._namespace, urlencode(params), fetch)
        if validate is not None:
            validate(data)
        return data

    def close(self) -> None:
        """Close only the HTTP client created by this instance."""
        if self._owned:
            self._client.close()

    def __enter__(self) -> WCSClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
