"""GRIB2 header parsing and index queries on hand-built bytes."""

from __future__ import annotations

import struct

import pytest

from supertraj_met.grib import GribFormatError, GribIndex, GribMessage, parse_header


def _section(number: int, body: bytes) -> bytes:
    return struct.pack(">IB", 5 + len(body), number) + body


def _message(level_pa: int = 25000, step: int = 3) -> bytes:
    grid = _section(3, b"\x00" + struct.pack(">I", 386061) + b"\x00\x00" + struct.pack(">H", 0))
    product = _section(
        4,
        struct.pack(">HH", 0, 0)
        + bytes([2, 2])  # octets 10-11: category 2 (momentum), number 2 (u)
        + b"\x00" * 6  # octets 12-17
        + bytes([1])  # octet 18: time unit = hour
        + struct.pack(">i", step)
        + bytes([100, 0])  # isobaric surface, scale 0
        + struct.pack(">i", level_pa),
    )
    representation = _section(5, struct.pack(">I", 386061) + struct.pack(">H", 42))
    body = _section(1, b"\x00" * 16) + grid + product + representation
    total = 16 + len(body) + 4
    return b"GRIB\x00\x00\x00\x02" + struct.pack(">Q", total) + body + b"7777"


def test_header_fields_are_decoded() -> None:
    msg = parse_header(_message(), offset=100)
    assert (msg.offset, msg.param, msg.level_type) == (100, (0, 2, 2), 100)
    assert (msg.level, msg.step_h, msg.packing, msg.npoints) == (25000.0, 3.0, 42, 386061)


@pytest.mark.parametrize(
    ("blob", "match"),
    [
        (b"NOTGRIB" + b"\x00" * 20, "no GRIB2 message"),
        (b"GRIB\x00\x00\x00\x02" + struct.pack(">Q", 3), "implausible"),
        (_message()[:40], "lacks"),
    ],
)
def test_invalid_headers_are_rejected(blob: bytes, match: str) -> None:
    with pytest.raises(GribFormatError, match=match):
        parse_header(blob, offset=0)


def test_index_queries_and_round_trip() -> None:
    msgs = [
        GribMessage(i * 10, 10, 0, 2, 2, 100, float(lev), float(step), 0, 42, 4)
        for i, (lev, step) in enumerate([(20000, 0), (25000, 0), (25000, 1)])
    ]
    idx = GribIndex("key", 30, msgs)
    assert idx.find((0, 2, 2), 100, 25000.0, 1.0) is msgs[2]
    assert idx.find((0, 0, 0), 100, 25000.0, 1.0) is None
    assert idx.levels((0, 2, 2), 100, step_h=1.0) == [25000.0]
    assert idx.steps() == [0.0, 1.0]
    back = GribIndex.from_json(idx.to_json())
    assert back.messages == msgs and back.find((0, 2, 2), 100, 20000.0, 0.0) == msgs[0]
