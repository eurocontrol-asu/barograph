from __future__ import annotations

import numcodecs
import numpy as np
import pytest

from supertraj_met.blosc import UnsupportedFrameError, decode_run, header_prefix_size, parse_header


def _frame(n: int = 600_000, blocksize: int = 65536) -> tuple[bytes, bytes]:
    data = (np.random.default_rng(1).standard_normal(n) * 10 + 250).astype("<f4")
    frame = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=blocksize).encode(data)
    return bytes(frame), data.tobytes()


def _partial(frame: bytes, start: int, stop: int) -> bytes:
    header = parse_header(frame[: header_prefix_size(frame[:16])])
    first, last = header.blocks_for(start, stop)
    c0, c1 = header.compressed_span(first, last)
    out = decode_run(header, first, last, frame[c0:c1])
    u0 = header.block_span(first)[0]
    return out[start - u0 : stop - u0]


def test_frame_has_many_blocks() -> None:
    frame, _ = _frame()
    assert parse_header(frame).nblocks >= 8


@pytest.mark.parametrize(
    ("start", "stop"),
    [(0, 4), (65530, 65540), (100_000, 700_000), (2_399_990, 2_400_000), (0, 2_400_000)],
)
def test_partial_decode_matches_the_uncompressed_bytes(start: int, stop: int) -> None:
    frame, raw = _frame()
    assert _partial(frame, start, stop) == raw[start:stop]


def test_leftover_block_alone_is_widened_to_stay_decodable() -> None:
    frame, raw = _frame(n=600_001)
    header = parse_header(frame)
    assert header.nbytes % header.blocksize != 0
    first, last = header.blocks_for(header.nbytes - 4, header.nbytes)
    assert (first, last) == (header.nblocks - 2, header.nblocks - 1)
    assert _partial(frame, header.nbytes - 4, header.nbytes) == raw[-4:]


def test_out_of_order_blocks_from_threaded_compression() -> None:
    numcodecs.blosc.set_nthreads(4)
    frame, raw = _frame(n=2_000_000, blocksize=0)
    header = parse_header(frame)
    for start in range(0, header.nbytes - 10, header.blocksize // 2):
        assert _partial(frame, start, start + 10) == raw[start : start + 10]


def test_uncompressed_frames_are_rejected() -> None:
    data = np.arange(100_000, dtype="<f4")
    frame = numcodecs.Blosc(cname="lz4", clevel=0).encode(data)
    with pytest.raises(UnsupportedFrameError):
        parse_header(bytes(frame))


def test_truncated_offsets_table_is_rejected() -> None:
    frame, _ = _frame()
    with pytest.raises(UnsupportedFrameError, match="prefix too short"):
        parse_header(frame[:20])


def _raw_header(
    version: int = 2, flags: int = 0x01, nbytes: int = 1024, blocksize: int = 512
) -> bytes:
    import struct

    return struct.pack("<BBBBiii", version, 1, flags, 4, nbytes, blocksize, 64)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (_raw_header(version=3), "version"),
        (_raw_header(flags=0x05), "extended"),
        (_raw_header(blocksize=0), "invalid"),
    ],
)
def test_frames_that_cannot_be_split_are_rejected(header: bytes, match: str) -> None:
    with pytest.raises(UnsupportedFrameError, match=match):
        parse_header(header + b"\x00" * 64)


def test_truncated_header_is_rejected() -> None:
    with pytest.raises(UnsupportedFrameError, match="truncated"):
        parse_header(_raw_header()[:10])


def test_inconsistent_block_offsets_are_rejected() -> None:
    import struct

    offsets = struct.pack("<ii", 40, 40)  # two blocks starting at the same offset
    with pytest.raises(UnsupportedFrameError, match="offsets"):
        parse_header(_raw_header() + offsets)
