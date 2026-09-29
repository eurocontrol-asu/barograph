"""Partial decoding of Blosc-1 frames: read only the blocks covering a byte range.

A Blosc-1 frame is ``header(16 B) | bstarts(int32 * nblocks) | block payloads``
where every block except the last holds ``blocksize`` uncompressed bytes and is
compressed independently (shuffle is applied per block). Given the header and
the offsets table, a contiguous run of blocks can be fetched with one HTTP range
request and re-framed into a smaller valid Blosc frame that the regular
decoder (numcodecs) decompresses. Format reference: c-blosc ``README_HEADER``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numcodecs
import numpy as np
from numpy.typing import NDArray

__all__ = ["BloscHeader", "UnsupportedFrameError", "decode_run", "parse_header"]

_HEADER = 16
_MEMCPYED = 0x02
_EXTENDED = 0x05  # blosc2 extended header marker (shuffle + bitshuffle bits both set)


class UnsupportedFrameError(ValueError):
    """The frame cannot be partially decoded; callers must fall back to a full read."""


@dataclass(frozen=True)
class BloscHeader:
    version: int
    versionlz: int
    flags: int
    typesize: int
    nbytes: int
    blocksize: int
    cbytes: int
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]

    @property
    def nblocks(self) -> int:
        return len(self.starts)

    @property
    def header_size(self) -> int:
        return _HEADER + 4 * self.nblocks

    def block_span(self, i: int) -> tuple[int, int]:
        """Uncompressed byte span of block ``i``."""
        return i * self.blocksize, min((i + 1) * self.blocksize, self.nbytes)

    def compressed_span(self, first: int, last: int) -> tuple[int, int]:
        """Smallest compressed byte span ``[start, stop)`` holding blocks ``first..last``.

        Multi-threaded compressors may store blocks out of order; the span is
        then a covering range that can include unrelated blocks (still valid).
        """
        run = slice(first, last + 1)
        return int(self.starts[run].min()), int(self.ends[run].max())

    def blocks_for(self, start: int, stop: int) -> tuple[int, int]:
        """Decodable inclusive block range covering uncompressed bytes ``[start, stop)``.

        A trailing partial ("leftover") block is decoded with different split
        rules than a full block, and a frame whose ``nbytes`` is below its
        ``blocksize`` is rejected by the decoder. A run made of the leftover
        block alone is therefore widened to include the preceding block.
        """
        if not 0 <= start < stop <= self.nbytes:
            raise ValueError(f"byte range [{start}, {stop}) outside frame of {self.nbytes} B")
        first, last = start // self.blocksize, (stop - 1) // self.blocksize
        leftover = self.nbytes % self.blocksize != 0
        if first == last == self.nblocks - 1 and leftover and first > 0:
            first -= 1
        return first, last


def header_prefix_size(first16: bytes) -> int:
    """Total header size (header + offsets) announced by the first 16 bytes."""
    if len(first16) < _HEADER:
        raise UnsupportedFrameError("truncated blosc header")
    nbytes, blocksize = (int(x) for x in struct.unpack("<ii", first16[4:12]))
    if blocksize <= 0:
        raise UnsupportedFrameError("invalid blocksize")
    return _HEADER + 4 * (-(-nbytes // blocksize))


def parse_header(prefix: bytes) -> BloscHeader:
    """Parse a Blosc-1 header and its block-offset table from a frame prefix."""
    if len(prefix) < _HEADER:
        raise UnsupportedFrameError("truncated blosc header")
    version, versionlz, flags, typesize = prefix[0], prefix[1], prefix[2], prefix[3]
    nbytes, blocksize, cbytes = struct.unpack("<iii", prefix[4:16])
    if version not in (1, 2):
        raise UnsupportedFrameError(f"unsupported blosc format version {version}")
    if flags & _MEMCPYED:
        raise UnsupportedFrameError("memcpyed frame")
    if flags & _EXTENDED == _EXTENDED:
        raise UnsupportedFrameError("blosc2 extended header")
    if blocksize <= 0 or nbytes <= 0:
        raise UnsupportedFrameError("empty or invalid frame")
    nblocks = -(-nbytes // blocksize)
    need = _HEADER + 4 * nblocks
    if len(prefix) < need:
        raise UnsupportedFrameError(f"prefix too short: need {need} bytes")
    starts = np.frombuffer(prefix[_HEADER:need], dtype="<i4").astype(np.int64)
    ordered = np.sort(starts)
    if np.any(np.diff(ordered) <= 0) or ordered[0] < need or ordered[-1] >= cbytes:
        raise UnsupportedFrameError("inconsistent block offsets")
    # a block ends where the physically next block starts (or at cbytes)
    following = np.append(ordered[1:], cbytes)
    ends = following[np.searchsorted(ordered, starts)]
    return BloscHeader(version, versionlz, flags, typesize, nbytes, blocksize, cbytes, starts, ends)


def decode_run(header: BloscHeader, first: int, last: int, payload: bytes) -> bytes:
    """Decode blocks ``first..last`` from their compressed ``payload``.

    ``payload`` must be exactly ``frame[compressed_span(first, last)]``. Returns
    the uncompressed bytes ``frame_uncompressed[block_span(first)[0]:block_span(last)[1]]``.
    """
    c0, c1 = header.compressed_span(first, last)
    if len(payload) != c1 - c0:
        raise ValueError(f"payload has {len(payload)} bytes, expected {c1 - c0}")
    n = last - first + 1
    u0, _ = header.block_span(first)
    _, u1 = header.block_span(last)
    new_header_size = _HEADER + 4 * n
    starts = header.starts[first : last + 1] - c0 + new_header_size
    head = struct.pack(
        "<BBBBiii",
        header.version,
        header.versionlz,
        header.flags,
        header.typesize,
        u1 - u0,
        header.blocksize,
        new_header_size + len(payload),
    )
    frame = head + starts.astype("<i4").tobytes() + payload
    out = numcodecs.Blosc().decode(frame)
    data = bytes(out)
    if len(data) != u1 - u0:
        raise UnsupportedFrameError("partial decode returned an unexpected size")
    return data
