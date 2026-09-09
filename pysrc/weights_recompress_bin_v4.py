#!/usr/bin/env python3
"""
weights_recompress_bin_v4.py

Experimental LOSSLESS recompressor for the already-decompressed transformer
weight file:

    6m-q4-fp32.tfwc2.bin  ->  6m-q4-fp32.tfwc4

IMPORTANT
---------
This script DOES NOT decompress .tfwc2. It takes the normal ~39 MB reference
tensor file directly and recompresses it.

It is intentionally separate from pysrc/weights_compress.py so the existing
TFWC2 path remains untouched.

Usage
-----
From repository root:

  python weights_recompress_bin_v4.py compress \
      6m-q4-fp32.tfwc2.bin 6m-q4-fp32.tfwc4

  python weights_recompress_bin_v4.py decompress \
      6m-q4-fp32.tfwc4 roundtrip.bin

  python weights_recompress_bin_v4.py verify \
      6m-q4-fp32.tfwc2.bin 6m-q4-fp32.tfwc4

The compressor tries several reversible models per tensor and stores whichever
produces the smallest payload.

TFWC4 candidate models
----------------------
INT8 / Q4:
  1  global adaptive 15-symbol tree
  2  previous-symbol + phase context
  3  previous-2-symbol + phase hashed context
  4  row-neighbour context
  5  modulo-15 delta from previous value
  6  modulo-15 delta from row-left value
  7  magnitude/sign factorization
  8  magnitude/sign with row-neighbour context

BF16:
  20 high byte + low byte conditioned on high byte
  21 XOR previous word + contextual bytes
  22 XOR row-left word + contextual bytes

F32 / I32:
  30 contextual byte planes
  31 XOR previous 32-bit word + contextual byte planes
  32 XOR row-left 32-bit word + contextual byte planes

RAW is always available as a fallback.

For rope.sin / rope.cos, the existing bit-exact CUDA-compatible regeneration
routine is reused if available and verified exactly.

No PPMd/LSTM/FXCM is included yet on purpose: first measure how far these much
smaller C++-friendly structural models go. The per-tensor report identifies
where a heavier PPMd/FXCM stage is worth adding.
"""

from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Normal reference tensor file support from your repository.
from pysrc.export_weights import (
    DTYPE_BF16_BITS,
    DTYPE_FLOAT32,
    DTYPE_INT8,
    DTYPE_INT32,
    _DTYPE_TO_NUMPY,
    read_tensor_file,
    write_tensor_file,
)

# Optional: existing bit-exact RoPE reconstruction from TFWC2.
try:
    from pysrc.weights_compress import cuda_rope_table
except Exception:
    cuda_rope_table = None


MAGIC = b"FX2TFWC4"

MODE_RAW = 0

MODE_Q4_GLOBAL = 1
MODE_Q4_PREV = 2
MODE_Q4_PREV2 = 3
MODE_Q4_ROW = 4
MODE_Q4_DELTA_PREV = 5
MODE_Q4_DELTA_ROW = 6
MODE_Q4_MAG_SIGN = 7
MODE_Q4_MAG_SIGN_ROW = 8

MODE_BF16_HILO = 20
MODE_BF16_XOR_PREV = 21
MODE_BF16_XOR_ROW = 22

MODE_PLANE4 = 30
MODE_PLANE4_XOR_PREV = 31
MODE_PLANE4_XOR_ROW = 32

MODE_ROPE_SIN = 40
MODE_ROPE_COS = 41

MODE_NAMES = {
    MODE_RAW: "raw",
    MODE_Q4_GLOBAL: "q4-global",
    MODE_Q4_PREV: "q4-prev",
    MODE_Q4_PREV2: "q4-prev2",
    MODE_Q4_ROW: "q4-row",
    MODE_Q4_DELTA_PREV: "q4-dprev",
    MODE_Q4_DELTA_ROW: "q4-drow",
    MODE_Q4_MAG_SIGN: "q4-mag",
    MODE_Q4_MAG_SIGN_ROW: "q4-mag-row",
    MODE_BF16_HILO: "bf16-hilo",
    MODE_BF16_XOR_PREV: "bf16-xprev",
    MODE_BF16_XOR_ROW: "bf16-xrow",
    MODE_PLANE4: "plane4",
    MODE_PLANE4_XOR_PREV: "plane4-xprev",
    MODE_PLANE4_XOR_ROW: "plane4-xrow",
    MODE_ROPE_SIN: "rope-sin",
    MODE_ROPE_COS: "rope-cos",
}

_TOP = 1 << 24
_MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Range coder
# ---------------------------------------------------------------------------

class BinEncoder:
    """LZMA-style binary range coder, 11-bit adaptive probabilities."""

    def __init__(self):
        self.low = 0
        self.range = _MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low > _MASK32:
            carry = self.low >> 32
            self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & _MASK32

    def bit(self, probs: Dict[int, int], key: int, bit: int):
        p = probs.get(key, 1024)
        bound = (self.range >> 11) * p
        if bit == 0:
            self.range = bound
            probs[key] = p + ((2048 - p) >> 5)
        else:
            self.low += bound
            self.range -= bound
            probs[key] = p - (p >> 5)

        while self.range < _TOP:
            self.range <<= 8
            self._shift_low()

    def finish(self):
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class BinDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 1
        self.range = _MASK32
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self):
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def bit(self, probs: Dict[int, int], key: int):
        p = probs.get(key, 1024)
        bound = (self.range >> 11) * p

        if self.code < bound:
            self.range = bound
            probs[key] = p + ((2048 - p) >> 5)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[key] = p - (p >> 5)
            bit = 1

        while self.range < _TOP:
            self.code = ((self.code << 8) | self._byte()) & _MASK32
            self.range <<= 8

        return bit


def enc_tree(enc: BinEncoder, probs: Dict[int, int], ctx: int, nbits: int, sym: int):
    node = 1
    for k in range(nbits - 1, -1, -1):
        b = (sym >> k) & 1
        key = (ctx << 9) | node
        enc.bit(probs, key, b)
        node = (node << 1) | b


def dec_tree(dec: BinDecoder, probs: Dict[int, int], ctx: int, nbits: int):
    node = 1
    for _ in range(nbits):
        key = (ctx << 9) | node
        node = (node << 1) | dec.bit(probs, key)
    return node - (1 << nbits)


# ---------------------------------------------------------------------------
# Container varints
# ---------------------------------------------------------------------------

def put_uvarint(x: int) -> bytes:
    if x < 0:
        raise ValueError("negative uvarint")
    out = bytearray()
    while x >= 0x80:
        out.append((x & 0x7F) | 0x80)
        x >>= 7
    out.append(x)
    return bytes(out)


def get_uvarint(f) -> int:
    value = 0
    shift = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("truncated uvarint")
        x = b[0]
        value |= (x & 0x7F) << shift
        if not (x & 0x80):
            return value
        shift += 7
        if shift > 63:
            raise ValueError("uvarint overflow")


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def row_width(array: np.ndarray) -> int:
    """
    Treat the final dimension as the local row width.
    For vectors/scalars there is no useful row predictor.
    """
    if array.ndim < 2:
        return 0
    w = int(array.shape[-1])
    return w if w > 1 else 0


def row_ref(i: int, width: int) -> int:
    if width and (i % width) != 0:
        return i - 1
    return -1


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------

def q4_vals(array: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(array, dtype=np.int8).reshape(-1).astype(np.int16)
    if len(v) and (int(v.min()) < -7 or int(v.max()) > 7):
        raise ValueError("Q4 candidate requires values in [-7,7]")
    return v


def q4_syms(array: np.ndarray) -> np.ndarray:
    return (q4_vals(array) + 7).astype(np.uint8)


def q4_encode_global(array):
    s = q4_syms(array)
    enc, probs = BinEncoder(), {}
    for x in s.tolist():
        enc_tree(enc, probs, 0, 4, int(x))
    return enc.finish()


def q4_decode_global(payload, count):
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    for i in range(count):
        out[i] = dec_tree(dec, probs, 0, 4) - 7
    return out


def q4_encode_prev(array):
    s = q4_syms(array)
    enc, probs = BinEncoder(), {}
    prev = 7
    for i, x0 in enumerate(s.tolist()):
        x = int(x0)
        ctx = ((i & 15) << 4) | prev
        enc_tree(enc, probs, ctx, 4, x)
        prev = x
    return enc.finish()


def q4_decode_prev(payload, count):
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    prev = 7
    for i in range(count):
        ctx = ((i & 15) << 4) | prev
        x = dec_tree(dec, probs, ctx, 4)
        out[i] = x - 7
        prev = x
    return out


def q4_encode_prev2(array):
    s = q4_syms(array)
    enc, probs = BinEncoder(), {}
    p1 = p2 = 7
    for i, x0 in enumerate(s.tolist()):
        x = int(x0)
        # 16 phases * 256 hashed previous-pair states.
        h = (p1 * 17 + p2 * 31) & 0xFF
        ctx = ((i & 15) << 8) | h
        enc_tree(enc, probs, ctx, 4, x)
        p2, p1 = p1, x
    return enc.finish()


def q4_decode_prev2(payload, count):
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    p1 = p2 = 7
    for i in range(count):
        h = (p1 * 17 + p2 * 31) & 0xFF
        ctx = ((i & 15) << 8) | h
        x = dec_tree(dec, probs, ctx, 4)
        out[i] = x - 7
        p2, p1 = p1, x
    return out


def q4_encode_row(array):
    s = q4_syms(array)
    w = row_width(array)
    enc, probs = BinEncoder(), {}
    seen = []
    prev = 7
    for i, x0 in enumerate(s.tolist()):
        x = int(x0)
        j = row_ref(i, w)
        left = seen[j] if j >= 0 else 7
        ctx = ((i & 7) << 8) | (left << 4) | prev
        enc_tree(enc, probs, ctx, 4, x)
        seen.append(x)
        prev = x
    return enc.finish()


def q4_decode_row(payload, count, shape):
    dummy = np.empty(shape, dtype=np.int8)
    w = row_width(dummy)
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    syms = []
    prev = 7
    for i in range(count):
        j = row_ref(i, w)
        left = syms[j] if j >= 0 else 7
        ctx = ((i & 7) << 8) | (left << 4) | prev
        x = dec_tree(dec, probs, ctx, 4)
        out[i] = x - 7
        syms.append(x)
        prev = x
    return out


def q4_encode_delta_prev(array):
    s = q4_syms(array)
    enc, probs = BinEncoder(), {}
    prev = 7
    prev_d = 0
    for i, x0 in enumerate(s.tolist()):
        x = int(x0)
        d = (x - prev) % 15
        ctx = ((i & 15) << 4) | prev_d
        enc_tree(enc, probs, ctx, 4, d)
        prev, prev_d = x, d
    return enc.finish()


def q4_decode_delta_prev(payload, count):
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    prev = 7
    prev_d = 0
    for i in range(count):
        ctx = ((i & 15) << 4) | prev_d
        d = dec_tree(dec, probs, ctx, 4)
        x = (prev + d) % 15
        out[i] = x - 7
        prev, prev_d = x, d
    return out


def q4_encode_delta_row(array):
    s = q4_syms(array)
    w = row_width(array)
    enc, probs = BinEncoder(), {}
    syms = []
    prev_d = 0
    for i, x0 in enumerate(s.tolist()):
        x = int(x0)
        j = row_ref(i, w)
        pred = syms[j] if j >= 0 else 7
        d = (x - pred) % 15
        ctx = ((i & 15) << 4) | prev_d
        enc_tree(enc, probs, ctx, 4, d)
        syms.append(x)
        prev_d = d
    return enc.finish()


def q4_decode_delta_row(payload, count, shape):
    dummy = np.empty(shape, dtype=np.int8)
    w = row_width(dummy)
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.int8)
    syms = []
    prev_d = 0
    for i in range(count):
        j = row_ref(i, w)
        pred = syms[j] if j >= 0 else 7
        ctx = ((i & 15) << 4) | prev_d
        d = dec_tree(dec, probs, ctx, 4)
        x = (pred + d) % 15
        out[i] = x - 7
        syms.append(x)
        prev_d = d
    return out


def _mag_ctx_key(kind: int, phase: int, a: int, b: int = 0):
    # compact deterministic integer context
    return (kind << 16) | ((phase & 15) << 12) | ((a & 15) << 4) | (b & 15)


def q4_encode_mag(array, row_mode=False):
    v = q4_vals(array)
    w = row_width(array) if row_mode else 0
    enc, probs = BinEncoder(), {}

    vals = []
    prev_mag = 0
    prev_sign = 0

    for i, x0 in enumerate(v.tolist()):
        x = int(x0)
        mag = abs(x)
        sign = 1 if x < 0 else 0
        phase = i & 15

        if row_mode:
            j = row_ref(i, w)
            left = vals[j] if j >= 0 else 0
            left_mag = abs(left)
            left_sign = 1 if left < 0 else 0
        else:
            left_mag = 0
            left_sign = 0

        nz = 1 if mag else 0
        zctx = _mag_ctx_key(1 if not row_mode else 4, phase, prev_mag, left_mag)
        enc.bit(probs, zctx, nz)

        if nz:
            # magnitude 1..7 => 0..6
            mctx = _mag_ctx_key(2 if not row_mode else 5, phase, prev_mag, left_mag)
            enc_tree(enc, probs, mctx, 3, mag - 1)

            sctx = _mag_ctx_key(
                3 if not row_mode else 6,
                phase,
                prev_sign,
                left_sign,
            )
            enc.bit(probs, sctx, sign)
            prev_sign = sign

        vals.append(x)
        prev_mag = mag

    return enc.finish()


def q4_decode_mag(payload, count, shape, row_mode=False):
    dummy = np.empty(shape, dtype=np.int8)
    w = row_width(dummy) if row_mode else 0
    dec, probs = BinDecoder(payload), {}

    out = np.empty(count, np.int8)
    vals = []
    prev_mag = 0
    prev_sign = 0

    for i in range(count):
        phase = i & 15

        if row_mode:
            j = row_ref(i, w)
            left = vals[j] if j >= 0 else 0
            left_mag = abs(left)
            left_sign = 1 if left < 0 else 0
        else:
            left_mag = 0
            left_sign = 0

        zctx = _mag_ctx_key(1 if not row_mode else 4, phase, prev_mag, left_mag)
        nz = dec.bit(probs, zctx)

        if nz:
            mctx = _mag_ctx_key(2 if not row_mode else 5, phase, prev_mag, left_mag)
            mag = dec_tree(dec, probs, mctx, 3) + 1

            sctx = _mag_ctx_key(
                3 if not row_mode else 6,
                phase,
                prev_sign,
                left_sign,
            )
            sign = dec.bit(probs, sctx)
            x = -mag if sign else mag
            prev_sign = sign
        else:
            mag = 0
            x = 0

        out[i] = x
        vals.append(x)
        prev_mag = mag

    return out


# ---------------------------------------------------------------------------
# BF16
# ---------------------------------------------------------------------------

def bf16_encode_hilo(array):
    v = np.ascontiguousarray(array, dtype=np.uint16).reshape(-1)
    enc, probs = BinEncoder(), {}
    prev_hi = 0

    for i, x0 in enumerate(v.tolist()):
        x = int(x0)
        hi, lo = (x >> 8) & 255, x & 255
        phase = i & 7
        enc_tree(enc, probs, (phase << 8) | prev_hi, 8, hi)
        enc_tree(enc, probs, (1 << 20) | (phase << 8) | hi, 8, lo)
        prev_hi = hi

    return enc.finish()


def bf16_decode_hilo(payload, count):
    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.uint16)
    prev_hi = 0

    for i in range(count):
        phase = i & 7
        hi = dec_tree(dec, probs, (phase << 8) | prev_hi, 8)
        lo = dec_tree(dec, probs, (1 << 20) | (phase << 8) | hi, 8)
        out[i] = (hi << 8) | lo
        prev_hi = hi

    return out


def bf16_encode_xor(array, row_mode=False):
    v = np.ascontiguousarray(array, dtype=np.uint16).reshape(-1)
    w = row_width(array) if row_mode else 0

    enc, probs = BinEncoder(), {}
    originals = []
    prev_xhi = 0

    for i, x0 in enumerate(v.tolist()):
        x = int(x0)
        if row_mode:
            j = row_ref(i, w)
            pred = originals[j] if j >= 0 else 0
        else:
            pred = originals[-1] if originals else 0

        d = x ^ pred
        hi, lo = (d >> 8) & 255, d & 255
        phase = i & 7
        base = 2 if row_mode else 1

        enc_tree(enc, probs, (base << 20) | (phase << 8) | prev_xhi, 8, hi)
        enc_tree(enc, probs, ((base + 4) << 20) | (phase << 8) | hi, 8, lo)

        originals.append(x)
        prev_xhi = hi

    return enc.finish()


def bf16_decode_xor(payload, count, shape, row_mode=False):
    dummy = np.empty(shape, dtype=np.uint16)
    w = row_width(dummy) if row_mode else 0

    dec, probs = BinDecoder(payload), {}
    out = np.empty(count, np.uint16)
    originals = []
    prev_xhi = 0

    for i in range(count):
        if row_mode:
            j = row_ref(i, w)
            pred = originals[j] if j >= 0 else 0
        else:
            pred = originals[-1] if originals else 0

        phase = i & 7
        base = 2 if row_mode else 1
        hi = dec_tree(dec, probs, (base << 20) | (phase << 8) | prev_xhi, 8)
        lo = dec_tree(dec, probs, ((base + 4) << 20) | (phase << 8) | hi, 8)
        x = ((hi << 8) | lo) ^ pred

        out[i] = x
        originals.append(x)
        prev_xhi = hi

    return out


# ---------------------------------------------------------------------------
# F32 / I32
# ---------------------------------------------------------------------------

def words_u32(array):
    raw = np.ascontiguousarray(array).tobytes()
    if len(raw) % 4:
        raise ValueError("32-bit tensor byte size not divisible by 4")
    return np.frombuffer(raw, dtype="<u4").copy()


def plane4_encode(array, predictor: str):
    words = words_u32(array)
    w = row_width(array) if predictor == "row" else 0

    transformed = np.empty_like(words)
    originals = []

    for i, x0 in enumerate(words.tolist()):
        x = int(x0)
        if predictor == "prev":
            pred = originals[-1] if originals else 0
        elif predictor == "row":
            j = row_ref(i, w)
            pred = originals[j] if j >= 0 else 0
        else:
            pred = 0

        transformed[i] = np.uint32(x ^ pred) if predictor != "none" else np.uint32(x)
        originals.append(x)

    raw = transformed.tobytes()
    enc, probs = BinEncoder(), {}
    prev_plane = [0, 0, 0, 0]

    for i, b in enumerate(raw):
        plane = i & 3
        phase = (i >> 2) & 7
        ctx = (plane << 19) | (phase << 16) | prev_plane[plane]
        enc_tree(enc, probs, ctx, 8, b)
        prev_plane[plane] = b

    return enc.finish()


def plane4_decode(payload, count, shape, dtype, predictor: str):
    nbytes = count * 4
    dec, probs = BinDecoder(payload), {}
    raw = bytearray(nbytes)
    prev_plane = [0, 0, 0, 0]

    for i in range(nbytes):
        plane = i & 3
        phase = (i >> 2) & 7
        ctx = (plane << 19) | (phase << 16) | prev_plane[plane]
        b = dec_tree(dec, probs, ctx, 8)
        raw[i] = b
        prev_plane[plane] = b

    t = np.frombuffer(bytes(raw), dtype="<u4")
    if predictor == "none":
        words = t.copy()
    else:
        words = np.empty_like(t)
        dummy = np.empty(shape, dtype=dtype)
        w = row_width(dummy) if predictor == "row" else 0
        originals = []

        for i, d0 in enumerate(t.tolist()):
            d = int(d0)
            if predictor == "prev":
                pred = originals[-1] if originals else 0
            else:
                j = row_ref(i, w)
                pred = originals[j] if j >= 0 else 0

            x = d ^ pred
            words[i] = np.uint32(x)
            originals.append(x)

    return np.frombuffer(words.tobytes(), dtype=dtype)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    mode: int
    payload: bytes


def best(cands: Iterable[Candidate]) -> Candidate:
    return min(cands, key=lambda c: (len(c.payload), c.mode))


def rope_candidate(name, dtype_code, data, seen):
    if cuda_rope_table is None:
        return None
    if dtype_code != DTYPE_FLOAT32:
        return None
    if name not in ("rope.sin", "rope.cos"):
        return None
    if data.ndim != 2 or "rope.inv_freq" not in seen:
        return None

    inv = seen["rope.inv_freq"]
    if inv.shape != (data.shape[1],):
        return None

    table = cuda_rope_table(inv, data.shape[0], cos=(name == "rope.cos"))
    if np.array_equal(
        np.ascontiguousarray(table).view(np.uint32),
        np.ascontiguousarray(data).view(np.uint32),
    ):
        return Candidate(
            MODE_ROPE_COS if name == "rope.cos" else MODE_ROPE_SIN,
            b"",
        )
    return None


def encode_tensor(name, dtype_code, data, seen):
    data = np.ascontiguousarray(data)

    rc = rope_candidate(name, dtype_code, data, seen)
    if rc is not None:
        return rc, []

    cands = [Candidate(MODE_RAW, data.tobytes())]

    if dtype_code == DTYPE_INT8:
        flat = data.reshape(-1)
        if len(flat) == 0 or (int(flat.min()) >= -7 and int(flat.max()) <= 7):
            cands.extend([
                Candidate(MODE_Q4_GLOBAL, q4_encode_global(data)),
                Candidate(MODE_Q4_PREV, q4_encode_prev(data)),
                Candidate(MODE_Q4_PREV2, q4_encode_prev2(data)),
                Candidate(MODE_Q4_ROW, q4_encode_row(data)),
                Candidate(MODE_Q4_DELTA_PREV, q4_encode_delta_prev(data)),
                Candidate(MODE_Q4_DELTA_ROW, q4_encode_delta_row(data)),
                Candidate(MODE_Q4_MAG_SIGN, q4_encode_mag(data, False)),
                Candidate(MODE_Q4_MAG_SIGN_ROW, q4_encode_mag(data, True)),
            ])

    elif dtype_code == DTYPE_BF16_BITS:
        cands.extend([
            Candidate(MODE_BF16_HILO, bf16_encode_hilo(data)),
            Candidate(MODE_BF16_XOR_PREV, bf16_encode_xor(data, False)),
            Candidate(MODE_BF16_XOR_ROW, bf16_encode_xor(data, True)),
        ])

    elif dtype_code in (DTYPE_FLOAT32, DTYPE_INT32):
        cands.extend([
            Candidate(MODE_PLANE4, plane4_encode(data, "none")),
            Candidate(MODE_PLANE4_XOR_PREV, plane4_encode(data, "prev")),
            Candidate(MODE_PLANE4_XOR_ROW, plane4_encode(data, "row")),
        ])

    winner = best(cands)
    ranked = sorted(cands, key=lambda c: (len(c.payload), c.mode))
    return winner, ranked


# ---------------------------------------------------------------------------
# File writer / reader
# ---------------------------------------------------------------------------

def write_v4(filename, entries, verbose=True):
    seen = {}
    raw_total = 0
    payload_total = 0
    by_mode = {}

    with open(filename, "wb") as f:
        f.write(MAGIC)
        f.write(put_uvarint(len(entries)))

        for idx, (name, dtype_code, array) in enumerate(entries):
            data = np.ascontiguousarray(array)
            name_b = name.encode("utf-8")

            winner, ranked = encode_tensor(name, dtype_code, data, seen)

            f.write(put_uvarint(len(name_b)))
            f.write(name_b)
            f.write(bytes([dtype_code]))
            f.write(bytes([data.ndim]))
            for d in data.shape:
                f.write(put_uvarint(int(d)))
            f.write(bytes([winner.mode]))
            f.write(put_uvarint(len(winner.payload)))
            f.write(winner.payload)

            raw_total += data.nbytes
            payload_total += len(winner.payload)
            by_mode[winner.mode] = by_mode.get(winner.mode, 0) + len(winner.payload)

            if dtype_code == DTYPE_FLOAT32:
                seen[name] = data

            if verbose:
                alt = ""
                if len(ranked) >= 2:
                    alt = (
                        f"  2nd={MODE_NAMES[ranked[1].mode]}:"
                        f"{len(ranked[1].payload):,}"
                    )

                print(
                    f"{idx:3d} {name:50s} "
                    f"{MODE_NAMES[winner.mode]:14s} "
                    f"{data.nbytes:10,d} -> {len(winner.payload):10,d}"
                    f"{alt}"
                )

    final_size = os.path.getsize(filename)

    if verbose:
        print()
        print(f"Raw tensor bytes : {raw_total:,}")
        print(f"Payload bytes    : {payload_total:,}")
        print(f"TFWC4 file bytes : {final_size:,}")
        if raw_total:
            print(f"Ratio            : {final_size/raw_total:.6f}")
            print(f"Bits/raw-byte    : {8.0*final_size/raw_total:.6f}")
        print("Winning modes:")
        for mode, size in sorted(by_mode.items()):
            print(f"  {MODE_NAMES[mode]:14s} {size:,}")

    return final_size


def read_v4(filename):
    tensors = {}

    with open(filename, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("not an FX2TFWC4 file")

        n = get_uvarint(f)

        for _ in range(n):
            name_len = get_uvarint(f)
            name = f.read(name_len).decode("utf-8")

            x = f.read(2)
            if len(x) != 2:
                raise EOFError("truncated tensor header")
            dtype_code = x[0]
            ndim = x[1]

            shape = tuple(get_uvarint(f) for _ in range(ndim))

            m = f.read(1)
            if not m:
                raise EOFError("truncated mode")
            mode = m[0]

            plen = get_uvarint(f)
            payload = f.read(plen)
            if len(payload) != plen:
                raise EOFError("truncated payload")

            dtype = np.dtype(_DTYPE_TO_NUMPY[dtype_code])
            count = int(np.prod(shape)) if ndim else 1

            if mode == MODE_RAW:
                expected = count * dtype.itemsize
                if plen != expected:
                    raise ValueError(f"{name}: bad RAW length")
                arr = np.frombuffer(payload, dtype=dtype).copy().reshape(shape)

            elif mode == MODE_Q4_GLOBAL:
                arr = q4_decode_global(payload, count).reshape(shape)

            elif mode == MODE_Q4_PREV:
                arr = q4_decode_prev(payload, count).reshape(shape)

            elif mode == MODE_Q4_PREV2:
                arr = q4_decode_prev2(payload, count).reshape(shape)

            elif mode == MODE_Q4_ROW:
                arr = q4_decode_row(payload, count, shape).reshape(shape)

            elif mode == MODE_Q4_DELTA_PREV:
                arr = q4_decode_delta_prev(payload, count).reshape(shape)

            elif mode == MODE_Q4_DELTA_ROW:
                arr = q4_decode_delta_row(payload, count, shape).reshape(shape)

            elif mode == MODE_Q4_MAG_SIGN:
                arr = q4_decode_mag(payload, count, shape, False).reshape(shape)

            elif mode == MODE_Q4_MAG_SIGN_ROW:
                arr = q4_decode_mag(payload, count, shape, True).reshape(shape)

            elif mode == MODE_BF16_HILO:
                arr = bf16_decode_hilo(payload, count).reshape(shape)

            elif mode == MODE_BF16_XOR_PREV:
                arr = bf16_decode_xor(payload, count, shape, False).reshape(shape)

            elif mode == MODE_BF16_XOR_ROW:
                arr = bf16_decode_xor(payload, count, shape, True).reshape(shape)

            elif mode == MODE_PLANE4:
                arr = plane4_decode(payload, count, shape, dtype, "none").reshape(shape)

            elif mode == MODE_PLANE4_XOR_PREV:
                arr = plane4_decode(payload, count, shape, dtype, "prev").reshape(shape)

            elif mode == MODE_PLANE4_XOR_ROW:
                arr = plane4_decode(payload, count, shape, dtype, "row").reshape(shape)

            elif mode in (MODE_ROPE_SIN, MODE_ROPE_COS):
                if cuda_rope_table is None:
                    raise RuntimeError("cuda_rope_table unavailable")
                if "rope.inv_freq" not in tensors:
                    raise ValueError("rope.inv_freq must precede rope table")
                inv_code, inv = tensors["rope.inv_freq"]
                if inv_code != DTYPE_FLOAT32:
                    raise ValueError("rope.inv_freq wrong dtype")
                arr = cuda_rope_table(
                    inv,
                    shape[0],
                    cos=(mode == MODE_ROPE_COS),
                ).reshape(shape)

            else:
                raise ValueError(f"{name}: unknown mode {mode}")

            tensors[name] = (dtype_code, np.ascontiguousarray(arr))

        if f.read(1):
            raise ValueError("trailing bytes after TFWC4")

    return tensors


def entries_from_reference(path):
    return [
        (name, dtype_code, arr)
        for name, (dtype_code, arr) in read_tensor_file(path).items()
    ]


def _dirichlet_sequence_bits(counts, prior_frequencies, concentration):
    """Ideal cost of a causal multinomial updated within one sequence."""
    priors = [
        max(1, (int(frequency) * concentration + 16384) // 32768)
        for frequency in prior_frequencies
    ]
    prior_total = sum(priors)
    count_total = int(sum(counts))
    result = math.lgamma(prior_total + count_total) - math.lgamma(prior_total)
    for prior, count in zip(priors, counts):
        result -= math.lgamma(prior + int(count)) - math.lgamma(prior)
    return result / math.log(2.0)


def screen_q4_contexts(src, dst):
    """Screen no-side-data row/column adaptation against static tensor coding."""
    tensors = read_recompressed(src)
    concentrations = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    totals = {
        "static_bits": 0.0,
        **{f"row_{alpha}_bits": 0.0 for alpha in concentrations},
        **{f"column_{alpha}_bits": 0.0 for alpha in concentrations},
    }
    rows = []

    for name, (dtype_code, array) in tensors.items():
        if dtype_code != DTYPE_INT8 or array.size == 0:
            continue
        values = np.ascontiguousarray(array, dtype=np.int8)
        if int(values.min()) < -7 or int(values.max()) > 7:
            continue
        symbols = values.astype(np.int16) + 7
        flat_counts = np.bincount(symbols.reshape(-1), minlength=15)
        from pysrc.weights_compress import _histogram_cumulative
        cumulative = _histogram_cumulative(symbols.reshape(-1).tolist())
        frequencies = np.diff(np.asarray(cumulative, dtype=np.int64))
        probabilities = frequencies / 32768.0
        static_bits = float(-(flat_counts * np.log2(probabilities)).sum())

        matrix = (symbols.reshape((-1, symbols.shape[-1]))
                  if symbols.ndim >= 2 else symbols.reshape((1, -1)))
        row_counts = [np.bincount(row, minlength=15) for row in matrix]
        column_counts = [np.bincount(matrix[:, column], minlength=15)
                         for column in range(matrix.shape[1])]
        row = {"tensor": name, "values": int(symbols.size),
               "static_bits": static_bits}
        totals["static_bits"] += static_bits
        for alpha in concentrations:
            row_bits = sum(_dirichlet_sequence_bits(counts, frequencies, alpha)
                           for counts in row_counts)
            column_bits = sum(_dirichlet_sequence_bits(counts, frequencies, alpha)
                              for counts in column_counts)
            row[f"row_{alpha}_bits"] = row_bits
            row[f"column_{alpha}_bits"] = column_bits
            totals[f"row_{alpha}_bits"] += row_bits
            totals[f"column_{alpha}_bits"] += column_bits
        rows.append(row)

    report = {
        "format": "fx4-q4-context-screen-v1",
        "source": os.path.abspath(src),
        "q4_tensors": len(rows),
        "q4_values": sum(row["values"] for row in rows),
        "totals": totals,
        "equivalent_savings_bytes": {
            key.removesuffix("_bits"): (totals["static_bits"] - value) / 8.0
            for key, value in totals.items() if key != "static_bits"
        },
        "tensors": rows,
    }
    with open(dst, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(f"Q4 tensors: {len(rows)}; values: {report['q4_values']:,}")
    print(f"Static ideal cost: {totals['static_bits'] / 8:,.1f} bytes")
    for key, saving in sorted(report["equivalent_savings_bytes"].items(),
                              key=lambda item: -item[1]):
        print(f"{key:12s}: {saving:+,.1f} ideal bytes vs static")
    print(f"Report: {dst}")


def assert_equal(a, b):
    if list(a.keys()) != list(b.keys()):
        raise AssertionError("tensor names/order differ")

    for name in a:
        ca, aa = a[name]
        cb, ab = b[name]

        if ca != cb:
            raise AssertionError(f"{name}: dtype code differs")
        if aa.dtype != ab.dtype:
            raise AssertionError(f"{name}: dtype differs")
        if aa.shape != ab.shape:
            raise AssertionError(f"{name}: shape differs")

        # Byte comparison catches floating NaN payload differences too.
        ba = np.ascontiguousarray(aa).tobytes()
        bb = np.ascontiguousarray(ab).tobytes()
        if ba != bb:
            raise AssertionError(f"{name}: bytes differ")


def cmd_compress(src, dst):
    ref = read_tensor_file(src)
    entries = [
        (name, dtype_code, arr)
        for name, (dtype_code, arr) in ref.items()
    ]

    print(f"Input reference : {src} ({os.path.getsize(src):,} bytes)")
    print(f"Compressing     : {dst}")
    print()

    write_v4(dst, entries, verbose=True)

    decoded = read_v4(dst)
    assert_equal(ref, decoded)

    print()
    print("BIT-EXACT ROUNDTRIP: PASS")
    print(f"Input .bin      : {os.path.getsize(src):,} bytes")
    print(f"Output .tfwc4   : {os.path.getsize(dst):,} bytes")


def cmd_decompress(src, dst):
    tensors = read_recompressed(src)
    entries = [
        (name, dtype_code, arr)
        for name, (dtype_code, arr) in tensors.items()
    ]
    write_tensor_file(dst, entries)
    print(f"{src} -> {dst} ({os.path.getsize(dst):,} bytes)")


def cmd_verify(ref_path, compressed_path):
    ref = read_tensor_file(ref_path)
    got = read_recompressed(compressed_path)
    assert_equal(ref, got)
    print("BIT-EXACT VERIFY: PASS")
    print(f"reference : {os.path.getsize(ref_path):,} bytes")
    print(f"compressed: {os.path.getsize(compressed_path):,} bytes")


def read_recompressed(path):
    with open(path, "rb") as source:
        magic = source.read(8)
    if magic == MAGIC:
        return read_v4(path)
    from pysrc.weights_compress import _read_compressed
    return _read_compressed(path)


def main():
    usage = f"""Usage:
  python {os.path.basename(sys.argv[0])} compress   input.bin output.tfwc4
  python {os.path.basename(sys.argv[0])} decompress input.tfwc4 output.bin
  python {os.path.basename(sys.argv[0])} verify     reference.bin input.tfwc4
  python {os.path.basename(sys.argv[0])} compact    input.bin output.tfwc5
    compact keeps shared metadata models and uses the production C++ v5 loader.
  python {os.path.basename(sys.argv[0])} screen     input.tfwc5 report.json
    screen estimates causal row/column Q4 adaptation with no row side tables.
"""

    if len(sys.argv) != 4:
        sys.exit(usage)

    cmd, a, b = sys.argv[1:]

    if not os.path.isfile(a):
        sys.exit(f"input not found: {a}")

    if cmd == "compact":
        from pysrc.weights_compress import write_tensor_file_v2, read_tensor_file_v2
        with open(a, "rb") as source:
            input_magic = source.read(8)
        ref = read_tensor_file(a) if input_magic == b"FX2TFW01" else read_recompressed(a)
        entries = [(name, code, arr) for name, (code, arr) in ref.items()]
        report = []
        write_tensor_file_v2(b, entries, compact=True, report=report)
        assert_equal(ref, read_tensor_file_v2(b, compact=True))
        modes = {}
        for row in report:
            modes[row["encoding"]] = modes.get(row["encoding"], 0) + 1
        print(f"TFWC5: {os.path.getsize(b):,} bytes; Q4 modes {modes}")
        print("BIT-EXACT ROUNDTRIP: PASS (metadata included; no side file)")
    elif cmd == "screen":
        screen_q4_contexts(a, b)
    elif cmd == "compress":
        cmd_compress(a, b)
    elif cmd == "decompress":
        cmd_decompress(a, b)
    elif cmd == "verify":
        if not os.path.isfile(b):
            sys.exit(f"input not found: {b}")
        cmd_verify(a, b)
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
