#!/usr/bin/env python3
"""FXRL96-v1 compact model serialization shared by training and inspection."""

from __future__ import annotations

import binascii
import struct
from pathlib import Path

import numpy as np


MAGIC = b"FXRL96\x01\x00"
HEADER = struct.Struct("<8sHHHHfIII")
INPUT = 32
HIDDEN = 96
OUTPUT = 8
LINEAR = 8
CONTEXTS = 512
CONFIDENCE = 7


def quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(np.abs(values), axis=1)
    scale = np.where(maximum > 0.0, maximum / 127.0, 0.0).astype(np.float32)
    denominator = np.where(scale > 0.0, scale, 1.0)[:, None]
    quantized = np.rint(values / denominator).clip(-127, 127).astype(np.int8)
    return scale, quantized


def dequantize_rows(values: np.ndarray) -> np.ndarray:
    scale, quantized = quantize_rows(values)
    # Runtime reads fp16 scales, so evaluate exactly those widened values.
    scale = scale.astype("<f2").astype(np.float32)
    return quantized.astype(np.float32) * scale[:, None]


def _qmatrix_bytes(values: np.ndarray) -> bytes:
    scale, quantized = quantize_rows(values)
    return scale.astype("<f2").tobytes() + quantized.tobytes()


def write_blob(path: Path, *, wih: np.ndarray, whh: np.ndarray,
               bias: np.ndarray, wout: np.ndarray, bout: np.ndarray,
               wlinear: np.ndarray, context_delta: np.ndarray,
               confidence_gate: np.ndarray, correction_limit: float = 2.0,
               reset_bytes: int = 256) -> int:
    arrays = {
        "wih": (wih, (4 * HIDDEN, INPUT)),
        "whh": (whh, (4 * HIDDEN, HIDDEN)),
        "bias": (bias, (4 * HIDDEN,)),
        "wout": (wout, (OUTPUT, HIDDEN)),
        "bout": (bout, (OUTPUT,)),
        "wlinear": (wlinear, (OUTPUT, LINEAR)),
        "context_delta": (context_delta, (CONTEXTS,)),
        "confidence_gate": (confidence_gate, (CONFIDENCE,)),
    }
    for name, (array, shape) in arrays.items():
        if np.asarray(array).shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {np.asarray(array).shape}")
    if not 0.0 < correction_limit <= 8.0:
        raise ValueError("correction_limit must be in (0, 8]")
    if not 0 < reset_bytes <= 1 << 20:
        raise ValueError("reset_bytes is outside the decoder contract")

    payload = b"".join((
        _qmatrix_bytes(wih),
        _qmatrix_bytes(whh),
        np.asarray(bias, dtype="<f2").tobytes(),
        _qmatrix_bytes(wout),
        np.asarray(bout, dtype="<f2").tobytes(),
        _qmatrix_bytes(wlinear),
        np.asarray(context_delta, dtype="<f2").tobytes(),
        np.asarray(confidence_gate, dtype="<f2").tobytes(),
    ))
    header = HEADER.pack(
        MAGIC, INPUT, HIDDEN, OUTPUT, 1, float(correction_limit),
        len(payload), binascii.crc32(payload) & 0xFFFFFFFF, reset_bytes)
    path.write_bytes(header + payload)
    return len(header) + len(payload)


def inspect_blob(path: Path) -> dict[str, int | float | str]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated FXRL96 blob")
    (magic, inputs, hidden, outputs, flags, limit, payload_bytes,
     crc, reset_bytes) = HEADER.unpack_from(raw)
    payload = raw[HEADER.size:]
    if magic != MAGIC or (inputs, hidden, outputs, flags) != (
            INPUT, HIDDEN, OUTPUT, 1):
        raise ValueError("unsupported FXRL96 blob")
    if payload_bytes != len(payload):
        raise ValueError("FXRL96 payload length mismatch")
    actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
    if actual_crc != crc:
        raise ValueError("FXRL96 payload checksum mismatch")
    return {
        "format": "FXRL96-v1",
        "bytes": len(raw),
        "payload_bytes": payload_bytes,
        "crc32": f"{crc:08x}",
        "correction_limit": limit,
        "reset_bytes": reset_bytes,
    }
