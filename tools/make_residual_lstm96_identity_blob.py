#!/usr/bin/env python3
"""Create a valid zero-correction FXRL96 blob for decoder-first parity tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from residual_lstm96_blob import (CONFIDENCE, CONTEXTS, HIDDEN, INPUT,
                                  LINEAR, OUTPUT, inspect_blob, write_blob)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--reset-bytes", type=int, default=256)
    args = parser.parse_args()
    zeros = lambda *shape: np.zeros(shape, dtype=np.float32)
    write_blob(
        args.output,
        wih=zeros(4 * HIDDEN, INPUT),
        whh=zeros(4 * HIDDEN, HIDDEN),
        bias=zeros(4 * HIDDEN),
        wout=zeros(OUTPUT, HIDDEN),
        bout=zeros(OUTPUT),
        wlinear=zeros(OUTPUT, LINEAR),
        context_delta=zeros(CONTEXTS),
        confidence_gate=np.ones(CONFIDENCE, dtype=np.float32),
        reset_bytes=args.reset_bytes,
    )
    print(json.dumps(inspect_blob(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
