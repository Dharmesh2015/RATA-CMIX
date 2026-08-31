#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from residual_lstm96_blob import inspect_blob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("blob", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_blob(args.blob), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
