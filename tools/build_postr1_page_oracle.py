#!/usr/bin/env python3
"""Build non-overlapping F4CP oracle spans from the canonical post-R1 page map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("packs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--coordinate-shift", type=int, default=0)
    parser.add_argument(
        "--experts", default="oracle+mini_cmix+donor_profile",
    )
    parser.add_argument("--profile", type=int, default=0)
    parser.add_argument("--residual-gain", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.length <= 0 or args.start < 0 or args.coordinate_shift < 0:
        raise SystemExit("start, length and coordinate shift must be non-negative")
    end = args.start + args.length
    selected: list[dict[str, object]] = []
    with args.packs.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            page_start = int(row["post_r1_start"])
            page_end = int(row["post_r1_end"])
            if page_start < args.start or page_end > end:
                continue
            selected.append({
                "offset": page_start - args.start + args.coordinate_shift,
                "length": page_end - page_start,
                "experts": args.experts,
                "stream_class": row.get("class_name", "mixed") or "mixed",
                "profile": args.profile,
                "residual_gain": args.residual_gain,
                "pack_id": row["pack_id"],
                "status": "oracle_only",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "offset", "length", "experts", "stream_class", "profile",
        "residual_gain", "pack_id", "status",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f"selected {len(selected)} complete post-R1 page packs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
