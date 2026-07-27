#!/usr/bin/env python3
"""Prune an FX4 causal donor plan using measured region payload sizes.

Because donor replay changes adaptive state, these are pathwise gains rather
than independent edge ablations. Re-run the pruned plan until it stabilizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
RECORD = struct.Struct("<HI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_plan", type=Path)
    parser.add_argument("baseline_stats", type=Path)
    parser.add_argument("assisted_stats", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--winners-csv", type=Path)
    parser.add_argument("--minimum-gain", type=int, default=6)
    return parser.parse_args()


def read_stats(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {
            int(row["region"]): row
            for row in csv.DictReader(source)
        }


def main() -> int:
    args = parse_args()
    data = args.input_plan.read_bytes()
    if len(data) < HEADER.size:
        raise RuntimeError("truncated donor plan")
    magic, version, flags, chunk, seed, stream_size, count, digest = (
        HEADER.unpack_from(data)
    )
    if magic != b"F4CP" or version != 1 or flags != 0:
        raise RuntimeError("unsupported donor plan")
    if len(data) != HEADER.size + count * RECORD.size:
        raise RuntimeError("invalid donor plan length")
    records = [
        RECORD.unpack_from(data, HEADER.size + index * RECORD.size)
        for index in range(count)
    ]

    baseline = read_stats(args.baseline_stats)
    assisted = read_stats(args.assisted_stats)
    winners = []
    for recipient, donor_offset in records:
        if recipient not in baseline or recipient not in assisted:
            continue
        base_bytes = int(baseline[recipient]["payload_bytes"])
        assisted_bytes = int(assisted[recipient]["payload_bytes"])
        gain = base_bytes - assisted_bytes
        if gain > args.minimum_gain:
            winners.append((recipient, donor_offset, base_bytes,
                            assisted_bytes, gain))

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                magic,
                version,
                flags,
                chunk,
                seed,
                stream_size,
                len(winners),
                digest,
            )
        )
        for recipient, donor_offset, _, _, _ in winners:
            output.write(RECORD.pack(recipient, donor_offset))

    csv_path = args.winners_csv or args.output_plan.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recipient_region",
                "donor_seed_offset",
                "baseline_payload_bytes",
                "assisted_payload_bytes",
                "measured_path_gain_bytes",
            ]
        )
        writer.writerows(winners)

    result = {
        "input_edges": count,
        "winning_edges": len(winners),
        "measured_path_gain_bytes": sum(row[4] for row in winners),
        "output_plan_bytes": args.output_plan.stat().st_size,
        "note": "rerun pruned plan because adaptive-state effects interact",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
