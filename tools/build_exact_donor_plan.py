#!/usr/bin/env python3
"""Build an F4CP plan from positive exact donor-discovery decisions."""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT = struct.Struct("<HI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_plan", type=Path)
    parser.add_argument("winner_csv", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--minimum-gain", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = args.trial_plan.read_bytes()
    if len(payload) < HEADER.size:
        raise SystemExit("trial plan is truncated")
    (
        magic,
        version,
        flags,
        chunk_size,
        seed_size,
        stream_size,
        count,
        digest,
    ) = HEADER.unpack_from(payload)
    if magic != b"F4CP" or version != 1 or flags != 0:
        raise SystemExit("unsupported trial plan")
    expected = HEADER.size + count * ASSIGNMENT.size
    if len(payload) != expected:
        raise SystemExit("trial plan assignment size mismatch")

    allowed: dict[int, int] = {}
    offset = HEADER.size
    for _ in range(count):
        recipient, donor = ASSIGNMENT.unpack_from(payload, offset)
        offset += ASSIGNMENT.size
        allowed[recipient] = donor

    winners: dict[int, int] = {}
    with args.winner_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["winner"] != "donor":
                continue
            gain = int(row["gain_bytes"])
            if gain < args.minimum_gain:
                continue
            recipient = int(row["recipient_region"])
            donor = int(row["donor_offset"])
            if allowed.get(recipient) != donor:
                raise SystemExit(
                    f"winner {recipient}:{donor} is absent from trial plan"
                )
            winners[recipient] = donor

    selected = sorted(winners.items())
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                magic,
                version,
                flags,
                chunk_size,
                seed_size,
                stream_size,
                len(selected),
                digest,
            )
        )
        for recipient, donor in selected:
            output.write(ASSIGNMENT.pack(recipient, donor))

    gross_gain = 0
    with args.winner_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if (
                row["winner"] == "donor"
                and int(row["recipient_region"]) in winners
            ):
                gross_gain += int(row["gain_bytes"])
    print(f"exact_winners={len(selected)}")
    print(f"gross_path_gain_bytes={gross_gain}")
    print(f"archive_plan_bytes={2 + 5 * len(selected)}")
    print(f"external_plan_bytes={args.output_plan.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
