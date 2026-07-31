#!/usr/bin/env python3
"""Build a production F4CP plan from exact regional selections."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT = struct.Struct("<HI")
NO_DONOR = 0xFFFFFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_plan", type=Path)
    parser.add_argument("selected_csv", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--minimum-gain", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = args.candidate_plan.read_bytes()
    if len(payload) < HEADER.size:
        raise SystemExit("candidate plan is truncated")
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
    if magic != b"F4CD" or version != 1 or flags != 1:
        raise SystemExit("expected an F4CD candidate plan")
    if len(payload) != HEADER.size + count * ASSIGNMENT.size:
        raise SystemExit("candidate plan assignment size mismatch")

    allowed: set[tuple[int, int]] = set()
    offset = HEADER.size
    for _ in range(count):
        recipient, donor = ASSIGNMENT.unpack_from(payload, offset)
        offset += ASSIGNMENT.size
        allowed.add((recipient, donor))

    selected: dict[int, tuple[int, int]] = {}
    with args.selected_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            recipient = int(row["recipient_region"])
            donor = int(row["selected_donor_offset"])
            gain = int(row["gain_bytes"])
            if donor == NO_DONOR or gain < args.minimum_gain:
                continue
            if (recipient, donor) not in allowed:
                raise SystemExit(
                    f"selection {recipient}:{donor} is absent from F4CD"
                )
            selected[recipient] = (donor, gain)

    assignments = sorted(
        (recipient, donor_gain[0])
        for recipient, donor_gain in selected.items()
    )
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                b"F4CP",
                1,
                0,
                chunk_size,
                seed_size,
                stream_size,
                len(assignments),
                digest,
            )
        )
        for recipient, donor in assignments:
            output.write(ASSIGNMENT.pack(recipient, donor))

    gross_gain = sum(gain for _donor, gain in selected.values())
    summary = {
        "selected_edges": len(assignments),
        "gross_local_gain_bytes": gross_gain,
        "archive_plan_bytes": 2 + 5 * len(assignments),
        "net_local_gain_after_archive_plan": (
            gross_gain - 2 - 5 * len(assignments)
        ),
        "external_plan_bytes": args.output_plan.stat().st_size,
        "validation": "requires_combined_full_stream_roundtrip",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
