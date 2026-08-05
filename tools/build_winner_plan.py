#!/usr/bin/env python3
"""Build a production F4CP-v3 plan from winner-search selections."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT_V3 = struct.Struct("<HIHB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_plan", type=Path)
    parser.add_argument("selected_csv", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--minimum-net-gain", type=int, default=1)
    return parser.parse_args()


def parse_donors(text: str) -> list[tuple[int, int]]:
    if not text or text == "-":
        return []
    donors = []
    for item in text.split(";"):
        offset_text, length_text = item.split(":", 1)
        donors.append((int(offset_text), int(length_text)))
    return donors


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
        _seed_size,
        stream_size,
        _count,
        digest,
    ) = HEADER.unpack_from(payload)
    if magic != b"F4CD" or flags != 1 or version not in (1, 2, 3):
        raise SystemExit("expected an F4CD discovery plan")

    latest: dict[int, tuple[int, list[tuple[int, int]]]] = {}
    with args.selected_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            region = int(row["recipient_region"])
            net_gain = int(row["net_gain_bytes"])
            latest[region] = (net_gain, parse_donors(row["donors"]))

    assignments: list[tuple[int, int, int, int]] = []
    gross_gain = 0
    for region, (net_gain, donors) in sorted(latest.items()):
        if net_gain < args.minimum_net_gain or not donors:
            continue
        gross_gain += net_gain + 3 + 7 * len(donors)
        recipient_offset = region * chunk_size
        for order, (offset, length) in enumerate(donors):
            if offset & 255:
                raise SystemExit(f"unaligned donor offset: {offset}")
            if length < 256 or length > 65536 or length & (length - 1):
                raise SystemExit(f"invalid donor length: {length}")
            if offset + length > recipient_offset:
                raise SystemExit(
                    f"noncausal donor {offset}:{length} for region {region}"
                )
            assignments.append((region, offset, length, order))

    if len(assignments) > 0xFFFF:
        raise SystemExit("F4CP-v3 supports at most 65535 assignments")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                b"F4CP",
                3,
                0,
                chunk_size,
                max((item[2] for item in assignments), default=0),
                stream_size,
                len(assignments),
                digest,
            )
        )
        for assignment in assignments:
            output.write(ASSIGNMENT_V3.pack(*assignment))

    archive_bytes = 3 + 7 * len(assignments)
    print(
        json.dumps(
            {
                "format": "F4CP-v3",
                "assignments": len(assignments),
                "recipient_regions": len({item[0] for item in assignments}),
                "gross_exact_gain_bytes": gross_gain,
                "archive_plan_bytes": archive_bytes,
                "net_exact_gain_bytes": gross_gain - archive_bytes,
                "external_plan_bytes": args.output_plan.stat().st_size,
                "stream_bytes": stream_size,
                "stream_sha256": digest.hex(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
