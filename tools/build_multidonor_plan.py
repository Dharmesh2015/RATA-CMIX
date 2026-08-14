#!/usr/bin/env python3
"""Build an F4CP v3 plan containing ordered variable-length donor edges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT_V3 = struct.Struct("<HIHB")
CHUNK_SIZE = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("edges_csv", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--stream-size", type=int, default=587_138_826)
    parser.add_argument(
        "--stream-sha256",
        default=(
            "7826ff63dedd526c119dda08e6e044be8"
            "fa8f6e89a55f3d6b1f3447cdfc5c1ce"
        ),
    )
    parser.add_argument("--minimum-gain", type=int, default=1)
    return parser.parse_args()


def hash_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.digest()


def varint_size(value: int) -> int:
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def zigzag_size(delta: int) -> int:
    encoded = delta * 2 if delta >= 0 else (-delta - 1) * 2 + 1
    return varint_size(encoded)


def compact_archive_size(
    edges: list[tuple[int, int, int, int, int]],
) -> int:
    """Exact bytes emitted by DonorPlan archive format v7."""
    groups: list[tuple[int, list[tuple[int, int, int, int, int]]]] = []
    for edge in edges:
        if not groups or groups[-1][0] != edge[0]:
            groups.append((edge[0], []))
        groups[-1][1].append(edge)

    # v7 version + flags, followed by the donor-group count.
    size = 2 + varint_size(len(groups))
    for recipient, donors in groups:
        size += varint_size(recipient) + varint_size(len(donors))
        previous = 0
        for index, (_recipient, donor, _length, order, _gain) in enumerate(donors):
            if order != index:
                raise SystemExit(
                    f"v7 replay order must be contiguous for region {recipient}"
                )
            shifted = donor >> 8
            size += varint_size(shifted) if index == 0 else zigzag_size(
                shifted - previous
            )
            size += 1  # length_log2
            previous = shifted
    size += varint_size(0)  # no shared expert profiles
    size += varint_size(0)  # no expert spans
    return size


def main() -> int:
    args = parse_args()
    stream_size = args.stream.stat().st_size if args.stream else args.stream_size
    digest = hash_file(args.stream) if args.stream else bytes.fromhex(
        args.stream_sha256
    )
    if len(digest) != 32:
        raise SystemExit("stream SHA-256 must contain 32 bytes")

    edges: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    seen_orders: set[tuple[int, int]] = set()
    next_order: dict[int, int] = {}
    with args.edges_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            recipient = int(row["recipient_region"])
            donor = int(row["donor_offset"])
            length = int(row["donor_length"])
            gain = int(row.get("gain_bytes") or 1)
            raw_order = row.get("replay_order")
            order = (
                int(raw_order)
                if raw_order is not None and raw_order.strip()
                else next_order.get(recipient, 0)
            )
            next_order[recipient] = max(next_order.get(recipient, 0), order + 1)
            if gain < args.minimum_gain:
                continue
            if recipient < 1 or recipient >= stream_size // CHUNK_SIZE:
                raise SystemExit(f"invalid recipient region: {recipient}")
            if donor < 0 or donor & 255:
                raise SystemExit(f"donor offset must be 256-byte aligned: {donor}")
            if length < 256 or length > 65536 or length & (length - 1):
                raise SystemExit(f"donor length must be a power of two: {length}")
            if donor + length > recipient * CHUNK_SIZE:
                raise SystemExit(
                    f"noncausal edge {donor}+{length} -> region {recipient}"
                )
            key = (recipient, donor)
            if key in seen:
                raise SystemExit(f"duplicate edge: {recipient}:{donor}")
            if order < 0 or order > 255:
                raise SystemExit(f"replay order must fit one byte: {order}")
            order_key = (recipient, order)
            if order_key in seen_orders:
                raise SystemExit(
                    f"duplicate replay order for recipient {recipient}: {order}"
                )
            seen.add(key)
            seen_orders.add(order_key)
            edges.append((recipient, donor, length, order, gain))

    edges.sort(key=lambda edge: (edge[0], edge[3], edge[1]))
    if len(edges) > 0xFFFF:
        raise SystemExit("F4CP v3 supports at most 65535 edges")

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                b"F4CP",
                3,
                0,
                CHUNK_SIZE,
                max((edge[2] for edge in edges), default=0),
                stream_size,
                len(edges),
                digest,
            )
        )
        for recipient, donor, length, order, _gain in edges:
            encoded_length = 0 if length == 65536 else length
            output.write(ASSIGNMENT_V3.pack(
                recipient, donor, encoded_length, order))

    gross_gain = sum(edge[4] for edge in edges)
    archive_plan_bytes = compact_archive_size(edges)
    print(
        json.dumps(
            {
                "format": "F4CP-v3 external / v7 compact archive",
                "edges": len(edges),
                "recipient_regions": len({edge[0] for edge in edges}),
                "distinct_donors": len({(edge[1], edge[2]) for edge in edges}),
                "gross_measured_gain_bytes": gross_gain,
                "archive_plan_bytes": archive_plan_bytes,
                "net_measured_gain_after_plan": gross_gain - archive_plan_bytes,
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
