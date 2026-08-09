#!/usr/bin/env python3
"""Build a multi-candidate causal donor discovery plan.

F4CD is a discovery-only format. It may contain several donor offsets for one
recipient. The production F4CP format remains one selected donor per recipient.
Future-donor edges are preserved in a separate CSV for the later topological
region-order format; they are never silently discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path


MAGIC = b"F4CD"
VERSION = 1
FLAGS = 1
CHUNK_SIZE = 1 << 20
SEED_SIZE = 4096
HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT = struct.Struct("<HI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("edges", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--deferred-csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--stream-size", type=int, default=587_138_826)
    parser.add_argument(
        "--stream-sha256",
        default=(
            "7826ff63dedd526c119dda08e6e044be8"
            "fa8f6e89a55f3d6b1f3447cdfc5c1ce"
        ),
    )
    parser.add_argument("--max-per-recipient", type=int, default=32)
    parser.add_argument("--minimum-proxy-gain", type=int, default=1)
    return parser.parse_args()


def hash_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.digest()


def proxy_gain(row: dict[str, str]) -> int:
    for name in ("deflate_gain_bytes", "proxy_gain_bytes", "gain_bytes"):
        value = row.get(name)
        if value not in (None, ""):
            return int(value)
    return 0


def main() -> int:
    args = parse_args()
    if args.max_per_recipient < 1:
        raise SystemExit("--max-per-recipient must be positive")

    stream_size = args.stream.stat().st_size if args.stream else args.stream_size
    digest = hash_file(args.stream) if args.stream else bytes.fromhex(
        args.stream_sha256
    )
    if len(digest) != 32:
        raise SystemExit("stream SHA-256 must contain 32 bytes")
    complete_regions = stream_size // CHUNK_SIZE

    causal: dict[int, list[dict[str, str]]] = defaultdict(list)
    deferred_rows: list[tuple[int, int, int]] = []
    duplicate = 0
    seen: set[tuple[int, int]] = set()
    with args.edges.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            recipient = int(row["recipient_region"])
            donor_text = row.get("donor_seed_offset") or row.get("donor_offset")
            if donor_text is None:
                raise SystemExit("edge CSV requires donor_seed_offset or donor_offset")
            donor = int(donor_text)
            gain = proxy_gain(row)
            key = (recipient, donor)
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            if (
                recipient >= complete_regions
                or donor < 0
                or donor + SEED_SIZE > stream_size
                or gain < args.minimum_proxy_gain
            ):
                continue
            if donor + SEED_SIZE > recipient * CHUNK_SIZE:
                deferred_rows.append((recipient, donor, gain))
                continue
            row["_gain"] = str(gain)
            row["_donor"] = str(donor)
            causal[recipient].append(row)

    selected: list[tuple[int, int, int, int]] = []
    for recipient, rows in causal.items():
        rows.sort(
            key=lambda row: (
                -int(row["_gain"]),
                int(row["_donor"]),
            )
        )
        for rank, row in enumerate(rows[: args.max_per_recipient]):
            selected.append(
                (
                    recipient,
                    int(row["_donor"]),
                    int(row["_gain"]),
                    rank,
                )
            )
    selected.sort(key=lambda item: (item[0], item[3], item[1]))
    if len(selected) > 0xFFFF:
        raise SystemExit("F4CD supports at most 65535 candidate edges")

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                MAGIC,
                VERSION,
                FLAGS,
                CHUNK_SIZE,
                SEED_SIZE,
                stream_size,
                len(selected),
                digest,
            )
        )
        for recipient, donor, _gain, _rank in selected:
            output.write(ASSIGNMENT.pack(recipient, donor))

    csv_path = args.output_csv or args.output_plan.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recipient_region",
                "recipient_offset",
                "candidate_rank",
                "donor_seed_offset",
                "proxy_gain_bytes",
            ]
        )
        for recipient, donor, gain, rank in selected:
            writer.writerow(
                [recipient, recipient * CHUNK_SIZE, rank, donor, gain]
            )

    deferred_path = args.deferred_csv or args.output_plan.with_suffix(
        ".noncausal.csv"
    )
    deferred_rows.sort(key=lambda item: (item[0], -item[2], item[1]))
    with deferred_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recipient_region",
                "recipient_offset",
                "donor_seed_offset",
                "proxy_gain_bytes",
                "status",
            ]
        )
        for recipient, donor, gain in deferred_rows:
            writer.writerow(
                [
                    recipient,
                    recipient * CHUNK_SIZE,
                    donor,
                    gain,
                    "deferred_requires_topological_region_order",
                ]
            )

    summary = {
        "format": "F4CD",
        "stream_bytes": stream_size,
        "stream_sha256": digest.hex(),
        "complete_regions": complete_regions,
        "candidate_edges": len(selected),
        "recipient_regions": len({item[0] for item in selected}),
        "distinct_donors": len({item[1] for item in selected}),
        "noncausal_edges_deferred": len(deferred_rows),
        "noncausal_csv": str(deferred_path),
        "duplicates_removed": duplicate,
        "maximum_candidates_per_recipient": args.max_per_recipient,
        "plan_bytes": args.output_plan.stat().st_size,
        "validation": "proxy_shortlist_pending_exact_fx4",
    }
    summary_path = args.summary or args.output_plan.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
