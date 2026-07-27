#!/usr/bin/env python3
"""Build a compact earlier-donor-only FX4 replay plan.

The input CSV is a screening result. This tool deliberately preserves the
prepared stream order and rejects future donors, so the decoder can rebuild
every 4 KiB seed from bytes it has already decoded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"F4CP"
VERSION = 1
CHUNK_SIZE = 1 << 20
SEED_SIZE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path)
    parser.add_argument("edges", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--minimum-proxy-gain", type=int, default=6)
    parser.add_argument("--max-edges", type=int, default=0)
    return parser.parse_args()


def stream_sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.digest()


def main() -> int:
    args = parse_args()
    stream_size = args.stream.stat().st_size
    complete_regions = stream_size // CHUNK_SIZE

    best: dict[int, dict[str, str]] = {}
    with args.edges.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            recipient = int(row["recipient_region"])
            donor_offset = int(row["donor_seed_offset"])
            gain = int(row["deflate_gain_bytes"])
            if recipient >= complete_regions or gain <= args.minimum_proxy_gain:
                continue
            recipient_offset = recipient * CHUNK_SIZE
            if donor_offset + SEED_SIZE > recipient_offset:
                continue
            current = best.get(recipient)
            if current is None or gain > int(current["deflate_gain_bytes"]):
                best[recipient] = row

    selected = sorted(
        best.values(),
        key=lambda row: (
            -int(row["deflate_gain_bytes"]),
            int(row["recipient_region"]),
        ),
    )
    if args.max_edges:
        selected = selected[: args.max_edges]
    selected.sort(key=lambda row: int(row["recipient_region"]))

    digest = stream_sha256(args.stream)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(MAGIC)
        output.write(
            struct.pack(
                "<HHIIQH",
                VERSION,
                0,
                CHUNK_SIZE,
                SEED_SIZE,
                stream_size,
                len(selected),
            )
        )
        output.write(digest)
        for row in selected:
            recipient = int(row["recipient_region"])
            donor_offset = int(row["donor_seed_offset"])
            output.write(struct.pack("<HI", recipient, donor_offset))

    csv_path = args.output_csv or args.output_plan.with_suffix(".csv")
    fields = [
        "recipient_region",
        "recipient_offset",
        "donor_region",
        "donor_seed_offset",
        "seed_size",
        "deflate_gain_bytes",
        "validation",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "recipient_region": row["recipient_region"],
                    "recipient_offset": row["recipient_offset"],
                    "donor_region": row["donor_region"],
                    "donor_seed_offset": row["donor_seed_offset"],
                    "seed_size": SEED_SIZE,
                    "deflate_gain_bytes": row["deflate_gain_bytes"],
                    "validation": "proxy_selected_pending_exact_fx4",
                }
            )

    summary = {
        "stream": str(args.stream),
        "stream_bytes": stream_size,
        "stream_sha256": digest.hex(),
        "complete_regions": complete_regions,
        "causal_edges": len(selected),
        "distinct_donors": len(
            {int(row["donor_seed_offset"]) for row in selected}
        ),
        "proxy_gross_gain_bytes": sum(
            int(row["deflate_gain_bytes"]) for row in selected
        ),
        "plan_bytes": args.output_plan.stat().st_size,
        "validation": "proxy_only_exact_fx4_run_required",
    }
    summary_path = args.summary or args.output_plan.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
