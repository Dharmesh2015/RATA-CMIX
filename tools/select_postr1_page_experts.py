#!/usr/bin/env python3
"""Select net-positive post-R1 page experts from one oracle pass.

The output is a candidate expert CSV for build_postr1_portfolio.py. Cross-
entropy is only a shortlist: the resulting archive must still beat the exact
canonical post-R1 gate before any span is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = {
    "mini_cmix": ("mini_cmix", "mini_gain_bytes", 1 << 16),
    "donor_profile": ("donor_profile", "donor_gain_bytes", 1 << 15),
    "mini_donor": (
        "mini_cmix+donor_profile",
        "combined_gain_bytes",
        (1 << 16) | (1 << 15),
    ),
}

STREAM_CLASSES = {
    0: "prose", 1: "xml", 2: "number", 3: "date", 4: "table",
    5: "reference", 6: "url", 7: "identifier", 8: "template",
    9: "list", 10: "mixed",
}


def varint_size(value: int) -> int:
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--span-source", type=Path,
        help="original expert CSV used to recover exact span lengths",
    )
    parser.add_argument(
        "--modes", default="mini_cmix,donor_profile,mini_donor",
        help="comma-separated candidate modes",
    )
    parser.add_argument("--min-net-bytes", type=float, default=1.0)
    parser.add_argument("--fixed-plan-cost", type=int, default=7)
    parser.add_argument("--profile-bank-cost", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = set(allowed) - set(MODES)
    if unknown:
        raise SystemExit(f"unknown modes: {', '.join(sorted(unknown))}")

    with args.oracle.open(newline="", encoding="utf-8-sig") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["offset"]))
    source_lengths: dict[int, int] = {}
    if args.span_source:
        with args.span_source.open(newline="", encoding="utf-8-sig") as source:
            source_lengths = {
                int(row["offset"]): int(row["length"])
                for row in csv.DictReader(source)
            }

    selected: list[dict[str, object]] = []
    previous_end = 0
    total_gain = 0.0
    total_span_cost = 0
    for row in rows:
        offset = int(row["offset"])
        length = source_lengths.get(offset, int(row["length"]))
        if length <= 0 or offset < previous_end:
            raise SystemExit(f"invalid/overlapping oracle span at {offset}")
        mode = max(allowed, key=lambda item: float(row[MODES[item][1]]))
        experts, gain_field, mask = MODES[mode]
        gain = float(row[gain_field])
        span_cost = (
            varint_size(offset - previous_end)
            + varint_size(length)
            + varint_size(mask)
            + 2
        )
        if gain - span_cost < args.min_net_bytes:
            continue
        profile_id = int(row["profile_id"])
        selected.append({
            "offset": offset,
            "length": length,
            "experts": experts,
            "stream_class": STREAM_CLASSES[int(row["stream_class"])],
            "profile": profile_id & 63,
            "residual_gain": 0.25 * (1 + ((profile_id >> 6) & 3)),
            "oracle_gain_bytes": f"{gain:.6f}",
            "estimated_span_cost": span_cost,
            "estimated_net_gain": f"{gain - span_cost:.6f}",
            "status": "candidate_exact_test_required",
        })
        previous_end = offset + length
        total_gain += gain
        total_span_cost += span_cost

    fixed_cost = (args.fixed_plan_cost + args.profile_bank_cost) if selected else 0
    total_net = total_gain - total_span_cost - fixed_cost
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "offset", "length", "experts", "stream_class", "profile",
        "residual_gain", "oracle_gain_bytes", "estimated_span_cost",
        "estimated_net_gain", "status",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "oracle": str(args.oracle),
        "selected_spans": len(selected),
        "oracle_gain_bytes": total_gain,
        "span_plan_bytes": total_span_cost,
        "fixed_and_profile_bank_bytes": fixed_cost,
        "estimated_net_gain_bytes": total_net,
        "acceptance_rule": "retain only after exact archive is below 197304 bytes",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
