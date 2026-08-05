#!/usr/bin/env python3
"""Rank complete post-R1 MiB recipients for exact warm FX4 testing.

The ranking combines existing structural and donor proxy measurements. It is a
search-priority list, not a substitute for full-prefix arithmetic coding.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


CHUNK_SIZE = 1 << 20
STREAM_SIZE = 587_138_826
COMPLETE_REGIONS = STREAM_SIZE // CHUNK_SIZE


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-proxy",
        type=Path,
        default=workspace
        / "special_scanner/all_disjoint_bpb_out/all_chunks_progress.csv",
    )
    parser.add_argument(
        "--structural",
        type=Path,
        default=workspace
        / "special_scanner/postwrt_readonly_analyzer/context4_122.csv",
    )
    parser.add_argument(
        "--donor-proxy",
        type=Path,
        default=repo / "research/donor_graph_post_r1_20260727/useful_edges.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "research/postr1_selective_priority.csv",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def normalized(value: float, low: float, high: float) -> float:
    return 0.0 if high <= low else (value - low) / (high - low)


def main() -> int:
    args = parse_args()

    chunk_rows: dict[int, dict[str, str]] = {}
    for row in read_rows(args.chunk_proxy):
        offset = int(row["offset"])
        size = int(row["size"])
        region = offset // CHUNK_SIZE
        if (
            offset % CHUNK_SIZE == 0
            and size == CHUNK_SIZE
            and 0 <= region < COMPLETE_REGIONS
        ):
            chunk_rows[region] = row

    structural_rows: dict[int, dict[str, str]] = {}
    for row in read_rows(args.structural):
        offset = int(row["offset"])
        region = offset // CHUNK_SIZE
        if (
            offset % CHUNK_SIZE == 0
            and int(row["size"]) == CHUNK_SIZE
            and 0 <= region < COMPLETE_REGIONS
        ):
            structural_rows[region] = row

    donor_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.donor_proxy):
        region = int(row["recipient_region"])
        if 0 <= region < COMPLETE_REGIONS and row["donor_precedes"] == "1":
            donor_rows[region].append(row)

    bpb_values = [float(chunk_rows[r]["bpb"]) for r in chunk_rows]
    if len(bpb_values) != COMPLETE_REGIONS:
        missing = sorted(set(range(COMPLETE_REGIONS)) - set(chunk_rows))
        raise SystemExit(f"chunk proxy is missing complete regions: {missing[:10]}")
    bpb_low, bpb_high = min(bpb_values), max(bpb_values)

    structural_raw: dict[int, float] = {}
    donor_raw: dict[int, float] = {}
    for region in range(COMPLETE_REGIONS):
        structural = structural_rows.get(region, {})
        macro = float(structural.get("macro_estimated_saving", 0) or 0)
        qfield = float(structural.get("qfield_estimated_saving", 0) or 0)
        line_delta = float(
            structural.get("line_delta_estimated_saving", 0) or 0
        )
        structural_raw[region] = max(0.0, macro + qfield + line_delta)
        edges = donor_rows.get(region, [])
        max_gain = max(
            (float(edge["projected_fx4_gain_bytes"]) for edge in edges),
            default=0.0,
        )
        donor_raw[region] = max_gain + 12.0 * math.log2(1.0 + len(edges))

    structural_high = max(structural_raw.values(), default=1.0)
    donor_high = max(donor_raw.values(), default=1.0)
    records: list[dict[str, object]] = []
    for region in range(COMPLETE_REGIONS):
        chunk = chunk_rows[region]
        structural = structural_rows.get(region, {})
        edges = donor_rows.get(region, [])
        bpb = float(chunk["bpb"])
        hard = normalized(bpb, bpb_low, bpb_high)
        structural_score = normalized(
            structural_raw[region], 0.0, structural_high
        )
        donor_score = normalized(donor_raw[region], 0.0, donor_high)
        composite = 0.60 * hard + 0.25 * structural_score + 0.15 * donor_score
        max_gain = max(
            (float(edge["projected_fx4_gain_bytes"]) for edge in edges),
            default=0.0,
        )
        if structural_score >= 0.45:
            mode = "cost-positive virtual replay, then donor profile"
        elif donor_score >= 0.35:
            mode = "seven-donor profile, then beam refinement"
        else:
            mode = "seven-donor profile with baseline fallback"
        records.append(
            {
                "region": region,
                "offset": region * CHUNK_SIZE,
                "proxy_bpb": bpb,
                "proxy_archive_bytes": int(chunk["archive_bytes"]),
                "hardness_score": hard,
                "structural_score": structural_score,
                "donor_score": donor_score,
                "composite_score": composite,
                "macro_estimated_saving": int(
                    float(structural.get("macro_estimated_saving", 0) or 0)
                ),
                "qfield_estimated_saving": int(
                    float(structural.get("qfield_estimated_saving", 0) or 0)
                ),
                "line_delta_estimated_saving": int(
                    float(
                        structural.get("line_delta_estimated_saving", 0) or 0
                    )
                ),
                "causal_donor_candidates": len(edges),
                "max_projected_donor_gain": max_gain,
                "recommended_first_mode": mode,
                "forced_reason": "",
            }
        )

    by_region = {int(record["region"]): record for record in records}
    by_region[421]["forced_reason"] = "pinned seven-donor seed"
    by_region[137]["forced_reason"] = "known difficult-region checkpoint"
    remainder = sorted(
        (record for record in records if record["region"] not in (421, 137)),
        key=lambda record: (
            -float(record["composite_score"]),
            -float(record["proxy_bpb"]),
            int(record["region"]),
        ),
    )
    ranked = [by_region[421], by_region[137], *remainder]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "region",
        "offset",
        "forced_reason",
        "proxy_bpb",
        "proxy_archive_bytes",
        "hardness_score",
        "structural_score",
        "donor_score",
        "composite_score",
        "macro_estimated_saving",
        "qfield_estimated_saving",
        "line_delta_estimated_saving",
        "causal_donor_candidates",
        "max_projected_donor_gain",
        "recommended_first_mode",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for priority, record in enumerate(ranked, 1):
            output = dict(record)
            output["priority"] = priority
            writer.writerow(output)

    easiest = min(records, key=lambda record: float(record["proxy_bpb"]))
    hardest = max(records, key=lambda record: float(record["proxy_bpb"]))
    print(f"wrote {len(ranked)} recipients to {args.output}")
    print(
        "proxy easiest: "
        f"region={easiest['region']} offset={easiest['offset']} "
        f"bpb={float(easiest['proxy_bpb']):.9f}"
    )
    print(
        "proxy hardest: "
        f"region={hardest['region']} offset={hardest['offset']} "
        f"bpb={float(hardest['proxy_bpb']):.9f}"
    )
    print("first 12 search priorities:")
    for priority, record in enumerate(ranked[:12], 1):
        print(
            f"{priority:3d} region={int(record['region']):3d} "
            f"offset={int(record['offset']):9d} "
            f"proxy_bpb={float(record['proxy_bpb']):.6f} "
            f"score={float(record['composite_score']):.4f} "
            f"{record['forced_reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
