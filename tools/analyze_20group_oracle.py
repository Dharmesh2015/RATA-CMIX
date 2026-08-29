#!/usr/bin/env python3
"""Rank G0-G19 shadow experts and emit metadata-aware exact-test spans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUFFIX = "_single_gain_bytes"
PLAN_FIXED_BYTES = 62
PLAN_SPAN_BYTES = 20


def write_experts(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["offset", "length", "experts", "stream_class", "profile",
              "residual_gain", "pack_id", "status"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--selective-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.oracle.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("empty span oracle")
    expert_columns = [name for name in rows[0] if name.endswith(SUFFIX)]
    if not expert_columns:
        raise SystemExit("span oracle contains no singleton expert columns")

    totals = {
        column[:-len(SUFFIX)]: sum(float(row[column]) for row in rows)
        for column in expert_columns
    }
    positive = {
        column[:-len(SUFFIX)]: sum(max(0.0, float(row[column])) for row in rows)
        for column in expert_columns
    }
    winning_groups = {
        column[:-len(SUFFIX)]: sum(float(row[column]) > 0.0 for row in rows)
        for column in expert_columns
    }

    global_name, global_gain = max(totals.items(), key=lambda item: item[1])
    global_rows: list[dict[str, str]] = []
    global_plan = PLAN_FIXED_BYTES + PLAN_SPAN_BYTES
    if global_gain > global_plan:
        begin = int(rows[0]["offset"])
        end = max(int(row["offset"]) + int(row["length"]) for row in rows)
        global_rows.append({
            "offset": str(begin), "length": str(end - begin),
            "experts": global_name, "stream_class": "mixed",
            "profile": "0", "residual_gain": "0.25", "pack_id": "all",
            "status": "oracle_global",
        })
    write_experts(args.global_csv, global_rows)

    selective_rows: list[dict[str, str]] = []
    selective_gross = 0.0
    for index, row in enumerate(rows):
        candidates = [
            (column[:-len(SUFFIX)], float(row[column]))
            for column in expert_columns
        ]
        name, gain = max(candidates, key=lambda item: item[1])
        if gain <= PLAN_SPAN_BYTES:
            continue
        selective_gross += gain
        selective_rows.append({
            "offset": row["offset"], "length": row["length"],
            "experts": name, "stream_class": "mixed", "profile": "0",
            "residual_gain": "0.25", "pack_id": str(index),
            "status": "oracle_selective",
        })
    selective_plan = PLAN_FIXED_BYTES + PLAN_SPAN_BYTES * len(selective_rows)
    if selective_gross <= selective_plan:
        selective_rows = []
    write_experts(args.selective_csv, selective_rows)

    ranking = sorted(totals, key=totals.get, reverse=True)
    report = {
        "groups": len(rows),
        "global_ranking": [
            {"expert": name, "aggregate_gain_bytes": totals[name],
             "positive_gain_bytes": positive[name],
             "winning_groups": winning_groups[name]}
            for name in ranking
        ],
        "global_candidate": global_name if global_rows else None,
        "global_predicted_net_bytes": global_gain - global_plan,
        "selective_spans": len(selective_rows),
        "selective_predicted_net_bytes":
            selective_gross - selective_plan if selective_rows else 0.0,
        "accounting": "F4CP file bytes are charged before exact testing.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
