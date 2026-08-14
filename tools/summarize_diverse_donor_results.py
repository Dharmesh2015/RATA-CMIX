#!/usr/bin/env python3
"""Attribute exact donor trials to page-aligned source groups."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_groups(path: Path) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            starts.append(int(row["post_r1_start"]))
            ends.append(int(row["post_r1_end"]))
    return starts, ends


def group_for(starts: list[int], ends: list[int], offset: int) -> int:
    group = bisect.bisect_right(starts, offset) - 1
    if group < 0 or offset >= ends[group]:
        raise ValueError(f"donor offset outside post-R1 groups: {offset}")
    return group


def parse_donors(text: str) -> list[tuple[int, int]]:
    if not text or text == "-":
        return []
    result: list[tuple[int, int]] = []
    for item in text.split(";"):
        offset, length = item.split(":")
        result.append((int(offset), int(length)))
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups_csv", type=Path)
    parser.add_argument("ledger_prefix", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    starts, ends = read_groups(args.groups_csv)
    trials = read_csv(Path(str(args.ledger_prefix) + ".winner_trials.csv"))
    selected = read_csv(Path(str(args.ledger_prefix) + ".winner_selected.csv"))
    marginals = read_csv(Path(str(args.ledger_prefix) + ".winner_marginals.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attributed_path = args.output_dir / "exact_trials_with_sources.csv"
    if trials:
        fields = list(trials[0]) + ["source_groups"]
        with attributed_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in trials:
                groups = [group_for(starts, ends, offset)
                          for offset, _length in parse_donors(row["donors"])]
                writer.writerow({**row, "source_groups": ";".join(
                    str(group) for group in groups)})

    positive_selected = [
        row for row in selected if int(row.get("net_gain_bytes") or 0) > 0
    ]
    positive_path = args.output_dir / "exact_positive_bundles.csv"
    if selected:
        fields = list(selected[0]) + ["source_groups"]
        with positive_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in positive_selected:
                groups = [group_for(starts, ends, offset)
                          for offset, _length in parse_donors(row["donors"])]
                writer.writerow({**row, "source_groups": ";".join(
                    str(group) for group in groups)})

    evidence: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in trials:
        if row.get("stage") != "single" or int(row.get("gain_bytes") or 0) <= 0:
            continue
        donors = parse_donors(row["donors"])
        if len(donors) != 1:
            continue
        group = group_for(starts, ends, donors[0][0])
        evidence[group]["positive_single_trials"] += 1
        evidence[group]["positive_single_raw_gain"] += int(row["gain_bytes"])
    for row in marginals:
        gain = int(row.get("marginal_gain_bytes") or 0)
        if gain <= 0:
            continue
        donor = parse_donors(row.get("removed_donor", ""))
        if len(donor) != 1:
            continue
        group = group_for(starts, ends, donor[0][0])
        evidence[group]["positive_marginals"] += 1
        evidence[group]["positive_marginal_gain"] += gain
    for row in positive_selected:
        groups = {group_for(starts, ends, offset)
                  for offset, _length in parse_donors(row["donors"])}
        for group in groups:
            evidence[group]["net_positive_bundles_involved"] += 1
            evidence[group]["bundle_net_gain_involved"] += int(
                row["net_gain_bytes"])

    source_path = args.output_dir / "exact_source_evidence.csv"
    fields = [
        "source_group", "positive_single_trials", "positive_single_raw_gain",
        "positive_marginals", "positive_marginal_gain",
        "net_positive_bundles_involved", "bundle_net_gain_involved",
    ]
    with source_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for group in sorted(evidence, key=lambda item: (
                -evidence[item]["positive_marginal_gain"],
                -evidence[item]["positive_single_raw_gain"], item)):
            writer.writerow({"source_group": group, **{
                field: evidence[group][field] for field in fields[1:]}})

    summary = {
        "exact_trials": len(trials),
        "selected_recipients": len(selected),
        "net_positive_bundles": len(positive_selected),
        "new_source_groups_with_positive_evidence": len(evidence),
        "net_selected_gain_bytes": sum(
            int(row["net_gain_bytes"]) for row in positive_selected),
        "note": "bundle gain is not additive across its source groups",
    }
    (args.output_dir / "exact_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
