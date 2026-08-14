#!/usr/bin/env python3
"""Write canonical, auditable reports for accepted donor collections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_windows(value: str) -> list[dict[str, int]]:
    windows = []
    for index, item in enumerate(filter(None, value.split(";")), start=1):
        offset_text, length_text = item.split(":", 1)
        offset = int(offset_text)
        length = int(length_text)
        windows.append({
            "index": index,
            "offset": offset,
            "length": length,
            "end": offset + length,
        })
    return windows


def read_augment(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def optional_int(row: dict[str, str], name: str) -> int | None:
    value = row.get(name, "").strip()
    return int(value) if value else None


def render_text(result: dict) -> str:
    donors = result["donors"]
    lines = [
        f'{result["name"]}:',
        f'Recipient:             {result["recipient_offset"]:,}:'
        f'{result["recipient_length"]:,} '
        f'(end {result["recipient_end"]:,})',
        f'Donor source group:    D{result["source_group"]}',
        f'Donor count:           {len(donors)}',
        f'Donor bytes:           {result["donor_bytes"]:,}',
        "",
        f'Cold/isolated baseline: {result["baseline_archive_bytes"]:,} bytes',
        f'{len(donors)} donors:'.ljust(24) +
        f'{result["payload_bytes"]:,} bytes'.rjust(13),
        "F4CP plan:".ljust(24) + f'{result["plan_bytes"]:,} bytes'.rjust(13),
        "All-in:".ljust(24) + f'{result["all_in_bytes"]:,} bytes'.rjust(13),
        "Saving:".ljust(24) + f'{result["net_saving_bytes"]:,} bytes'.rjust(13),
        "",
        "Donor locations:",
    ]
    for donor in donors:
        lines.append(
            f'  {donor["index"]}. offset {donor["offset"]:,}, '
            f'length {donor["length"]:,}, end {donor["end"]:,}'
        )
    lines.extend([
        "",
        f'Roundtrip evidence:    {result["roundtrip"]}',
        f'Full-prefix status:    {result["full_prefix_status"]}',
    ])

    scr2 = result.get("scr2")
    if scr2:
        lines.extend([
            "",
            "SCR2 virtual replay:",
            f'  Status:              {scr2["status"]}',
            f'  Combined payload:    {scr2["payload_bytes"]:,} bytes'
            if scr2.get("payload_bytes") is not None else
            "  Combined payload:    pending",
            f'  F4VR plan:           {scr2["plan_bytes"]:,} bytes'
            if scr2.get("plan_bytes") is not None else
            "  F4VR plan:           pending",
            f'  Combined all-in:     {scr2["all_in_bytes"]:,} bytes'
            if scr2.get("all_in_bytes") is not None else
            "  Combined all-in:     pending",
            f'  Delta vs donor-only: {scr2["delta_vs_donor_bytes"]:+,} bytes'
            if scr2.get("delta_vs_donor_bytes") is not None else
            "  Delta vs donor-only: pending",
            f'  Candidate hits:      {scr2["candidate_occurrences"]:,}'
            if scr2.get("candidate_occurrences") is not None else
            "  Candidate hits:      pending",
            f'  Selected events:     {scr2["selected_events"]:,}'
            if scr2.get("selected_events") is not None else
            "  Selected events:     pending",
            f'  Roundtrip:           {scr2["roundtrip"]}',
            f'  Base-state invariant:{scr2["probability_state"]}',
            f'  Decision:            {scr2["decision"]}',
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--augment-csv", type=Path)
    args = parser.parse_args()

    augment = read_augment(args.augment_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    with args.accepted_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        donors = parse_windows(row["donor_windows"])
        baseline = int(row["baseline_archive_bytes"])
        payload = int(row["payload_bytes"])
        plan_bytes = int(row["plan_bytes"])
        all_in = payload + plan_bytes
        if all_in != int(row["all_in_bytes"]):
            raise ValueError(f'{row["name"]}: inconsistent all-in byte count')
        if baseline - all_in != int(row["net_saving_bytes"]):
            raise ValueError(f'{row["name"]}: inconsistent saving byte count')

        result = {
            "schema": "FX4_DONOR_COLLECTION_V1",
            "name": row["name"],
            "recipient_offset": int(row["recipient_offset"]),
            "recipient_length": int(row["recipient_length"]),
            "recipient_end": int(row["recipient_offset"]) +
            int(row["recipient_length"]),
            "source_group": int(row["source_group"]),
            "donor_count": len(donors),
            "donor_bytes": sum(item["length"] for item in donors),
            "donors": donors,
            "baseline_archive_bytes": baseline,
            "payload_bytes": payload,
            "plan_bytes": plan_bytes,
            "all_in_bytes": all_in,
            "net_saving_bytes": baseline - all_in,
            "roundtrip": row.get("confidence", "unknown"),
            "full_prefix_status": row.get("remaining_gate", "unknown"),
        }

        extra = augment.get(row["name"])
        if extra:
            result["scr2"] = {
                "status": extra.get("scr2_status", "unknown"),
                "payload_bytes": optional_int(extra, "scr2_payload_bytes"),
                "plan_bytes": optional_int(extra, "scr2_plan_bytes"),
                "all_in_bytes": optional_int(extra, "scr2_all_in_bytes"),
                "delta_vs_donor_bytes": optional_int(
                    extra, "scr2_delta_vs_donor_bytes"),
                "candidate_occurrences": optional_int(
                    extra, "scr2_candidate_occurrences"),
                "selected_events": optional_int(
                    extra, "scr2_selected_events"),
                "roundtrip": extra.get("scr2_roundtrip", "not_run"),
                "probability_state": extra.get(
                    "scr2_probability_state", "not_checked"),
                "decision": extra.get("scr2_decision", "pending"),
            }

        report_dir = args.output_dir / row["name"]
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (report_dir / "result.txt").write_text(
            render_text(result), encoding="utf-8")
        with (report_dir / "donors.csv").open(
                "w", newline="", encoding="utf-8") as donor_file:
            writer = csv.DictWriter(
                donor_file, fieldnames=["index", "offset", "length", "end"])
            writer.writeheader()
            writer.writerows(donors)
        index_rows.append({
            "name": result["name"],
            "recipient_offset": result["recipient_offset"],
            "recipient_length": result["recipient_length"],
            "donor_count": result["donor_count"],
            "donor_bytes": result["donor_bytes"],
            "payload_bytes": result["payload_bytes"],
            "plan_bytes": result["plan_bytes"],
            "all_in_bytes": result["all_in_bytes"],
            "net_saving_bytes": result["net_saving_bytes"],
            "full_prefix_status": result["full_prefix_status"],
        })

    if index_rows:
        with (args.output_dir / "index.csv").open(
                "w", newline="", encoding="utf-8") as index_file:
            writer = csv.DictWriter(index_file, fieldnames=index_rows[0].keys())
            writer.writeheader()
            writer.writerows(index_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
