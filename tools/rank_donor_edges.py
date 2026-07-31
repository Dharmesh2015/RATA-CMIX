#!/usr/bin/env python3
"""Build complete donor/recipient rankings from the durable discovery ledger."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_HEADER = [
    "recipient_region",
    "recipient_offset",
    "candidate_rank",
    "donor_offset",
    "baseline_payload_bytes",
    "donor_payload_bytes",
    "gain_bytes",
    "status",
]
REGION_BYTES = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank every tested donor/recipient edge, including ties, losses, "
            "and failed trials."
        )
    )
    parser.add_argument("ledger", type=Path, help="Durable edges.csv ledger")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: ledger directory)",
    )
    return parser.parse_args()


def read_complete_csv_text(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    dropped_partial_row = bool(raw) and not raw.endswith(b"\n")
    if dropped_partial_row:
        newline = raw.rfind(b"\n")
        raw = raw[: newline + 1] if newline >= 0 else b""
    return raw.decode("utf-8"), dropped_partial_row


def read_edges(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    text, dropped_partial_row = read_complete_csv_text(path)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != EXPECTED_HEADER:
        raise ValueError(
            f"unexpected ledger header: {reader.fieldnames!r}; "
            f"expected {EXPECTED_HEADER!r}"
        )

    by_edge: dict[tuple[int, int], dict[str, Any]] = {}
    malformed_rows = 0
    duplicate_rows = 0
    for source_row, row in enumerate(reader, start=2):
        try:
            edge = {
                "recipient_region": int(row["recipient_region"]),
                "recipient_offset": int(row["recipient_offset"]),
                "candidate_rank": int(row["candidate_rank"]),
                "donor_offset": int(row["donor_offset"]),
                "baseline_payload_bytes": int(row["baseline_payload_bytes"]),
                "donor_payload_bytes": int(row["donor_payload_bytes"]),
                "gain_bytes": int(row["gain_bytes"]),
                "status": row["status"].strip(),
                "source_row": source_row,
            }
        except (KeyError, TypeError, ValueError):
            malformed_rows += 1
            continue

        edge["valid"] = edge["status"] != "error"
        if not edge["valid"]:
            edge["outcome"] = "ERROR"
        elif edge["gain_bytes"] > 0:
            edge["outcome"] = "WIN"
        elif edge["gain_bytes"] == 0:
            edge["outcome"] = "TIE"
        else:
            edge["outcome"] = "LOSS"
        edge["gain_bpb"] = edge["gain_bytes"] * 8.0 / REGION_BYTES

        key = (edge["recipient_region"], edge["donor_offset"])
        if key in by_edge:
            duplicate_rows += 1
        by_edge[key] = edge

    diagnostics = {
        "dropped_partial_row": int(dropped_partial_row),
        "malformed_rows": malformed_rows,
        "duplicate_rows": duplicate_rows,
    }
    return list(by_edge.values()), diagnostics


def edge_sort_key(edge: dict[str, Any]) -> tuple[Any, ...]:
    if not edge["valid"]:
        return (
            1,
            0,
            0,
            edge["recipient_region"],
            edge["candidate_rank"],
            edge["donor_offset"],
        )
    return (
        0,
        -edge["gain_bytes"],
        edge["donor_payload_bytes"],
        edge["recipient_region"],
        edge["candidate_rank"],
        edge["donor_offset"],
    )


def atomic_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def ranked_edge_rows(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recipient_rank: dict[tuple[int, int], int] = {}
    by_recipient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_recipient[edge["recipient_region"]].append(edge)
    for region_edges in by_recipient.values():
        for rank, edge in enumerate(sorted(region_edges, key=edge_sort_key), start=1):
            recipient_rank[
                (edge["recipient_region"], edge["donor_offset"])
            ] = rank

    result = []
    for global_rank, edge in enumerate(sorted(edges, key=edge_sort_key), start=1):
        row = dict(edge)
        row["global_exact_rank"] = global_rank
        row["recipient_exact_rank"] = recipient_rank[
            (edge["recipient_region"], edge["donor_offset"])
        ]
        row["gain_bpb"] = f"{edge['gain_bpb']:.9f}"
        result.append(row)
    return result


def donor_summaries(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[edge["donor_offset"]].append(edge)

    summaries = []
    for donor_offset, donor_edges in grouped.items():
        valid = [edge for edge in donor_edges if edge["valid"]]
        positives = [edge for edge in valid if edge["gain_bytes"] > 0]
        ties = [edge for edge in valid if edge["gain_bytes"] == 0]
        losses = [edge for edge in valid if edge["gain_bytes"] < 0]
        best = max(
            valid,
            key=lambda edge: (
                edge["gain_bytes"],
                -edge["donor_payload_bytes"],
                -edge["recipient_region"],
            ),
            default=None,
        )
        worst = min(
            valid,
            key=lambda edge: (
                edge["gain_bytes"],
                edge["donor_payload_bytes"],
                edge["recipient_region"],
            ),
            default=None,
        )
        net_gain = sum(edge["gain_bytes"] for edge in valid)
        positive_gain = sum(edge["gain_bytes"] for edge in positives)
        summaries.append(
            {
                "donor_offset": donor_offset,
                "tested_edges": len(donor_edges),
                "valid_edges": len(valid),
                "positive_edges": len(positives),
                "tie_edges": len(ties),
                "negative_edges": len(losses),
                "error_edges": len(donor_edges) - len(valid),
                "total_positive_gain_bytes": positive_gain,
                "net_gain_bytes": net_gain,
                "average_gain_bytes": (
                    f"{net_gain / len(valid):.6f}" if valid else ""
                ),
                "best_gain_bytes": best["gain_bytes"] if best else "",
                "best_recipient_region": best["recipient_region"] if best else "",
                "best_recipient_offset": best["recipient_offset"] if best else "",
                "worst_gain_bytes": worst["gain_bytes"] if worst else "",
                "worst_recipient_region": worst["recipient_region"] if worst else "",
                "worst_recipient_offset": worst["recipient_offset"] if worst else "",
            }
        )

    summaries.sort(
        key=lambda row: (
            -row["total_positive_gain_bytes"],
            -row["positive_edges"],
            -row["net_gain_bytes"],
            row["donor_offset"],
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["donor_potential_rank"] = rank
    return summaries


def recipient_summaries(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[edge["recipient_region"]].append(edge)

    summaries = []
    for region, recipient_edges in grouped.items():
        valid = [edge for edge in recipient_edges if edge["valid"]]
        positives = [edge for edge in valid if edge["gain_bytes"] > 0]
        ties = [edge for edge in valid if edge["gain_bytes"] == 0]
        losses = [edge for edge in valid if edge["gain_bytes"] < 0]
        best = max(
            valid,
            key=lambda edge: (
                edge["gain_bytes"],
                -edge["donor_payload_bytes"],
                -edge["candidate_rank"],
            ),
            default=None,
        )
        worst = min(
            valid,
            key=lambda edge: (
                edge["gain_bytes"],
                edge["donor_payload_bytes"],
                edge["candidate_rank"],
            ),
            default=None,
        )
        summaries.append(
            {
                "recipient_region": region,
                "recipient_offset": recipient_edges[0]["recipient_offset"],
                "candidate_edges": len(recipient_edges),
                "valid_edges": len(valid),
                "positive_edges": len(positives),
                "tie_edges": len(ties),
                "negative_edges": len(losses),
                "error_edges": len(recipient_edges) - len(valid),
                "baseline_payload_bytes": (
                    valid[0]["baseline_payload_bytes"] if valid else ""
                ),
                "best_donor_offset": best["donor_offset"] if best else "",
                "best_payload_bytes": best["donor_payload_bytes"] if best else "",
                "best_gain_bytes": best["gain_bytes"] if best else "",
                "best_gain_bpb": f"{best['gain_bpb']:.9f}" if best else "",
                "worst_donor_offset": worst["donor_offset"] if worst else "",
                "worst_gain_bytes": worst["gain_bytes"] if worst else "",
            }
        )

    summaries.sort(
        key=lambda row: (
            -(row["best_gain_bytes"] if row["best_gain_bytes"] != "" else -(1 << 60)),
            row["recipient_region"],
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["recipient_opportunity_rank"] = rank
    return summaries


def outcome_counts(edges: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"WIN": 0, "TIE": 0, "LOSS": 0, "ERROR": 0}
    for edge in edges:
        counts[edge["outcome"]] += 1
    return counts


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.ledger.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    edges, diagnostics = read_edges(args.ledger)
    ranked = ranked_edge_rows(edges)
    donors = donor_summaries(edges)
    recipients = recipient_summaries(edges)

    edge_fields = [
        "global_exact_rank",
        "recipient_exact_rank",
        "outcome",
        "recipient_region",
        "recipient_offset",
        "candidate_rank",
        "donor_offset",
        "baseline_payload_bytes",
        "donor_payload_bytes",
        "gain_bytes",
        "gain_bpb",
        "status",
        "source_row",
    ]
    atomic_csv(output_dir / "all_edges_ranked.csv", edge_fields, ranked)
    atomic_csv(
        output_dir / "all_edges_by_recipient.csv",
        edge_fields,
        sorted(
            ranked,
            key=lambda row: (
                row["recipient_region"],
                row["recipient_exact_rank"],
            ),
        ),
    )

    donor_fields = [
        "donor_potential_rank",
        "donor_offset",
        "tested_edges",
        "valid_edges",
        "positive_edges",
        "tie_edges",
        "negative_edges",
        "error_edges",
        "total_positive_gain_bytes",
        "net_gain_bytes",
        "average_gain_bytes",
        "best_gain_bytes",
        "best_recipient_region",
        "best_recipient_offset",
        "worst_gain_bytes",
        "worst_recipient_region",
        "worst_recipient_offset",
    ]
    atomic_csv(output_dir / "donor_rankings.csv", donor_fields, donors)

    recipient_fields = [
        "recipient_opportunity_rank",
        "recipient_region",
        "recipient_offset",
        "candidate_edges",
        "valid_edges",
        "positive_edges",
        "tie_edges",
        "negative_edges",
        "error_edges",
        "baseline_payload_bytes",
        "best_donor_offset",
        "best_payload_bytes",
        "best_gain_bytes",
        "best_gain_bpb",
        "worst_donor_offset",
        "worst_gain_bytes",
    ]
    atomic_csv(
        output_dir / "recipient_rankings.csv",
        recipient_fields,
        recipients,
    )

    counts = outcome_counts(edges)
    summary = {
        "ledger": str(args.ledger.resolve()),
        "unique_tested_edges": len(edges),
        "expected_total_edges": 9819,
        "remaining_edges": max(0, 9819 - len(edges)),
        "recipient_count": len(recipients),
        "unique_donor_count": len(donors),
        "outcomes": counts,
        **diagnostics,
    }
    atomic_json(output_dir / "ranking_summary.json", summary)

    print(
        "ranked_edges={unique_tested_edges} remaining={remaining_edges} "
        "wins={WIN} ties={TIE} losses={LOSS} errors={ERROR} "
        "donors={unique_donor_count} recipients={recipient_count}".format(
            **summary, **counts
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
