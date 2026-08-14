#!/usr/bin/env python3
"""Build a selective post-R1 campaign around the accepted region-421 win.

The canonical accepted recipient is a fixed 1 MiB interval and is deliberately
kept separate from the page-aligned 531-group discovery map. This tool maps the
two coordinate systems, injects the seven accepted donor windows as exact
probes, and ranks later recipients using already-computed causal group edges.

No row produced here is treated as a compression win. Only the canonical F4CP
plan is accepted; every analogous recipient remains an exact-test candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Group:
    pack_id: int
    start: int
    end: int
    stream_class: int
    class_name: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Donor:
    offset: int
    length: int
    order: int
    gain: int


def read_groups(path: Path) -> list[Group]:
    result: list[Group] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            result.append(Group(
                int(row["pack_id"]),
                int(row["post_r1_start"]),
                int(row["post_r1_end"]),
                int(row["stream_class"]),
                row["class_name"],
            ))
    result.sort(key=lambda item: item.pack_id)
    for index, group in enumerate(result):
        if group.pack_id != index or group.end <= group.start:
            raise ValueError("invalid grouped recipient map")
        if index and group.start != result[index - 1].end:
            raise ValueError("grouped recipients are not contiguous")
    return result


def group_for_offset(groups: list[Group], offset: int) -> Group:
    low = 0
    high = len(groups)
    while low < high:
        middle = (low + high) // 2
        if groups[middle].end <= offset:
            low = middle + 1
        else:
            high = middle
    if low >= len(groups) or offset < groups[low].start:
        raise ValueError(f"offset outside grouped stream: {offset}")
    return groups[low]


def read_accepted(path: Path) -> tuple[int, list[Donor]]:
    donors: list[Donor] = []
    recipient_offset: int | None = None
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            current = int(row["recipient_offset"])
            if recipient_offset is None:
                recipient_offset = current
            elif current != recipient_offset:
                raise ValueError("accepted donor CSV has multiple recipients")
            donors.append(Donor(
                int(row["donor_offset"]),
                int(row["donor_length"]),
                int(row["replay_order"]),
                int(row.get("gain_bytes") or 0),
            ))
    donors.sort(key=lambda item: item.order)
    if recipient_offset is None or not donors:
        raise ValueError("accepted donor CSV is empty")
    if [item.order for item in donors] != list(range(len(donors))):
        raise ValueError("accepted donor replay order is not contiguous")
    return recipient_offset, donors


def read_edges(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            result[(int(row["source_group"]), int(row["recipient_group"]))] = row
    return result


def edge_metrics(
    edges: dict[tuple[int, int], dict[str, str]],
    sources: set[int],
    recipient: int,
) -> tuple[int, float, int]:
    best = (0, 0.0, -1)
    for source in sources:
        row = edges.get((source, recipient))
        if row is None:
            continue
        candidate = (
            int(row["overlap_windows"]),
            float(row["overlap_fraction"]),
            source,
        )
        if (candidate[1], candidate[0]) > (best[1], best[0]):
            best = candidate
    return best


def read_candidates(path: Path) -> tuple[list[str], dict[int, list[dict[str, str]]]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or ())
        expected = [
            "pack_id", "recipient_offset", "recipient_length",
            "candidate_rank", "donor_offset", "donor_length",
            "phrase_score", "signature_hits", "recipient_class",
            "donor_class",
        ]
        if fields != expected:
            raise ValueError(f"unexpected candidate header: {fields}")
        rows: dict[int, list[dict[str, str]]] = {}
        for row in reader:
            rows.setdefault(int(row["pack_id"]), []).append(row)
    return fields, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups_csv", type=Path)
    parser.add_argument("all_edges_csv", type=Path)
    parser.add_argument("ranked_candidates_csv", type=Path)
    parser.add_argument("accepted_edges_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--canonical-length", type=int, default=1 << 20)
    parser.add_argument("--top-recipients", type=int, default=64)
    parser.add_argument("--probe-recipients", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=19)
    args = parser.parse_args()

    groups = read_groups(args.groups_csv)
    edges = read_edges(args.all_edges_csv)
    canonical_start, donors = read_accepted(args.accepted_edges_csv)
    canonical_end = canonical_start + args.canonical_length
    if canonical_end > groups[-1].end:
        raise ValueError("canonical recipient exceeds post-R1 stream")

    canonical_groups = [
        group for group in groups
        if group.start < canonical_end and group.end > canonical_start
    ]
    donor_groups = {
        group_for_offset(groups, donor.offset).pack_id for donor in donors
    }
    for donor in donors:
        group = group_for_offset(groups, donor.offset)
        if donor.offset + donor.length > group.end:
            raise ValueError("accepted donor crosses a grouped boundary")
        if donor.offset + donor.length > canonical_start:
            raise ValueError("accepted donor is not causal")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output_dir / "region421_coordinate_map.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "kind", "id", "start", "end", "length", "overlap_bytes",
            "status",
        ])
        writer.writerow([
            "canonical_recipient", 421, canonical_start, canonical_end,
            args.canonical_length, args.canonical_length,
            "accepted_fixed_1mib_keep_exact",
        ])
        for group in canonical_groups:
            overlap = min(group.end, canonical_end) - max(
                group.start, canonical_start)
            writer.writerow([
                "page_aligned_group", group.pack_id, group.start, group.end,
                group.length, overlap,
                "discovery_probe_not_accepted_substitute",
            ])
        for group_id in sorted(donor_groups):
            group = groups[group_id]
            writer.writerow([
                "accepted_donor_source_group", group_id,
                group.start, group.end, group.length,
                sum(
                    donor.length for donor in donors
                    if group_for_offset(groups, donor.offset).pack_id == group_id
                ),
                "contains_accepted_donor_windows",
            ])

    canonical_source_groups = {group.pack_id for group in canonical_groups}
    candidates: list[dict[str, object]] = []
    first_causal_group = max(donor_groups) + 1
    for group in groups[first_causal_group:]:
        donor_windows, donor_fraction, donor_source = edge_metrics(
            edges, donor_groups, group.pack_id)
        canonical_windows, canonical_fraction, canonical_source = edge_metrics(
            edges, canonical_source_groups, group.pack_id)
        class_match = int(any(
            group.stream_class == source.stream_class
            for source in canonical_groups
        ))
        # This is a screen, not a byte-saving estimate. Exact accepted-source
        # overlap dominates; canonical-recipient similarity and class only
        # break ties and broaden the shortlist.
        screen_score = (
            donor_fraction * 1_000_000.0
            + canonical_fraction * 350_000.0
            + donor_windows * 2.0
            + canonical_windows * 0.5
            + class_match * 8.0
        )
        candidates.append({
            "pack_id": group.pack_id,
            "recipient_offset": group.start,
            "recipient_length": group.length,
            "recipient_class": group.stream_class,
            "class_name": group.class_name,
            "known_source_group": donor_source,
            "known_source_overlap_windows": donor_windows,
            "known_source_overlap_fraction": donor_fraction,
            "canonical_source_group": canonical_source,
            "canonical_overlap_windows": canonical_windows,
            "canonical_overlap_fraction": canonical_fraction,
            "class_match": class_match,
            "screen_score": screen_score,
        })
    candidates.sort(key=lambda row: (
        -float(row["screen_score"]),
        -int(row["known_source_overlap_windows"]),
        int(row["pack_id"]),
    ))
    source_ranked = sorted(candidates, key=lambda row: (
        -float(row["known_source_overlap_fraction"]),
        -int(row["known_source_overlap_windows"]),
        int(row["pack_id"]),
    ))
    source_path = args.output_dir / "accepted_source_recipients.csv"
    with source_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "rank", "pack_id", "recipient_offset", "recipient_length",
            "known_source_group", "overlap_windows", "overlap_fraction",
            "class_name", "status",
        ])
        for rank, row in enumerate(source_ranked[:args.top_recipients]):
            writer.writerow([
                rank,
                row["pack_id"],
                row["recipient_offset"],
                row["recipient_length"],
                row["known_source_group"],
                row["known_source_overlap_windows"],
                f"{float(row['known_source_overlap_fraction']):.9f}",
                row["class_name"],
                "accepted_bundle_probe_exact_test_required",
            ])
    similar_path = args.output_dir / "region421_like_recipients.csv"
    similar_fields = [
        "rank", "pack_id", "recipient_offset", "recipient_length",
        "recipient_class", "class_name", "known_source_group",
        "known_source_overlap_windows", "known_source_overlap_fraction",
        "canonical_source_group", "canonical_overlap_windows",
        "canonical_overlap_fraction", "class_match", "screen_score", "status",
    ]
    with similar_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=similar_fields)
        writer.writeheader()
        for rank, row in enumerate(candidates[:args.top_recipients]):
            writer.writerow({
                "rank": rank,
                **row,
                "known_source_overlap_fraction": (
                    f"{float(row['known_source_overlap_fraction']):.9f}"
                ),
                "canonical_overlap_fraction": (
                    f"{float(row['canonical_overlap_fraction']):.9f}"
                ),
                "screen_score": f"{float(row['screen_score']):.6f}",
                "status": "screen_only_exact_test_required",
            })

    fields, existing = read_candidates(args.ranked_candidates_csv)
    target_ids = {
        group.pack_id for group in canonical_groups
    }
    combined_count = (args.probe_recipients + 1) // 2
    source_count = args.probe_recipients // 2
    target_ids.update(
        int(row["pack_id"]) for row in candidates[:combined_count]
    )
    target_ids.update(
        int(row["pack_id"]) for row in source_ranked[:source_count]
    )
    seeded_path = args.output_dir / "enwik9.grouped_selective_candidates.csv"
    with seeded_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            rows = existing.get(group.pack_id, [])
            merged: list[dict[str, str]] = []
            seen: set[tuple[int, int]] = set()
            if group.pack_id in target_ids:
                top_score = max(
                    (int(row["phrase_score"]) for row in rows), default=0)
                source_edge = max(
                    (
                        edges.get((source, group.pack_id))
                        for source in donor_groups
                        if edges.get((source, group.pack_id)) is not None
                    ),
                    key=lambda row: float(row["overlap_fraction"]),
                    default=None,
                )
                signature_hits = (
                    int(source_edge["overlap_windows"]) if source_edge else 0
                )
                for donor_index, donor in enumerate(donors):
                    if donor.offset + donor.length > group.start:
                        continue
                    key = (donor.offset, donor.length)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append({
                        "pack_id": str(group.pack_id),
                        "recipient_offset": str(group.start),
                        "recipient_length": str(group.length),
                        "candidate_rank": "0",
                        "donor_offset": str(donor.offset),
                        "donor_length": str(donor.length),
                        "phrase_score": str(
                            top_score + len(donors) - donor_index),
                        "signature_hits": str(signature_hits),
                        "recipient_class": str(group.stream_class),
                        "donor_class": str(groups[max(donor_groups)].stream_class),
                    })
            for row in sorted(rows, key=lambda item: int(item["candidate_rank"])):
                key = (int(row["donor_offset"]), int(row["donor_length"]))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(row)
                if len(merged) >= args.max_candidates:
                    break
            for rank, row in enumerate(merged[:args.max_candidates]):
                row = dict(row)
                row["candidate_rank"] = str(rank)
                writer.writerow(row)

    recipients_path = Path(str(seeded_path) + ".recipients.csv")
    candidate_by_id = {
        int(row["pack_id"]): row for row in candidates
    }
    selected_ranks = [
        candidate_by_id[pack_id]
        for pack_id in target_ids
        if pack_id in candidate_by_id
    ]
    selected_ranks.sort(key=lambda row: (
        -float(row["screen_score"]),
        -int(row["known_source_overlap_windows"]),
        int(row["pack_id"]),
    ))
    for group in reversed(canonical_groups):
        selected_ranks.insert(0, {
            "pack_id": group.pack_id,
            "recipient_offset": group.start,
            "recipient_length": group.length,
            "screen_score": 1_000_000.0,
        })
    with recipients_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "rank", "pack_id", "recipient_offset", "recipient_length",
            "best_phrase_score", "best_signature_hits", "candidate_count",
        ])
        for rank, row in enumerate(selected_ranks):
            pack_id = int(row["pack_id"])
            count = sum(
                1 for candidate in existing.get(pack_id, [])
            ) + len(donors)
            writer.writerow([
                rank,
                pack_id,
                int(row["recipient_offset"]),
                int(row["recipient_length"]),
                max(1, int(float(row["screen_score"]))),
                int(row.get("known_source_overlap_windows", 0)),
                min(args.max_candidates, count),
            ])

    for suffix in (".profiles.csv", ".shared.csv"):
        source = Path(str(args.ranked_candidates_csv) + suffix)
        if source.exists():
            shutil.copyfile(source, Path(str(seeded_path) + suffix))

    summary = {
        "canonical_recipient": {
            "offset": canonical_start,
            "length": args.canonical_length,
            "grouped_recipients": [group.pack_id for group in canonical_groups],
            "group_overlap_bytes": {
                str(group.pack_id): (
                    min(group.end, canonical_end)
                    - max(group.start, canonical_start)
                )
                for group in canonical_groups
            },
            "accepted_plan": "results/accepted_region421_197360/region421.f4cp",
        },
        "accepted_donors": {
            "count": len(donors),
            "bytes": sum(donor.length for donor in donors),
            "source_groups": sorted(donor_groups),
            "windows": [
                {
                    "offset": donor.offset,
                    "length": donor.length,
                    "order": donor.order,
                }
                for donor in donors
            ],
        },
        "screened_recipients": len(candidates),
        "analog_recipients_written": min(args.top_recipients, len(candidates)),
        "accepted_donor_probe_recipients": sorted(target_ids),
        "seeded_candidates": str(seeded_path),
        "recipient_filter": str(recipients_path),
        "mapping": str(mapping_path),
        "similar_recipients": str(similar_path),
        "accepted_source_recipients": str(source_path),
        "rules": [
            "canonical fixed interval remains the accepted regression gate",
            "grouped 421/422 are probes and cannot replace the canonical result",
            "screen scores are not compression gains",
            "only exact archive savings after F4CP/F4VR bytes may be retained",
        ],
    }
    summary_path = args.output_dir / "campaign.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
