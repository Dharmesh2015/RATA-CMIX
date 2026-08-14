#!/usr/bin/env python3
"""Build a source-diverse exact donor campaign over all post-R1 groups.

The cheap phrase/overlap scores in this tool are scheduling signals only.
Only warm-prefix arithmetic trials may become production donor assignments.
Donors are installed as the page-scoped probability specialist, so discovery
does not replay bytes into PPMd, LSTM, FXCM, match models, or the main mixer.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
from collections import defaultdict
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
class Candidate:
    recipient: int
    offset: int
    length: int
    phrase_score: int
    signature_hits: int
    source: int
    source_class: int
    overlap_windows: int
    overlap_fraction: float
    class_match: int

    @property
    def score(self) -> float:
        return (
            self.phrase_score * 1024.0
            + self.signature_hits * 128.0
            + self.overlap_fraction * 1_000_000.0
            + self.overlap_windows * 4.0
            + self.class_match * 64.0
        )


def read_groups(path: Path) -> list[Group]:
    groups: list[Group] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            groups.append(Group(
                int(row["pack_id"]),
                int(row["post_r1_start"]),
                int(row["post_r1_end"]),
                int(row["stream_class"]),
                row["class_name"],
            ))
    groups.sort(key=lambda group: group.pack_id)
    for index, group in enumerate(groups):
        if group.pack_id != index or group.end <= group.start:
            raise ValueError("invalid grouped recipient map")
        if index and group.start != groups[index - 1].end:
            raise ValueError("grouped recipient map is not contiguous")
    if not groups:
        raise ValueError("grouped recipient map is empty")
    return groups


def read_edges(path: Path) -> dict[tuple[int, int], tuple[int, float, int]]:
    result: dict[tuple[int, int], tuple[int, float, int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            result[(int(row["source_group"]), int(row["recipient_group"]))] = (
                int(row["overlap_windows"]),
                float(row["overlap_fraction"]),
                int(row["class_match"]),
            )
    return result


def source_group(groups: list[Group], starts: list[int], offset: int) -> int:
    index = bisect.bisect_right(starts, offset) - 1
    if index < 0 or offset >= groups[index].end:
        raise ValueError(f"donor offset outside post-R1 groups: {offset}")
    return index


def read_candidates(
    path: Path,
    groups: list[Group],
    edges: dict[tuple[int, int], tuple[int, float, int]],
    excluded_sources: set[int],
) -> tuple[list[str], dict[int, list[Candidate]]]:
    starts = [group.start for group in groups]
    result: dict[int, list[Candidate]] = defaultdict(list)
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
            raise ValueError(f"unexpected candidate CSV header: {fields}")
        for row in reader:
            recipient = int(row["pack_id"])
            offset = int(row["donor_offset"])
            length = int(row["donor_length"])
            if recipient <= 0 or recipient >= len(groups):
                continue
            group = groups[recipient]
            if (int(row["recipient_offset"]) != group.start or
                    int(row["recipient_length"]) != group.length):
                raise ValueError(f"recipient coordinates changed for G{recipient}")
            donor_group = source_group(groups, starts, offset)
            if donor_group in excluded_sources or donor_group >= recipient:
                continue
            if offset + length > groups[donor_group].end:
                continue
            overlap, fraction, class_match = edges.get(
                (donor_group, recipient), (0, 0.0, 0))
            result[recipient].append(Candidate(
                recipient, offset, length, int(row["phrase_score"]),
                int(row["signature_hits"]), donor_group,
                groups[donor_group].stream_class, overlap, fraction,
                class_match,
            ))
    return fields, result


def select_candidates(
    rows: list[Candidate], max_sources: int, per_source: int,
    max_candidates: int,
) -> list[Candidate]:
    unique: dict[tuple[int, int], Candidate] = {}
    for row in rows:
        key = (row.offset, row.length)
        if key not in unique or row.score > unique[key].score:
            unique[key] = row
    by_source: dict[int, list[Candidate]] = defaultdict(list)
    for row in unique.values():
        by_source[row.source].append(row)
    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: (-row.score, row.offset, row.length))
    source_order = sorted(by_source, key=lambda source: (
        -by_source[source][0].score,
        -by_source[source][0].overlap_fraction,
        source,
    ))[:max_sources]
    selected: list[Candidate] = []
    for depth in range(per_source):
        for source in source_order:
            if depth < len(by_source[source]):
                selected.append(by_source[source][depth])
                if len(selected) == max_candidates:
                    return selected
    return selected


def source_and_recipient_order(
    selected: dict[int, list[Candidate]], groups: list[Group],
    deferred: set[int],
) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    source_recipients: dict[int, dict[int, Candidate]] = defaultdict(dict)
    for recipient, rows in selected.items():
        for row in rows:
            current = source_recipients[row.source].get(recipient)
            if current is None or row.score > current.score:
                source_recipients[row.source][recipient] = row

    source_rows: list[dict[str, object]] = []
    for source, recipients in source_recipients.items():
        values = list(recipients.values())
        classes = {groups[row.recipient].stream_class for row in values}
        source_rows.append({
            "source_group": source,
            "source_start": groups[source].start,
            "source_length": groups[source].length,
            "source_class": groups[source].class_name,
            "candidate_recipients": len(values),
            "recipient_classes": len(classes),
            "aggregate_phrase_score": sum(row.phrase_score for row in values),
            "aggregate_overlap_windows": sum(
                row.overlap_windows for row in values),
            "best_recipient_score": max(row.score for row in values),
        })
    source_rows.sort(key=lambda row: (
        -int(row["candidate_recipients"]),
        -int(row["recipient_classes"]),
        -int(row["aggregate_overlap_windows"]),
        -int(row["aggregate_phrase_score"]),
        int(row["source_group"]),
    ))

    ordered: list[tuple[int, int]] = []
    seen: set[int] = set()
    lists = {
        int(source["source_group"]): sorted(
            source_recipients[int(source["source_group"])].values(),
            key=lambda row: (-row.score, row.recipient),
        )
        for source in source_rows
    }
    depth = 0
    while True:
        added = False
        for source in source_rows:
            source_id = int(source["source_group"])
            rows = lists[source_id]
            while depth < len(rows) and rows[depth].recipient in seen:
                rows.pop(depth)
            if depth < len(rows):
                recipient = rows[depth].recipient
                if recipient not in deferred and recipient not in seen:
                    seen.add(recipient)
                    ordered.append((recipient, source_id))
                    added = True
        if not added:
            break
        depth += 1

    remaining = sorted(
        (recipient for recipient in selected if recipient not in seen and
         recipient not in deferred),
        key=lambda recipient: (
            -selected[recipient][0].score, recipient),
    )
    for recipient in remaining:
        ordered.append((recipient, selected[recipient][0].source))
        seen.add(recipient)
    for recipient in sorted(deferred):
        if recipient in selected and recipient not in seen:
            ordered.append((recipient, selected[recipient][0].source))
    return source_rows, ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups_csv", type=Path)
    parser.add_argument("all_edges_csv", type=Path)
    parser.add_argument("ranked_candidates_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exclude-source", type=int, action="append", default=[])
    parser.add_argument("--defer-recipient", type=int, action="append", default=[])
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=24)
    args = parser.parse_args()
    if (args.max_sources <= 0 or args.per_source <= 0 or args.max_candidates <= 0):
        raise ValueError("candidate limits must be positive")

    groups = read_groups(args.groups_csv)
    edges = read_edges(args.all_edges_csv)
    fields, raw = read_candidates(
        args.ranked_candidates_csv, groups, edges, set(args.exclude_source))
    selected = {
        recipient: select_candidates(
            rows, args.max_sources, args.per_source, args.max_candidates)
        for recipient, rows in raw.items()
    }
    selected = {recipient: rows for recipient, rows in selected.items() if rows}
    source_rows, recipient_order = source_and_recipient_order(
        selected, groups, set(args.defer_recipient))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "enwik9.grouped_diverse_candidates.csv"
    source_map_path = args.output_dir / "candidate_source_map.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as output, \
            source_map_path.open("w", newline="", encoding="utf-8") as source_output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        source_writer = csv.writer(source_output)
        source_writer.writerow([
            "pack_id", "recipient_offset", "candidate_rank", "source_group",
            "source_start", "donor_offset", "donor_length", "phrase_score",
            "signature_hits", "overlap_windows", "overlap_fraction",
            "class_match", "screen_score",
        ])
        for recipient in sorted(selected):
            group = groups[recipient]
            for rank, row in enumerate(selected[recipient]):
                writer.writerow({
                    "pack_id": recipient,
                    "recipient_offset": group.start,
                    "recipient_length": group.length,
                    "candidate_rank": rank,
                    "donor_offset": row.offset,
                    "donor_length": row.length,
                    "phrase_score": row.phrase_score,
                    "signature_hits": row.signature_hits,
                    "recipient_class": group.stream_class,
                    "donor_class": row.source_class,
                })
                source_writer.writerow([
                    recipient, group.start, rank, row.source,
                    groups[row.source].start, row.offset, row.length,
                    row.phrase_score, row.signature_hits,
                    row.overlap_windows, f"{row.overlap_fraction:.9f}",
                    row.class_match, f"{row.score:.6f}",
                ])

    recipient_path = Path(str(candidate_path) + ".recipients.csv")
    with recipient_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "rank", "pack_id", "recipient_offset", "recipient_length",
            "best_phrase_score", "best_signature_hits", "candidate_count",
        ])
        for rank, (recipient, scheduling_source) in enumerate(recipient_order):
            rows = selected[recipient]
            writer.writerow([
                rank, recipient, groups[recipient].start, groups[recipient].length,
                max(row.phrase_score for row in rows),
                max(row.signature_hits for row in rows), len(rows),
            ])

    rankings_path = args.output_dir / "donor_source_rankings.csv"
    with rankings_path.open("w", newline="", encoding="utf-8") as output:
        fields_out = [
            "rank", "source_group", "source_start", "source_length",
            "source_class", "candidate_recipients", "recipient_classes",
            "aggregate_phrase_score", "aggregate_overlap_windows",
            "best_recipient_score",
        ]
        writer = csv.DictWriter(output, fieldnames=fields_out)
        writer.writeheader()
        for rank, row in enumerate(source_rows):
            writer.writerow({"rank": rank, **row})

    window_recipients: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    window_score: dict[tuple[int, int, int], float] = defaultdict(float)
    window_hits: dict[tuple[int, int, int], int] = defaultdict(int)
    for recipient, rows in selected.items():
        for row in rows:
            key = (row.source, row.offset, row.length)
            window_recipients[key].add(recipient)
            window_score[key] += row.score
            window_hits[key] += row.signature_hits
    window_rows = sorted(window_recipients, key=lambda key: (
        -len(window_recipients[key]), -window_score[key],
        key[0], key[1], key[2],
    ))
    window_rankings_path = args.output_dir / "donor_window_rankings.csv"
    with window_rankings_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "rank", "source_group", "donor_offset", "donor_length",
            "candidate_recipients", "aggregate_screen_score",
            "aggregate_signature_hits", "first_recipients",
        ])
        for rank, key in enumerate(window_rows):
            recipients = sorted(window_recipients[key])
            writer.writerow([
                rank, key[0], key[1], key[2], len(recipients),
                f"{window_score[key]:.6f}", window_hits[key],
                ";".join(str(recipient) for recipient in recipients[:16]),
            ])

    for suffix in (".profiles.csv", ".shared.csv"):
        source = Path(str(args.ranked_candidates_csv) + suffix)
        if source.exists():
            shutil.copyfile(source, Path(str(candidate_path) + suffix))

    summary = {
        "groups": len(groups),
        "eligible_recipients": len(selected),
        "candidate_rows": sum(len(rows) for rows in selected.values()),
        "distinct_source_groups": len(source_rows),
        "excluded_source_groups": sorted(set(args.exclude_source)),
        "deferred_recipients": sorted(set(args.defer_recipient)),
        "candidate_csv": str(candidate_path),
        "recipient_order_csv": str(recipient_path),
        "source_rankings_csv": str(rankings_path),
        "window_rankings_csv": str(window_rankings_path),
        "candidate_source_map_csv": str(source_map_path),
        "acceptance": (
            "exact warm-prefix payload gain must exceed compact F4CP metadata"
        ),
        "state_rule": (
            "donor profile is page scoped and never replays into base models"
        ),
    }
    (args.output_dir / "campaign.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Source-diverse post-R1 donor campaign\n\n"
        "This campaign deliberately excludes accepted source G388 during the "
        "novel-source pass. Candidate scores are screens, never compression "
        "wins. Exact warm-prefix arithmetic trials retain only positive net "
        "F4CP edges. The donor probability specialist is reset per selected "
        "group and does not update PPMd, LSTM, FXCM, match, or main-mixer "
        "state.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
