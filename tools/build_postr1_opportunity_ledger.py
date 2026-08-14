#!/usr/bin/env python3
"""Build a conservative post-R1 donor/SCR2/mini-cmix opportunity ledger.

Only rows in accepted_exact.csv are treated as exact archive wins. Other donor savings are
calibrated screening estimates, SCR2 values are raw structural opportunity,
and mini-cmix values are oracle upper bounds.  Production still requires an
exact warm-prefix arithmetic-coding trial with all plan bytes counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCR2_MAGIC = b"SCR2M1\x00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as src:
        return list(csv.DictReader(src))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data) or shift >= 64:
            raise ValueError("invalid SCR2 uvarint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def read_scr2_meta(path: Path) -> tuple[int, int, bytes, list[bytes]]:
    blob = path.read_bytes()
    if not blob.startswith(SCR2_MAGIC) or len(blob) < len(SCR2_MAGIC) + 8:
        raise ValueError(f"invalid SCR2 metadata: {path}")
    pos = len(SCR2_MAGIC)
    raw_len, expected_crc = struct.unpack_from("<II", blob, pos)
    payload = zlib.decompress(blob[pos + 8 :])
    if len(payload) != raw_len or zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
        raise ValueError("SCR2 metadata length/CRC mismatch")
    pos = 0
    version, marker, original_size = struct.unpack_from("<BBQ", payload, pos)
    pos += struct.calcsize("<BBQ")
    if version != 1:
        raise ValueError(f"unsupported SCR2 metadata version: {version}")
    original_sha = payload[pos : pos + 32]
    pos += 32
    count, pos = read_uvarint(payload, pos)
    patterns: list[bytes] = []
    for _ in range(count):
        length, pos = read_uvarint(payload, pos)
        patterns.append(payload[pos : pos + length])
        pos += length
    if pos != len(payload):
        raise ValueError("trailing SCR2 metadata")
    return marker, original_size, original_sha, patterns


@dataclass
class Group:
    group: int
    start: int
    end: int
    length: int
    stream_class: int
    class_name: str


def load_groups(path: Path) -> list[Group]:
    groups = []
    for row in read_csv(path):
        groups.append(
            Group(
                group=int(row["pack_id"]),
                start=int(row["post_r1_start"]),
                end=int(row["post_r1_end"]),
                length=int(row["post_r1_length"]),
                stream_class=int(row["stream_class"]),
                class_name=row["class_name"],
            )
        )
    groups.sort(key=lambda item: item.start)
    for index, group in enumerate(groups):
        if group.group != index or group.end - group.start != group.length:
            raise ValueError("group map is not canonical/contiguous")
        if index and groups[index - 1].end != group.start:
            raise ValueError("group map has a gap or overlap")
    return groups


def map_scr2_events(
    encoded_path: Path, meta_path: Path, groups: list[Group]
) -> tuple[dict[int, dict], dict]:
    marker, original_size, original_sha, patterns = read_scr2_meta(meta_path)
    if groups[-1].end != original_size:
        raise ValueError("SCR2 stream and group-map sizes differ")

    metrics = {
        group.group: {
            "scr2_macro_hits": 0,
            "scr2_raw_saving_bytes": 0,
            "scr2_cross_boundary_hits": 0,
            "scr2_top_macros": defaultdict(int),
        }
        for group in groups
    }
    marker_byte = bytes((marker,))
    original_pos = 0
    encoded_pos = 0
    group_index = 0
    escaped = 0
    macro_hits = 0
    # Hashing the original stream is intentionally omitted here: this pass maps
    # the already round-trip-verified SCR2 stream without expanding 587 MB.
    with encoded_path.open("rb") as src, mmap.mmap(
        src.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        size = len(data)
        while encoded_pos < size:
            hit = data.find(marker_byte, encoded_pos)
            if hit < 0:
                original_pos += size - encoded_pos
                encoded_pos = size
                break
            original_pos += hit - encoded_pos
            encoded_pos = hit
            while group_index + 1 < len(groups) and original_pos >= groups[group_index].end:
                group_index += 1
            if encoded_pos + 1 >= size:
                raise ValueError("truncated SCR2 marker")
            code = data[encoded_pos + 1]
            encoded_pos += 2
            if code == 0:
                original_pos += 1
                escaped += 1
                continue
            if code > len(patterns):
                raise ValueError(f"invalid SCR2 code {code}")
            pattern_length = len(patterns[code - 1])
            group = groups[group_index]
            entry = metrics[group.group]
            if original_pos + pattern_length <= group.end:
                entry["scr2_macro_hits"] += 1
                entry["scr2_raw_saving_bytes"] += pattern_length - 2
                entry["scr2_top_macros"][code] += pattern_length - 2
            else:
                entry["scr2_cross_boundary_hits"] += 1
            macro_hits += 1
            original_pos += pattern_length

    if original_pos != original_size:
        raise ValueError(
            f"SCR2 decoded length mismatch: {original_pos} != {original_size}"
        )
    for group in groups:
        entry = metrics[group.group]
        top = sorted(entry.pop("scr2_top_macros").items(), key=lambda x: (-x[1], x[0]))[:5]
        entry["scr2_top_macros"] = ";".join(f"M{code}:{gain}" for code, gain in top)
        entry["scr2_raw_reduction_pct"] = (
            100.0 * entry["scr2_raw_saving_bytes"] / group.length
        )
    return metrics, {
        "marker": marker,
        "patterns": len(patterns),
        "macro_hits": macro_hits,
        "escaped_markers": escaped,
        "decoded_size": original_pos,
        "original_sha256": original_sha.hex(),
    }


def load_mini_class_priors(path: Path) -> tuple[dict[int, dict], dict]:
    totals: dict[int, dict] = defaultdict(lambda: {"bits": 0, "positive": 0.0, "signed": 0.0})
    rows = read_csv(path)
    for row in rows:
        stream_class = int(row["stream_class"])
        gain = float(row["mini_gain_bytes"])
        totals[stream_class]["bits"] += int(row["bits"])
        totals[stream_class]["signed"] += gain
        totals[stream_class]["positive"] += max(0.0, gain)
    priors = {}
    for stream_class, values in totals.items():
        modeled_bytes = values["bits"] / 8.0
        priors[stream_class] = {
            **values,
            "modeled_bytes": modeled_bytes,
            "positive_density": values["positive"] / max(1.0, modeled_bytes),
        }
    return priors, {"trace_rows": len(rows), "classes": len(priors)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--accepted-edges", type=Path, required=True)
    parser.add_argument("--accepted-source-recipients", type=Path, required=True)
    parser.add_argument("--diverse-candidates", type=Path, required=True)
    parser.add_argument("--scr2-encoded", type=Path, required=True)
    parser.add_argument("--scr2-meta", type=Path, required=True)
    parser.add_argument("--mini-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = load_groups(args.groups)
    group_by_id = {group.group: group for group in groups}
    accepted = json.loads(args.accepted.read_text(encoding="utf-8"))
    accepted_edges = read_csv(args.accepted_edges)
    accepted_gain = int(accepted["baseline_entropy_bytes"]) - int(
        accepted["accepted_entropy_bytes"]
    )
    accepted_plan = int(accepted["compact_v7_plan_bytes"])
    accepted_overlap = 34.0
    accepted_windows = ";".join(
        f'{row["donor_offset"]}:{row["donor_length"]}' for row in accepted_edges
    )

    scr2_metrics, scr2_summary = map_scr2_events(
        args.scr2_encoded, args.scr2_meta, groups
    )
    mini_priors, mini_summary = load_mini_class_priors(args.mini_oracle)

    source_pairs: dict[tuple[int, int], dict] = {}
    # The validated D388 profile gets stronger, but still discounted, estimates.
    for row in read_csv(args.accepted_source_recipients):
        recipient = int(row["pack_id"])
        if recipient not in group_by_id:
            continue
        overlap = int(row["overlap_windows"])
        raw_high = accepted_gain * min(2.0, overlap / accepted_overlap)
        expected = raw_high * 0.25
        source_pairs[(recipient, 388)] = {
            "recipient_group": recipient,
            "source_group": 388,
            "donor_windows": accepted_windows,
            "donor_window_count": 7,
            "donor_bytes": sum(int(edge["donor_length"]) for edge in accepted_edges),
            "phrase_score": 0,
            "signature_hits": overlap,
            "overlap_windows": overlap,
            "screen_score": float(row.get("overlap_fraction", 0.0)),
            "donor_estimate_low": 0.0,
            "donor_estimate_expected": expected,
            "donor_estimate_high": raw_high,
            "evidence": "validated_source_proxy",
        }

    # New sources have no exact winner yet.  Aggregate their best ranked windows
    # and apply a deliberately smaller calibration factor.
    aggregated: dict[tuple[int, int], dict] = {}
    for row in read_csv(args.diverse_candidates):
        recipient = int(row["pack_id"])
        source = int(row["source_group"])
        if recipient not in group_by_id or source not in group_by_id:
            continue
        key = (recipient, source)
        item = aggregated.setdefault(
            key,
            {
                "recipient_group": recipient,
                "source_group": source,
                "windows": [],
                "phrase_score": 0,
                "signature_hits": 0,
                "overlap_windows": int(row["overlap_windows"]),
                "screen_score": 0.0,
            },
        )
        window = (int(row["donor_offset"]), int(row["donor_length"]))
        if window not in item["windows"]:
            item["windows"].append(window)
        item["phrase_score"] += int(row["phrase_score"])
        item["signature_hits"] += int(row["signature_hits"])
        item["screen_score"] = max(item["screen_score"], float(row["screen_score"]))

    for key, item in aggregated.items():
        overlap = item["overlap_windows"]
        raw_high = accepted_gain * min(2.0, math.sqrt(max(1.0, overlap) / accepted_overlap))
        expected = raw_high * 0.10
        candidate = {
            **item,
            "donor_windows": ";".join(f"{offset}:{length}" for offset, length in item["windows"]),
            "donor_window_count": len(item["windows"]),
            "donor_bytes": sum(length for _, length in item["windows"]),
            "donor_estimate_low": 0.0,
            "donor_estimate_expected": expected,
            "donor_estimate_high": raw_high,
            "evidence": "uncalibrated_source_proxy",
        }
        candidate.pop("windows", None)
        existing = source_pairs.get(key)
        if existing is None or candidate["donor_estimate_expected"] > existing["donor_estimate_expected"]:
            source_pairs[key] = candidate

    source_use_count: dict[int, int] = defaultdict(int)
    for candidate in source_pairs.values():
        source_use_count[candidate["source_group"]] += 1

    candidate_rows = []
    for candidate in source_pairs.values():
        recipient = group_by_id[candidate["recipient_group"]]
        source = group_by_id[candidate["source_group"]]
        uses = source_use_count[source.group]
        # Accepted v7 used 25 bytes for seven windows plus one recipient.  Model
        # this as 18 bytes registered once and seven bytes per recipient.
        amortized_plan = 7.0 + 18.0 / max(1, uses)
        donor_net = candidate["donor_estimate_expected"] - amortized_plan
        scr2 = scr2_metrics[recipient.group]
        mini = mini_priors.get(recipient.stream_class, {})
        mini_high = float(mini.get("positive_density", 0.0)) * recipient.length
        row = {
            **candidate,
            "recipient_start": recipient.start,
            "recipient_length": recipient.length,
            "recipient_class": recipient.class_name,
            "source_start": source.start,
            "source_class": source.class_name,
            "source_reuse_candidates": uses,
            "estimated_plan_bytes_amortized": amortized_plan,
            "donor_estimate_net": donor_net,
            **scr2,
            "scr2_entropy_estimate": 0.0,
            "mini11_oracle_high": mini_high,
            "mini11_expected_net": 0.0,
            "combined_expected_net": max(0.0, donor_net),
            "confidence": "medium" if candidate["evidence"] == "validated_source_proxy" else "low",
            "validation": "exact_warm_prefix_required",
        }
        candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda row: (
            -row["combined_expected_net"],
            -row["scr2_raw_saving_bytes"],
            row["recipient_group"],
            row["source_group"],
        )
    )
    for rank, row in enumerate(candidate_rows, 1):
        row["rank"] = rank

    best_by_recipient: dict[int, dict] = {}
    for row in candidate_rows:
        best_by_recipient.setdefault(row["recipient_group"], row)

    recipient_rows = []
    for group in groups:
        best = best_by_recipient.get(group.group)
        scr2 = scr2_metrics[group.group]
        mini = mini_priors.get(group.stream_class, {})
        mini_high = float(mini.get("positive_density", 0.0)) * group.length
        recipient_rows.append(
            {
                "recipient_group": group.group,
                "recipient_start": group.start,
                "recipient_length": group.length,
                "recipient_class": group.class_name,
                "best_source_group": "" if best is None else best["source_group"],
                "best_donor_windows": "" if best is None else best["donor_windows"],
                "donor_evidence": "none" if best is None else best["evidence"],
                "donor_expected_gross": 0.0 if best is None else best["donor_estimate_expected"],
                "donor_expected_net": 0.0 if best is None else best["donor_estimate_net"],
                "donor_high_bound": 0.0 if best is None else best["donor_estimate_high"],
                **scr2,
                "scr2_entropy_estimate": 0.0,
                "mini11_oracle_high": mini_high,
                "mini11_expected_net": 0.0,
                "recommended_trial": (
                    "donor+scr2_oracle"
                    if best is not None and scr2["scr2_raw_reduction_pct"] >= 3.75
                    else "donor"
                    if best is not None
                    else "scr2_oracle"
                    if scr2["scr2_raw_reduction_pct"] >= 3.75
                    else "defer"
                ),
                "validation": "exact_warm_prefix_required",
            }
        )
    recipient_rows.sort(
        key=lambda row: (
            -max(0.0, row["donor_expected_net"]),
            -row["scr2_raw_saving_bytes"],
            -row["mini11_oracle_high"],
        )
    )
    for rank, row in enumerate(recipient_rows, 1):
        row["priority_rank"] = rank

    source_rows = []
    by_source: dict[int, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        by_source[row["source_group"]].append(row)
    for source_group, rows in by_source.items():
        source = group_by_id[source_group]
        positive = [row for row in rows if row["donor_estimate_net"] > 0]
        source_rows.append(
            {
                "source_group": source_group,
                "source_start": source.start,
                "source_length": source.length,
                "source_class": source.class_name,
                "candidate_recipients": len(rows),
                "positive_expected_recipients": len(positive),
                "estimated_expected_net_sum": sum(row["donor_estimate_net"] for row in positive),
                "estimated_high_gross_sum": sum(row["donor_estimate_high"] for row in rows),
                "best_recipient": rows[0]["recipient_group"],
                "best_expected_net": max(row["donor_estimate_net"] for row in rows),
                "evidence": (
                    "validated_profile" if source_group in (2, 388)
                    else "proxy_only"
                ),
            }
        )
    source_rows.sort(key=lambda row: -row["estimated_expected_net_sum"])
    for rank, row in enumerate(source_rows, 1):
        row["rank"] = rank

    c421_accepted = {
        "name": "C421-D388-7",
        "recipient_offset": accepted["recipient_offset"],
        "recipient_length": accepted["input_bytes"],
        "source_group": 388,
        "donor_windows": accepted_windows,
        "baseline_archive_bytes": accepted["baseline_archive_bytes"],
        "payload_bytes": accepted["verified_archive_bytes"],
        "plan_bytes": accepted_plan,
        "all_in_bytes": accepted["accounted_all_in_bytes"],
        "net_saving_bytes": accepted["net_saving_vs_baseline_archive"],
        "confidence": "exact_isolated_roundtrip",
        "remaining_gate": "full_prefix_validation",
    }
    g389_accepted = {
        "name": "G389-D388-7",
        "recipient_offset": 407838930,
        "recipient_length": 1041373,
        "source_group": 388,
        "donor_windows": accepted_windows,
        "baseline_archive_bytes": 290286,
        "payload_bytes": 287496,
        "plan_bytes": 25,
        "all_in_bytes": 287521,
        "net_saving_bytes": 2765,
        "confidence": "exact_isolated_roundtrip",
        "remaining_gate": "full_prefix_validation",
    }
    g3_accepted = {
        "name": "G3-D2-2",
        "recipient_offset": 3146668,
        "recipient_length": 1048846,
        "source_group": 2,
        "donor_windows": "2097920:16384;3080960:65536",
        "baseline_archive_bytes": 126072,
        "payload_bytes": 120548,
        "plan_bytes": 25,
        "all_in_bytes": 120573,
        "net_saving_bytes": 5499,
        "confidence": "exact_isolated_roundtrip",
        "remaining_gate": "full_prefix_validation",
    }
    accepted_rows = [c421_accepted, g389_accepted, g3_accepted]

    queue_rows = []
    queued_recipients: set[int] = set()

    def enqueue(mode: str, row: dict, note: str) -> None:
        recipient = int(row["recipient_group"])
        queue_rows.append(
            {
                "queue_rank": len(queue_rows) + 1,
                "mode": mode,
                "recipient_group": recipient,
                "recipient_start": row["recipient_start"],
                "recipient_length": row["recipient_length"],
                "recipient_class": row["recipient_class"],
                "source_group": row.get("source_group", ""),
                "donor_windows": row.get("donor_windows", ""),
                "estimated_net_bytes": row.get("donor_estimate_net", 0.0),
                "estimated_high_gross_bytes": row.get("donor_estimate_high", 0.0),
                "scr2_raw_saving_bytes": row.get("scr2_raw_saving_bytes", 0),
                "mini11_oracle_high": row.get("mini11_oracle_high", 0.0),
                "evidence": row.get("evidence", "screen_only"),
                "note": note,
                "acceptance": "exact warm-prefix net win after all side data",
            }
        )
        queued_recipients.add(recipient)

    # First exploit the one validated donor profile on new recipients.
    for row in candidate_rows:
        if row["source_group"] != 388 or row["recipient_group"] in (3, 389, 421, 422):
            continue
        enqueue("donor_D388_7", row, "reuse validated seven-window source profile")
        if sum(item["mode"] == "donor_D388_7" for item in queue_rows) >= 32:
            break

    # Reuse the independently validated D2 two-window profile on later list
    # recipients. Its second window ends at 3,146,496.
    d2_windows = "2097920:16384;3080960:65536"
    for source_row in candidate_rows:
        if (
            source_row["source_group"] != 2
            or source_row["recipient_group"] in (3, 389, 421, 422)
            or source_row["recipient_start"] < 3146496
        ):
            continue
        row = dict(source_row)
        row["donor_windows"] = d2_windows
        row["evidence"] = "validated_source_proxy"
        enqueue("donor_D2_2", row, "reuse validated two-window source profile")
        if sum(item["mode"] == "donor_D2_2" for item in queue_rows) >= 16:
            break

    # Establish genuinely new sources without collapsing onto one donor hub.
    queued_sources: set[int] = {2, 388}
    for row in candidate_rows:
        recipient = row["recipient_group"]
        source = row["source_group"]
        if (
            source in queued_sources
            or recipient in queued_recipients
            or recipient in (3, 389, 421, 422)
        ):
            continue
        enqueue("donor_new_source", row, "refine source to an ordered 4-8 window bundle")
        queued_sources.add(source)
        if sum(item["mode"] == "donor_new_source" for item in queue_rows) >= 64:
            break

    # SCR2 is an oracle candidate only: retain cost-positive virtual events.
    scr2_ranked = sorted(
        recipient_rows,
        key=lambda row: (-row["scr2_raw_reduction_pct"], -row["scr2_raw_saving_bytes"]),
    )
    for recipient in scr2_ranked:
        if recipient["recipient_group"] in queued_recipients:
            continue
        enqueue(
            "scr2_virtual_replay_oracle",
            recipient,
            "retain only arithmetic-cost-positive macro events",
        )
        if sum(item["mode"] == "scr2_virtual_replay_oracle" for item in queue_rows) >= 32:
            break

    # Mini-cmix stays diagnostic until selection and binary growth break even.
    mini_ranked = sorted(recipient_rows, key=lambda row: -row["mini11_oracle_high"])
    for recipient in mini_ranked:
        if recipient["recipient_group"] in queued_recipients:
            continue
        enqueue(
            "mini11_oracle_only",
            recipient,
            "trace all 11 subsets; require executable-adjusted net gain",
        )
        if sum(item["mode"] == "mini11_oracle_only" for item in queue_rows) >= 16:
            break
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_edges.csv", list(candidate_rows[0].keys()), candidate_rows)
    write_csv(output / "recipient_priority.csv", list(recipient_rows[0].keys()), recipient_rows)
    write_csv(output / "source_reuse.csv", list(source_rows[0].keys()), source_rows)
    write_csv(output / "accepted_exact.csv", list(c421_accepted.keys()), accepted_rows)
    write_csv(output / "validation_queue.csv", list(queue_rows[0].keys()), queue_rows)

    donor_expected_total = sum(
        max(0.0, row["donor_estimate_net"]) for row in best_by_recipient.values()
    )
    donor_high_total = sum(row["donor_estimate_high"] for row in best_by_recipient.values())
    summary = {
        "groups": len(groups),
        "candidate_edges": len(candidate_rows),
        "candidate_sources": len(source_rows),
        "recipients_with_donor_candidate": len(best_by_recipient),
        "exact_accepted": accepted_rows,
        "screening_projection": {
            "donor_expected_net_bytes": donor_expected_total,
            "donor_high_gross_bytes": donor_high_total,
            "scr2_raw_saving_bytes": sum(
                item["scr2_raw_saving_bytes"] for item in scr2_metrics.values()
            ),
            "scr2_entropy_saving_bytes": 0,
            "mini11_entropy_saving_bytes": 0,
        },
        "scr2_mapping": scr2_summary,
        "mini_oracle": mini_summary,
        "calibration": {
            "accepted_entropy_gain_bytes": accepted_gain,
            "accepted_plan_bytes": accepted_plan,
            "accepted_overlap_windows": accepted_overlap,
            "validated_source_discount": 0.25,
            "new_source_discount": 0.10,
            "warning": "proxy estimates are ranking aids, not additive archive savings",
        },
        "production_acceptance": (
            "Keep an edge only when exact warm-prefix payload gain exceeds all "
            "profile, assignment, span, executable, and transform metadata bytes."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Post-R1 opportunity ledger\n\n"
        "`accepted_exact.csv` contains exact isolated roundtrip results only.\n\n"
        "`candidate_edges.csv` ranks donor/source pairs. Expected and high values "
        "are calibrated screening estimates, never submission claims.\n\n"
        "`recipient_priority.csv` gives one next trial per canonical page-aligned "
        "group. SCR2 columns are exact raw macro opportunity mapped from the "
        "round-trip-verified SCR2 stream, but its entropy estimate remains zero "
        "until a cost-positive virtual-replay oracle confirms a win.\n\n"
        "`mini11_oracle_high` is a C421 class-prior upper bound. Its expected net "
        "is deliberately zero because the current 11-model oracle does not pay "
        "for selection plus executable cost.\n\n"
        "All proposed modes are recipient scoped. Donor bytes and virtual replay "
        "must update only their specialist/event coder; original bytes continue "
        "through PPMd, LSTM, FXCM, match models, and the main mixer unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
