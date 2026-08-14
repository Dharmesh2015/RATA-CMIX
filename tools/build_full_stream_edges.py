#!/usr/bin/env python3
"""Phase 1: exhaustive cheap causal-pair screening over the grouped post-R1
stream, per the "Claude two-stage full-stream search" plan.

For every causal pair (source group entirely before recipient group), score
by 32-byte non-overlapping window overlap between the two 1 MiB regions, plus
class-name match and distance. This is the literal signal donor replay
exploits (repeated byte windows), computed directly from the post-R1 bytes,
not compression.

Deliberately NOT computed here (no verified definition found in this
codebase): WRT token histogram similarity, "Q/L/M field" overlap. Do not
invent placeholder values for those columns.

Writes every pair (not just winners) plus a per-recipient top-N shortlist.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


WINDOW = 32


def load_groups(path: Path) -> list[dict]:
    groups = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups.append({
                "pack_id": int(row["pack_id"]),
                "start": int(row["post_r1_start"]),
                "end": int(row["post_r1_end"]),
                "length": int(row["post_r1_length"]),
                "class_name": row.get("class_name", "mixed") or "mixed",
            })
    groups.sort(key=lambda g: g["pack_id"])
    return groups


def window_hashes(mm: np.memmap, start: int, end: int) -> np.ndarray:
    """Non-overlapping WINDOW-byte hashes as a sorted unique uint64 array."""
    n = (end - start) // WINDOW
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    usable = n * WINDOW
    buf = np.frombuffer(mm, dtype=np.uint8, count=usable, offset=start)
    words = buf.reshape(n, WINDOW // 8, 8).view(np.uint64).reshape(n, WINDOW // 8)
    # XOR-fold the 4 uint64 words per window into one hash. Cheap, vectorized,
    # sufficient for a screening pass (false-positive collisions only inflate
    # a candidate's score slightly; they never hide a real match).
    h = words[:, 0] ^ (words[:, 1] * np.uint64(0x9E3779B97F4A7C15))
    h ^= (words[:, 2] * np.uint64(0xC2B2AE3D27D4EB4F))
    h ^= (words[:, 3] * np.uint64(0x165667B19E3779F9))
    return np.unique(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", type=Path)
    ap.add_argument("recipients_csv", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--top-n", type=int, default=32)
    ap.add_argument("--force-source", type=str, action="append", default=[],
                     help="pack_id:recipient_pack_id to force into a shortlist")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(args.recipients_csv)
    n_groups = len(groups)
    stream_bytes = groups[-1]["end"]

    mm = np.memmap(args.stream, dtype=np.uint8, mode="r", shape=(stream_bytes,))

    print(f"hashing {n_groups} groups at {WINDOW}-byte non-overlapping windows...")
    hashes: list[np.ndarray] = []
    for g in groups:
        hashes.append(window_hashes(mm, g["start"], g["end"]))
        if g["pack_id"] % 50 == 0:
            print(f"  hashed group {g['pack_id']}/{n_groups-1} "
                  f"({len(hashes[-1])} windows)")

    edges_path = args.output_dir / "all_group_edges.csv"
    shortlist_path = args.output_dir / "recipient_shortlists.csv"

    total_pairs = 0
    per_recipient: dict[int, list[tuple]] = {r["pack_id"]: [] for r in groups}

    with edges_path.open("w", newline="", encoding="utf-8") as ef:
        ew = csv.writer(ef)
        ew.writerow([
            "source_group", "recipient_group", "source_start", "recipient_start",
            "distance", "overlap_windows", "source_window_count",
            "recipient_window_count", "overlap_fraction", "class_match",
            "source_class", "recipient_class",
        ])
        for r in groups:
            r_id = r["pack_id"]
            if r_id == 0:
                continue  # no causal predecessor
            r_hash = hashes[r_id]
            r_count = len(r_hash)
            for s in groups[:r_id]:
                s_id = s["pack_id"]
                if s["end"] > r["start"]:
                    continue  # causality guard: donor end <= recipient start
                s_hash = hashes[s_id]
                s_count = len(s_hash)
                overlap = 0
                if s_count and r_count:
                    overlap = int(np.intersect1d(
                        s_hash, r_hash, assume_unique=True).shape[0])
                denom = min(s_count, r_count) or 1
                frac = overlap / denom
                class_match = 1 if s["class_name"] == r["class_name"] else 0
                distance = r["start"] - s["end"]
                row = (s_id, r_id, s["start"], r["start"], distance, overlap,
                       s_count, r_count, frac, class_match,
                       s["class_name"], r["class_name"])
                ew.writerow(row)
                per_recipient[r_id].append(
                    (frac, overlap, s_id, distance, class_match))
                total_pairs += 1
            if r_id % 50 == 0:
                print(f"  scored recipient {r_id}/{n_groups-1}, "
                      f"{total_pairs} pairs so far")

    forced: dict[int, set[int]] = {}
    for spec in args.force_source:
        s_str, r_str = spec.split(":")
        forced.setdefault(int(r_str), set()).add(int(s_str))

    with shortlist_path.open("w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["recipient_group", "rank", "source_group",
                     "overlap_fraction", "overlap_windows", "distance",
                     "class_match", "forced"])
        for r_id, cands in per_recipient.items():
            cands.sort(key=lambda t: (-t[0], -t[1]))
            top = cands[: args.top_n]
            top_sources = {c[2] for c in top}
            force_ids = forced.get(r_id, set())
            for rank, (frac, overlap, s_id, distance, class_match) in enumerate(top):
                sw.writerow([r_id, rank, s_id, f"{frac:.6f}", overlap,
                             distance, class_match, s_id in force_ids])
            for s_id in force_ids - top_sources:
                match = next((c for c in cands if c[2] == s_id), None)
                if match is None:
                    continue
                frac, overlap, _, distance, class_match = match
                sw.writerow([r_id, -1, s_id, f"{frac:.6f}", overlap,
                             distance, class_match, True])
                print(f"  FORCED source {s_id} into recipient {r_id} "
                      f"shortlist (was not naturally in top {args.top_n}: "
                      f"rank would have been "
                      f"{next(i for i,c in enumerate(cands) if c[2]==s_id)})")

    print(f"total causal pairs scored: {total_pairs}")
    print(f"wrote {edges_path}")
    print(f"wrote {shortlist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
