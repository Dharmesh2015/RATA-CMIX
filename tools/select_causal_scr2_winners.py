#!/usr/bin/env python3
"""Select a shared causal-SCR2 phrase profile and page-aligned winning ranges.

The input is FX4_CAUSAL_SCR2_STATS output. Selection is offline and does not
enter S1. The reported net gain charges the complete causal F4VR payload:
version, prior, reusable profile ID, range count, and range varints. An
unrecognized research mask additionally pays for its 128-bit custom mask.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


BUILTIN_PROFILES = {
    frozenset(range(1, 129)): 0,
    frozenset({1, 3, 14, 22, 23, 40, 46, 47, 59, 63, 76, 78}): 1,
    frozenset({1, 3, 7, 11, 15, 38, 76}): 2,
}
BUILTIN_FIXED_PLAN_BYTES = 1 + 1 + 1 + 4
CUSTOM_FIXED_PLAN_BYTES = BUILTIN_FIXED_PLAN_BYTES + 16


@dataclass(frozen=True)
class Region:
    index: int
    offset: int
    length: int


def varint_bytes(value: int) -> int:
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def covers_complete_stream(regions: set[int], metadata: dict[int, Region]) -> bool:
    if regions != set(metadata):
        return False
    expected = 0
    for index in sorted(regions, key=lambda value: metadata[value].offset):
        region = metadata[index]
        if region.offset != expected:
            return False
        expected += region.length
    return expected > 0


def profile_id(patterns: set[int]) -> int | None:
    return BUILTIN_PROFILES.get(frozenset(patterns))


def merged_ranges(regions: set[int],
                  metadata: dict[int, Region]) -> list[Region]:
    merged: list[Region] = []
    for index in sorted(regions, key=lambda value: metadata[value].offset):
        region = metadata[index]
        if merged and merged[-1].offset + merged[-1].length == region.offset:
            previous = merged[-1]
            merged[-1] = Region(previous.index, previous.offset,
                                previous.length + region.length)
        else:
            merged.append(region)
    return merged


def plan_bytes(regions: set[int], patterns: set[int],
               metadata: dict[int, Region]) -> int:
    if not regions or not patterns:
        return 0
    total = (BUILTIN_FIXED_PLAN_BYTES if profile_id(patterns) is not None
             else CUSTOM_FIXED_PLAN_BYTES)
    if covers_complete_stream(regions, metadata):
        return total
    previous_end = 0
    for region in merged_ranges(regions, metadata):
        total += varint_bytes(region.offset - previous_end)
        total += varint_bytes(region.length)
        previous_end = region.offset + region.length
    return total


def gross_bits(regions: set[int], patterns: set[int],
               gains: dict[tuple[int, int], float]) -> float:
    return sum(gains.get((region, pattern), 0.0)
               for region in regions for pattern in patterns)


def objective(regions: set[int], patterns: set[int],
              metadata: dict[int, Region],
              gains: dict[tuple[int, int], float]) -> float:
    if not regions or not patterns:
        return 0.0
    return gross_bits(regions, patterns, gains) / 8.0 - plan_bytes(
        regions, patterns, metadata)


def optimize(seed: set[int], all_patterns: set[int],
             metadata: dict[int, Region],
             gains: dict[tuple[int, int], float]) -> tuple[set[int], set[int]]:
    regions = set(seed)
    patterns = set(all_patterns)
    changed = True
    while changed:
        changed = False
        selected_patterns = {
            pattern for pattern in all_patterns
            if sum(gains.get((region, pattern), 0.0) for region in regions) > 0
        }
        if selected_patterns != patterns:
            patterns = selected_patterns
            changed = True
        if not patterns:
            return set(), set()

        # Exact greedy additions/removals include changing varint costs.
        while True:
            current = objective(regions, patterns, metadata, gains)
            best_delta = 0.0
            best_action: tuple[str, int] | None = None
            for region in metadata:
                if region in regions:
                    candidate = regions - {region}
                    delta = objective(candidate, patterns, metadata, gains) - current
                    action = ("remove", region)
                else:
                    candidate = regions | {region}
                    delta = objective(candidate, patterns, metadata, gains) - current
                    action = ("add", region)
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_action = action
            if best_action is None:
                break
            if best_action[0] == "add":
                regions.add(best_action[1])
            else:
                regions.remove(best_action[1])
            changed = True
    return regions, patterns


def optimize_regions(seed: set[int], patterns: set[int],
                     metadata: dict[int, Region],
                     gains: dict[tuple[int, int], float]) -> set[int]:
    regions = set(seed)
    while True:
        current = objective(regions, patterns, metadata, gains)
        best_delta = 0.0
        best_action: tuple[str, int] | None = None
        for region in metadata:
            if region in regions:
                candidate = regions - {region}
                action = ("remove", region)
            else:
                candidate = regions | {region}
                action = ("add", region)
            delta = objective(candidate, patterns, metadata, gains) - current
            if delta > best_delta + 1e-9:
                best_delta = delta
                best_action = action
        if best_action is None:
            return regions
        if best_action[0] == "add":
            regions.add(best_action[1])
        else:
            regions.remove(best_action[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path)
    parser.add_argument("--ranges", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--single-offset", type=int, default=0,
                        help="offset for legacy one-range stats")
    parser.add_argument("--single-length", type=int,
                        help="length for legacy one-range stats")
    args = parser.parse_args()

    metadata: dict[int, Region] = {}
    gains: dict[tuple[int, int], float] = {}
    patterns: set[int] = set()
    with args.stats.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        ranged = {"range", "offset", "length"}.issubset(
            reader.fieldnames or [])
        if not ranged and (args.single_length is None or
                           args.single_length <= 0 or
                           args.single_offset < 0):
            parser.error("legacy stats require --single-length and a valid "
                         "--single-offset")
        for row in reader:
            region = int(row["range"]) if ranged else 0
            metadata[region] = Region(
                region,
                int(row["offset"]) if ranged else args.single_offset,
                int(row["length"]) if ranged else args.single_length,
            )
            pattern = int(row["pattern"])
            patterns.add(pattern)
            gains[(region, pattern)] = float(row["estimated_gain_bits"])

    seeds = [set(metadata)]
    ranked_regions = sorted(
        metadata,
        key=lambda region: sum(max(0.0, gains.get((region, pattern), 0.0))
                               for pattern in patterns),
        reverse=True,
    )
    seeds.extend({region} for region in ranked_regions[:32])
    best_regions: set[int] = set()
    best_patterns: set[int] = set()
    best_score = 0.0
    for seed in seeds:
        candidate_regions, candidate_patterns = optimize(
            seed, patterns, metadata, gains)
        score = objective(candidate_regions, candidate_patterns,
                          metadata, gains)
        if score > best_score:
            best_score = score
            best_regions = candidate_regions
            best_patterns = candidate_patterns

    # A nearby custom optimum can lose only because its 128-bit mask must be
    # stored. Explicitly evaluate every profile that is already compiled into
    # S1; snapping to one of these profiles removes that archive tax.
    for built_patterns in BUILTIN_PROFILES:
        candidate_patterns = set(built_patterns)
        ranked_for_profile = sorted(
            metadata,
            key=lambda region: sum(
                gains.get((region, pattern), 0.0)
                for pattern in candidate_patterns),
            reverse=True,
        )
        profile_seeds = [set(metadata)]
        profile_seeds.extend({region} for region in ranked_for_profile[:32])
        for seed in profile_seeds:
            candidate_regions = optimize_regions(
                seed, candidate_patterns, metadata, gains)
            score = objective(candidate_regions, candidate_patterns,
                              metadata, gains)
            if score > best_score:
                best_score = score
                best_regions = candidate_regions
                best_patterns = candidate_patterns

    args.ranges.parent.mkdir(parents=True, exist_ok=True)
    with args.ranges.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["offset", "length"])
        for region in merged_ranges(best_regions, metadata):
            writer.writerow([region.offset, region.length])
    args.patterns.write_text(
        ",".join(str(value) for value in sorted(best_patterns)) + "\n",
        encoding="ascii",
    )

    gross = gross_bits(best_regions, best_patterns, gains) / 8.0
    side = plan_bytes(best_regions, best_patterns, metadata)
    selected_profile = profile_id(best_patterns)
    report = {
        "selected_regions": len(best_regions),
        "encoded_ranges": len(merged_ranges(best_regions, metadata)),
        "selected_patterns": len(best_patterns),
        "patterns": sorted(best_patterns),
        "built_in_profile_id": selected_profile,
        "gross_estimated_bytes": gross,
        "complete_plan_bytes": side,
        "net_estimated_bytes": gross - side,
        "range_indices": sorted(best_regions),
        "rule": "Exact compression and roundtrip remain mandatory.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
