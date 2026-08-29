#!/usr/bin/env python3
"""Build an F4VR plan from exact FX4 byte costs and structural macros.

The plan never changes predictor history. Selected bytes are omitted from the
arithmetic stream, stored once as macro definitions, and replayed through all
predictors by both encoder and decoder.
"""

from __future__ import annotations

import argparse
import array
import bisect
import csv
import json
import struct
import sys
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path


TRACE_HEADER = struct.Struct("<4sHHQ")
EXTERNAL_HEADER = struct.Struct("<4sHHQ")
META_MAGIC = b"SCR2M1\x00"
IO_BLOCK = 1 << 20


def uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("negative uvarint")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def read_uvarint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data) or shift >= 64:
            raise ValueError("invalid uvarint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def load_patterns(meta_path: Path) -> list[bytes]:
    blob = meta_path.read_bytes()
    if not blob.startswith(META_MAGIC) or len(blob) < len(META_MAGIC) + 8:
        raise ValueError("invalid SCR2 metadata")
    raw_len, crc = struct.unpack_from("<II", blob, len(META_MAGIC))
    payload = zlib.decompress(blob[len(META_MAGIC) + 8 :])
    if len(payload) != raw_len or zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise ValueError("damaged SCR2 metadata")
    version, _marker, _original_size = struct.unpack_from("<BBQ", payload, 0)
    if version != 1:
        raise ValueError(f"unsupported SCR2 metadata version {version}")
    pos = struct.calcsize("<BBQ") + 32
    count, pos = read_uvarint(payload, pos)
    patterns: list[bytes] = []
    for _ in range(count):
        length, pos = read_uvarint(payload, pos)
        end = pos + length
        if length == 0 or end > len(payload):
            raise ValueError("truncated SCR2 pattern")
        patterns.append(payload[pos:end])
        pos = end
    if pos != len(payload):
        raise ValueError("trailing SCR2 metadata")
    return patterns


def load_costs(trace_path: Path, expected_size: int) -> array.array[float]:
    with trace_path.open("rb") as source:
        header = source.read(TRACE_HEADER.size)
        magic, version, flags, size = TRACE_HEADER.unpack(header)
        if magic != b"F4TC" or version != 1 or flags != 0:
            raise ValueError("invalid FX4 cost trace")
        if size != expected_size:
            raise ValueError(f"trace size {size} != stream size {expected_size}")
        costs = array.array("f")
        costs.fromfile(source, expected_size)
        if sys.byteorder != "little":
            costs.byteswap()
        if source.read(1):
            raise ValueError("trailing cost trace data")
    if len(costs) != expected_size:
        raise ValueError("truncated cost trace")
    return costs


class AhoCorasick:
    def __init__(self, patterns: list[bytes]) -> None:
        self.next: list[dict[int, int]] = [{}]
        self.fail = [0]
        self.out: list[list[int]] = [[]]
        for pattern_id, pattern in enumerate(patterns):
            node = 0
            for byte in pattern:
                child = self.next[node].get(byte)
                if child is None:
                    child = len(self.next)
                    self.next[node][byte] = child
                    self.next.append({})
                    self.fail.append(0)
                    self.out.append([])
                node = child
            self.out[node].append(pattern_id)
        queue: deque[int] = deque(self.next[0].values())
        while queue:
            parent = queue.popleft()
            for byte, child in self.next[parent].items():
                queue.append(child)
                fallback = self.fail[parent]
                while fallback and byte not in self.next[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.next[fallback].get(byte, 0)
                self.out[child].extend(self.out[self.fail[child]])


@dataclass(frozen=True)
class Region:
    region_id: int
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    pattern: int
    gross_bytes: float
    region: int = -1


def load_regions(path: Path, expected_size: int) -> list[Region]:
    regions: list[Region] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            region = Region(
                int(row["pack_id"]),
                int(row["post_r1_start"]),
                int(row["post_r1_end"]),
            )
            if region.end <= region.start:
                raise ValueError("invalid group boundary")
            if regions and (
                region.region_id != regions[-1].region_id + 1
                or region.start != regions[-1].end
            ):
                raise ValueError("groups are not contiguous")
            if not regions and (region.region_id != 0 or region.start != 0):
                raise ValueError("groups must start at byte zero")
            regions.append(region)
    if not regions or regions[-1].end != expected_size:
        raise ValueError("groups do not cover the complete stream")
    return regions


def find_candidates(
    data: bytes,
    costs: array.array[float],
    patterns: list[bytes],
    minimum_gross: float,
    regions: list[Region] | None = None,
) -> list[Candidate]:
    prefix = array.array("d", [0.0])
    running = 0.0
    for cost in costs:
        running += cost
        prefix.append(running)
    matcher = AhoCorasick(patterns)
    candidates: list[Candidate] = []
    region_ends = [region.end for region in regions] if regions else []
    node = 0
    for pos, byte in enumerate(data):
        while node and byte not in matcher.next[node]:
            node = matcher.fail[node]
        node = matcher.next[node].get(byte, 0)
        for pattern_id in matcher.out[node]:
            length = len(patterns[pattern_id])
            start = pos + 1 - length
            end = pos + 1
            if start // IO_BLOCK != (end - 1) // IO_BLOCK:
                continue
            region_id = -1
            if regions:
                region_index = bisect.bisect_right(region_ends, start)
                if (
                    region_index >= len(regions)
                    or end > regions[region_index].end
                ):
                    continue
                region_id = regions[region_index].region_id
            gross = (prefix[end] - prefix[start]) / 8.0
            if gross >= minimum_gross:
                candidates.append(Candidate(
                    start, end, pattern_id, gross, region_id))
    return candidates


def interval_select(
    candidates: list[Candidate], allowed: set[int], event_tax: float
) -> list[Candidate]:
    filtered = [
        item
        for item in candidates
        if item.pattern in allowed and item.gross_bytes > event_tax
    ]
    filtered.sort(key=lambda item: (item.end, item.start, -item.gross_bytes))
    ends = [item.end for item in filtered]
    previous: list[int] = []
    best = [0.0] * (len(filtered) + 1)
    take = [False] * len(filtered)
    for i, item in enumerate(filtered):
        pred = bisect.bisect_right(ends, item.start, 0, i) - 1
        previous.append(pred)
        with_item = best[pred + 1] + item.gross_bytes - event_tax
        if with_item > best[i] + 1e-9:
            best[i + 1] = with_item
            take[i] = True
        else:
            best[i + 1] = best[i]
    selected: list[Candidate] = []
    i = len(filtered) - 1
    while i >= 0:
        if take[i] and best[i + 1] > best[i] + 1e-9:
            selected.append(filtered[i])
            i = previous[i]
        else:
            i -= 1
    selected.reverse()
    return selected


def event_costs(selected: list[Candidate], id_map: dict[int, int]) -> list[int]:
    result: list[int] = []
    previous_end = 0
    for event in selected:
        result.append(
            len(uvarint(event.start - previous_end))
            + len(uvarint(id_map[event.pattern]))
        )
        previous_end = event.end
    return result


def prune_plan(
    candidates: list[Candidate], patterns: list[bytes], event_tax: float
) -> list[Candidate]:
    allowed = set(range(len(patterns)))
    selected: list[Candidate] = []
    for _ in range(12):
        selected = interval_select(candidates, allowed, event_tax)
        if not selected:
            return []
        used = sorted({item.pattern for item in selected})
        id_map = {pattern: index for index, pattern in enumerate(used)}
        costs = event_costs(selected, id_map)
        profitable = [
            item.gross_bytes > cost
            for item, cost in zip(selected, costs)
        ]
        by_pattern: dict[int, float] = {pattern: 0.0 for pattern in used}
        for item, cost, keep in zip(selected, costs, profitable):
            if keep:
                by_pattern[item.pattern] += item.gross_bytes - cost
        new_allowed = {
            pattern
            for pattern, net in by_pattern.items()
            if net > len(patterns[pattern]) + 2
        }
        if new_allowed == allowed and all(profitable):
            return selected
        allowed = new_allowed
        if not allowed:
            return []
    return interval_select(candidates, allowed, event_tax)


def serialize_payload(
    selected: list[Candidate], patterns: list[bytes]
) -> tuple[bytes, dict[int, int]]:
    used = sorted(
        {item.pattern for item in selected},
        key=lambda pattern: (
            -sum(1 for item in selected if item.pattern == pattern),
            pattern,
        ),
    )
    id_map = {pattern: index for index, pattern in enumerate(used)}
    payload = bytearray((1, len(used) & 0xFF, len(used) >> 8))
    for pattern in used:
        value = patterns[pattern]
        payload.extend(struct.pack("<H", len(value)))
        payload.extend(value)
    payload.extend(struct.pack("<I", len(selected)))
    previous_end = 0
    for event in selected:
        payload.extend(uvarint(event.start - previous_end))
        payload.extend(uvarint(id_map[event.pattern]))
        previous_end = event.end
    return bytes(payload), id_map


def plan_net(
    selected: list[Candidate], patterns: list[bytes], fixed_bytes: int
) -> tuple[float, int]:
    if not selected:
        return 0.0, 0
    payload, _ = serialize_payload(selected, patterns)
    return (
        sum(item.gross_bytes for item in selected)
        - len(payload)
        - fixed_bytes,
        len(payload),
    )


def group_marginals(
    selected: list[Candidate], patterns: list[bytes], fixed_bytes: int
) -> dict[int, float]:
    full_net, _ = plan_net(selected, patterns, fixed_bytes)
    result: dict[int, float] = {}
    for region in sorted({item.region for item in selected}):
        without = [item for item in selected if item.region != region]
        without_net, _ = plan_net(without, patterns, fixed_bytes)
        result[region] = full_net - without_net
    return result


def prune_selective_groups(
    candidates: list[Candidate],
    patterns: list[bytes],
    event_tax: float,
    minimum_group_net: float,
    fixed_bytes: int,
) -> tuple[list[Candidate], dict[int, float]]:
    allowed = {item.region for item in candidates}
    while allowed:
        selected = prune_plan(
            [item for item in candidates if item.region in allowed],
            patterns,
            event_tax,
        )
        if not selected:
            return [], {}
        marginals = group_marginals(selected, patterns, fixed_bytes)
        rejected = [
            (margin, region)
            for region, margin in marginals.items()
            if margin + 1.0e-9 < minimum_group_net
        ]
        if not rejected:
            return selected, marginals
        # Shared pattern definitions make group costs non-additive. Removing
        # one group and recomputing keeps every accepted group profitable.
        _margin, region = min(rejected)
        allowed.remove(region)
    return [], {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("scr2_meta", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-gross", type=float, default=2.25)
    parser.add_argument("--event-tax", type=float, default=3.0)
    parser.add_argument("--minimum-net", type=float, default=8.0)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--group", type=int, action="append", default=[])
    parser.add_argument("--minimum-group-net", type=float, default=0.0)
    parser.add_argument("--fixed-plan-bytes", type=int, default=0)
    parser.add_argument("--group-report", type=Path)
    parser.add_argument(
        "--event-report",
        type=Path,
        help="optional CSV containing every retained virtual-replay event",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        help="optional CSV containing every gross-qualified phrase occurrence",
    )
    parser.add_argument(
        "--prefix-hex",
        default="",
        help="bytes prepended by the compressor before prediction, as hex",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logical_data = args.stream.read_bytes()
    try:
        prefix = bytes.fromhex(args.prefix_hex)
    except ValueError as error:
        raise ValueError("--prefix-hex must contain complete hex bytes") from error
    data = prefix + logical_data
    patterns = load_patterns(args.scr2_meta)
    costs = load_costs(args.trace, len(data))
    regions = (
        load_regions(args.groups_csv, len(logical_data))
        if args.groups_csv else None
    )
    if regions and prefix:
        regions = [
            Region(region.region_id, region.start + len(prefix),
                   region.end + len(prefix))
            for region in regions
        ]
    if args.group and not regions:
        raise ValueError("--group requires --groups-csv")
    allowed_groups = set(args.group)
    if regions and allowed_groups:
        known_groups = {region.region_id for region in regions}
        unknown = allowed_groups - known_groups
        if unknown:
            raise ValueError(f"unknown group IDs: {sorted(unknown)}")
    candidates = find_candidates(
        data, costs, patterns, args.minimum_gross, regions
    )
    if allowed_groups:
        candidates = [
            item for item in candidates if item.region in allowed_groups
        ]
    if args.candidate_report:
        args.candidate_report.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_report.open(
                "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "start", "end", "length", "pattern", "gross_bytes",
                "region",
            ])
            for item in candidates:
                writer.writerow([
                    item.start,
                    item.end,
                    item.end - item.start,
                    item.pattern + 1,
                    f"{item.gross_bytes:.9f}",
                    item.region,
                ])
    if regions:
        selected, marginals = prune_selective_groups(
            candidates,
            patterns,
            args.event_tax,
            args.minimum_group_net,
            args.fixed_plan_bytes,
        )
    else:
        selected = prune_plan(candidates, patterns, args.event_tax)
        marginals = {}
    payload, _id_map = serialize_payload(selected, patterns) if selected else (b"", {})
    gross = sum(item.gross_bytes for item in selected)
    estimated_net = gross - len(payload) - args.fixed_plan_bytes
    result = {
        "stream_bytes": len(data),
        "logical_stream_bytes": len(logical_data),
        "predictor_prefix_bytes": len(prefix),
        "patterns_considered": len(patterns),
        "candidate_occurrences": len(candidates),
        "selected_patterns": len({item.pattern for item in selected}),
        "selected_events": len(selected),
        "selected_logical_bytes": sum(item.end - item.start for item in selected),
        "baseline_cost_removed_bytes": gross,
        "archive_plan_bytes": len(payload),
        "fixed_plan_bytes": args.fixed_plan_bytes,
        "estimated_net_saving_bytes": estimated_net,
        "selected_groups": len({item.region for item in selected}) if regions else 0,
        "minimum_group_net_bytes": args.minimum_group_net if regions else None,
    }
    if args.candidate_report:
        result["candidate_report"] = str(args.candidate_report)
    if regions:
        group_report = args.group_report or args.output_plan.with_suffix(
            args.output_plan.suffix + ".groups.csv"
        )
        candidate_counts: dict[int, int] = {}
        selected_counts: dict[int, int] = {}
        selected_gross: dict[int, float] = {}
        for item in candidates:
            candidate_counts[item.region] = candidate_counts.get(item.region, 0) + 1
        for item in selected:
            selected_counts[item.region] = selected_counts.get(item.region, 0) + 1
            selected_gross[item.region] = (
                selected_gross.get(item.region, 0.0) + item.gross_bytes
            )
        group_report.parent.mkdir(parents=True, exist_ok=True)
        with group_report.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "pack_id", "post_r1_start", "post_r1_end",
                "candidate_occurrences", "selected_events",
                "selected_gross_bytes", "marginal_net_bytes", "status",
            ])
            for region in regions:
                if allowed_groups and region.region_id not in allowed_groups:
                    continue
                selected_count = selected_counts.get(region.region_id, 0)
                writer.writerow([
                    region.region_id,
                    region.start,
                    region.end,
                    candidate_counts.get(region.region_id, 0),
                    selected_count,
                    f"{selected_gross.get(region.region_id, 0.0):.6f}",
                    f"{marginals.get(region.region_id, 0.0):.6f}",
                    "selected" if selected_count else "baseline",
                ])
        result["group_report"] = str(group_report)
    if args.event_report:
        args.event_report.parent.mkdir(parents=True, exist_ok=True)
        with args.event_report.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "start", "end", "length", "pattern", "gross_bytes",
                "region",
            ])
            for item in selected:
                writer.writerow([
                    item.start,
                    item.end,
                    item.end - item.start,
                    item.pattern + 1,
                    f"{item.gross_bytes:.9f}",
                    item.region,
                ])
        result["event_report"] = str(args.event_report)
    report_path = args.report or args.output_plan.with_suffix(
        args.output_plan.suffix + ".json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not selected or estimated_net < args.minimum_net:
        args.output_plan.unlink(missing_ok=True)
        print("F4VR rejected before coding: estimated net gain is too small")
        return 2
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_bytes(
        EXTERNAL_HEADER.pack(b"F4VR", 1, 0, len(data)) + payload
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
