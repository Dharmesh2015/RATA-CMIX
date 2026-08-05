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
class Candidate:
    start: int
    end: int
    pattern: int
    gross_bytes: float


def find_candidates(
    data: bytes,
    costs: array.array[float],
    patterns: list[bytes],
    minimum_gross: float,
) -> list[Candidate]:
    prefix = array.array("d", [0.0])
    running = 0.0
    for cost in costs:
        running += cost
        prefix.append(running)
    matcher = AhoCorasick(patterns)
    candidates: list[Candidate] = []
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
            gross = (prefix[end] - prefix[start]) / 8.0
            if gross >= minimum_gross:
                candidates.append(Candidate(start, end, pattern_id, gross))
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = args.stream.read_bytes()
    patterns = load_patterns(args.scr2_meta)
    costs = load_costs(args.trace, len(data))
    candidates = find_candidates(
        data, costs, patterns, args.minimum_gross
    )
    selected = prune_plan(candidates, patterns, args.event_tax)
    payload, _id_map = serialize_payload(selected, patterns) if selected else (b"", {})
    gross = sum(item.gross_bytes for item in selected)
    estimated_net = gross - len(payload)
    result = {
        "stream_bytes": len(data),
        "patterns_considered": len(patterns),
        "candidate_occurrences": len(candidates),
        "selected_patterns": len({item.pattern for item in selected}),
        "selected_events": len(selected),
        "selected_logical_bytes": sum(item.end - item.start for item in selected),
        "baseline_cost_removed_bytes": gross,
        "archive_plan_bytes": len(payload),
        "estimated_net_saving_bytes": estimated_net,
    }
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
