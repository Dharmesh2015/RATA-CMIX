#!/usr/bin/env python3
"""Score FX4 donor offsets and build an acyclic offset-only replay plan.

This is an offline screening tool. Deflate's preset-dictionary gain is used
only to reject donor/recipient pairs that do not share useful byte sequences.
Final archive decisions still require exact FX4 coding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import struct
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


_stream_file = None
_stream = None
_chunk_size = 0
_seed_size = 0
_level = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path)
    parser.add_argument("candidate_edges", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument("--seed-size", type=int, default=4096)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--metadata-cost", type=int, default=6)
    return parser.parse_args()


def init_worker(
    stream_path: str, chunk_size: int, seed_size: int, level: int
) -> None:
    global _stream_file, _stream, _chunk_size, _seed_size, _level
    _stream_file = open(stream_path, "rb")
    _stream = mmap.mmap(_stream_file.fileno(), 0, access=mmap.ACCESS_READ)
    _chunk_size = chunk_size
    _seed_size = seed_size
    _level = level


def deflate_size(data: bytes, dictionary: bytes | None = None) -> int:
    options = (zlib.DEFLATED, -15, 8, zlib.Z_DEFAULT_STRATEGY)
    if dictionary is None:
        coder = zlib.compressobj(_level, *options)
    else:
        coder = zlib.compressobj(_level, *options, zdict=dictionary)
    return len(coder.compress(data)) + len(coder.flush())


def score_recipient(item: tuple[int, list[dict[str, str]]]) -> list[dict]:
    recipient, edges = item
    start = recipient * _chunk_size
    data = _stream[start : start + _chunk_size]
    baseline = deflate_size(data)
    results = []
    for edge in edges:
        seed_offset = int(edge["donor_seed_offset"])
        seed = _stream[seed_offset : seed_offset + _seed_size]
        assisted = deflate_size(data, seed)
        row = dict(edge)
        row["deflate_baseline_bytes"] = baseline
        row["deflate_assisted_bytes"] = assisted
        row["deflate_gain_bytes"] = baseline - assisted
        results.append(row)
    return results


def creates_cycle(parent: dict[int, int], donor: int, recipient: int) -> bool:
    node = donor
    seen = {recipient}
    while node in parent:
        if node in seen:
            return True
        seen.add(node)
        node = parent[node]
    return node in seen


def topological_order(region_count: int, parent: dict[int, int]) -> list[int]:
    children: dict[int, list[int]] = defaultdict(list)
    indegree = [0] * region_count
    for recipient, donor in parent.items():
        children[donor].append(recipient)
        indegree[recipient] += 1
    ready = [region for region, degree in enumerate(indegree) if degree == 0]
    ready.sort(reverse=True)
    result = []
    while ready:
        node = ready.pop()
        result.append(node)
        for child in sorted(children[node], reverse=True):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(reverse=True)
    if len(result) != region_count:
        raise RuntimeError("selected graph is cyclic")
    return result


def write_binary_plan(
    path: Path,
    stream_sha256: bytes,
    chunk_size: int,
    seed_size: int,
    region_count: int,
    selected: list[dict],
) -> None:
    if region_count > 65535:
        raise ValueError("region count exceeds plan format")
    with path.open("wb") as output:
        output.write(b"F4DG")
        output.write(struct.pack("<HHIIHH", 1, 0, chunk_size, seed_size,
                                 region_count, len(selected)))
        output.write(stream_sha256)
        for edge in sorted(selected, key=lambda row: int(row["recipient_region"])):
            recipient = int(edge["recipient_region"])
            seed_offset = int(edge["donor_seed_offset"])
            if seed_offset > 0xFFFFFFFF:
                raise ValueError("seed offset exceeds plan format")
            output.write(struct.pack("<HI", recipient, seed_offset))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_size = args.stream.stat().st_size
    region_count = stream_size // args.chunk_size

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with args.candidate_edges.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            grouped[int(row["recipient_region"])].append(row)

    tasks = sorted(grouped.items())
    scored = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            str(args.stream),
            args.chunk_size,
            args.seed_size,
            args.level,
        ),
    ) as executor:
        for index, rows in enumerate(executor.map(score_recipient, tasks), 1):
            scored.extend(rows)
            if index % 32 == 0 or index == len(tasks):
                print(f"scored_recipients={index}/{len(tasks)}", flush=True)

    scored.sort(
        key=lambda row: (
            int(row["recipient_region"]),
            -int(row["deflate_gain_bytes"]),
            int(row["donor_region"]),
        )
    )
    fieldnames = list(scored[0]) if scored else []
    with (args.output_dir / "scored_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    useful = [
        row
        for row in scored
        if int(row["deflate_gain_bytes"]) > args.metadata_cost
    ]
    useful.sort(
        key=lambda row: (
            -int(row["deflate_gain_bytes"]),
            int(row["recipient_region"]),
            int(row["donor_region"]),
        )
    )
    with (args.output_dir / "useful_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(useful)

    parent: dict[int, int] = {}
    selected = []
    for row in useful:
        donor = int(row["donor_region"])
        recipient = int(row["recipient_region"])
        if recipient in parent or creates_cycle(parent, donor, recipient):
            continue
        parent[recipient] = donor
        selected.append(row)

    order = topological_order(region_count, parent)
    selected.sort(key=lambda row: int(row["recipient_region"]))
    with (args.output_dir / "selected_acyclic_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    with (args.output_dir / "topological_order.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["compression_order", "original_region"])
        writer.writerows(enumerate(order))

    digest = hashlib.sha256()
    with args.stream.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    stream_sha256 = digest.digest()
    write_binary_plan(
        args.output_dir / "fx4_donor_plan.bin",
        stream_sha256,
        args.chunk_size,
        args.seed_size,
        region_count,
        selected,
    )
    plan = (args.output_dir / "fx4_donor_plan.bin").read_bytes()
    compressed_plan = zlib.compress(plan, 9)
    (args.output_dir / "fx4_donor_plan.bin.z").write_bytes(compressed_plan)

    gross_gain = sum(int(row["deflate_gain_bytes"]) for row in selected)
    net_gain = gross_gain - len(compressed_plan)
    summary = {
        "stream": str(args.stream),
        "stream_bytes": stream_size,
        "stream_sha256": stream_sha256.hex(),
        "chunk_size": args.chunk_size,
        "complete_regions": region_count,
        "tail_bytes": stream_size - region_count * args.chunk_size,
        "input_candidate_edges": len(scored),
        "useful_proxy_edges": len(useful),
        "selected_acyclic_edges": len(selected),
        "deflate_gross_gain_bytes": gross_gain,
        "raw_plan_bytes": len(plan),
        "compressed_plan_bytes": len(compressed_plan),
        "deflate_net_gain_after_compressed_plan": net_gain,
        "validation": "deflate_proxy_only_exact_fx4_required",
    }
    (args.output_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
