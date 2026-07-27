#!/usr/bin/env python3
"""Validate an offset-only FX4 donor graph against its prepared stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import mmap
import statistics
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path)
    parser.add_argument("selected_edges", type=Path)
    parser.add_argument("topological_order", type=Path)
    parser.add_argument("binary_plan", type=Path)
    parser.add_argument("--scored-edges", type=Path)
    return parser.parse_args()


def fnv1a(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = value * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return value


def summarize(label: str, gains: list[int]) -> None:
    ordered = sorted(gains)
    percentile = lambda fraction: ordered[
        min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    ]
    print(
        f"{label}: count={len(gains)} min={ordered[0]} "
        f"p25={percentile(0.25)} p50={percentile(0.50)} "
        f"p75={percentile(0.75)} p95={percentile(0.95)} "
        f"max={ordered[-1]} mean={statistics.mean(gains):.2f}"
    )
    print(
        f"{label}_thresholds: "
        + " ".join(
            f">{threshold}={sum(gain > threshold for gain in gains)}"
            for threshold in (0, 6, 32, 64, 128, 256, 512)
        )
    )


def main() -> int:
    args = parse_args()
    with args.selected_edges.open(newline="", encoding="utf-8") as source:
        selected = list(csv.DictReader(source))
    with args.topological_order.open(newline="", encoding="utf-8") as source:
        order_rows = list(csv.DictReader(source))
    order = [int(row["original_region"]) for row in order_rows]
    position = {region: index for index, region in enumerate(order)}

    stream_digest = hashlib.sha256()
    with args.stream.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            stream_digest.update(block)
    stream_sha256 = stream_digest.digest()

    with args.stream.open("rb") as source:
        data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        for edge in selected:
            donor = int(edge["donor_region"])
            recipient = int(edge["recipient_region"])
            donor_offset = int(edge["donor_region_offset"])
            seed_offset = int(edge["donor_seed_offset"])
            seed_size = int(edge["seed_size"])
            if not donor_offset <= seed_offset:
                raise RuntimeError(f"seed before donor region: {edge}")
            if seed_offset + seed_size > donor_offset + (1 << 20):
                raise RuntimeError(f"seed after donor region: {edge}")
            if position[donor] >= position[recipient]:
                raise RuntimeError(f"dependency order violation: {edge}")
            actual_hash = fnv1a(data[seed_offset : seed_offset + seed_size])
            expected_hash = int(edge["donor_seed_hash"], 16)
            if actual_hash != expected_hash:
                raise RuntimeError(f"seed hash mismatch: {edge}")

    plan = args.binary_plan.read_bytes()
    if len(plan) < 52 or plan[:4] != b"F4DG":
        raise RuntimeError("invalid binary plan header")
    version, flags, chunk_size, seed_size, region_count, edge_count = (
        struct.unpack_from("<HHIIHH", plan, 4)
    )
    plan_sha256 = plan[20:52]
    if version != 1 or flags != 0:
        raise RuntimeError("unsupported binary plan")
    if plan_sha256 != stream_sha256:
        raise RuntimeError("binary plan stream hash mismatch")
    if edge_count != len(selected) or region_count != len(order):
        raise RuntimeError("binary plan counts mismatch")
    expected_size = 52 + edge_count * 6
    if len(plan) != expected_size:
        raise RuntimeError("binary plan size mismatch")
    encoded = {
        struct.unpack_from("<HI", plan, 52 + index * 6)
        for index in range(edge_count)
    }
    expected = {
        (int(edge["recipient_region"]), int(edge["donor_seed_offset"]))
        for edge in selected
    }
    if encoded != expected:
        raise RuntimeError("binary plan assignments mismatch")

    summarize(
        "selected",
        [int(edge["deflate_gain_bytes"]) for edge in selected],
    )
    if args.scored_edges:
        with args.scored_edges.open(newline="", encoding="utf-8") as source:
            scored = list(csv.DictReader(source))
        summarize(
            "all_candidates",
            [int(edge["deflate_gain_bytes"]) for edge in scored],
        )
    print(
        f"graph_valid=yes stream_sha256={stream_sha256.hex()} "
        f"regions={region_count} edges={edge_count} "
        f"chunk_size={chunk_size} seed_size={seed_size} "
        f"plan_bytes={len(plan)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
