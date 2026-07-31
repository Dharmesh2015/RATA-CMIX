#!/usr/bin/env python3
"""Map post-WRT donor edges into the exact post-R1 predictor byte domain.

The F4RM1 map is emitted by r1_reorder_transform.cpp. A source interval may
become several post-R1 fragments. This tool keeps their source-byte order,
assigns an explicit replay order, and rejects edges whose donor bytes are not
fully decoded before the mapped recipient starts.
"""

from __future__ import annotations

import argparse
import csv
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"F4RM1\r\n\0"
U64 = struct.Struct("<Q")
SEGMENT = struct.Struct("<QQQ")


@dataclass(frozen=True)
class Segment:
    source: int
    destination: int
    length: int

    @property
    def source_end(self) -> int:
        return self.source + self.length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("r1_map", type=Path)
    parser.add_argument("post_wrt_edges", type=Path)
    parser.add_argument("post_r1_edges", type=Path)
    parser.add_argument(
        "--recipient-size",
        type=int,
        default=1 << 20,
        help="Source-domain recipient size when CSV has recipient_region.",
    )
    parser.add_argument(
        "--require-contiguous-recipient",
        action="store_true",
        help="Reject recipients split or permuted by R1.",
    )
    return parser.parse_args()


def read_u64(source) -> int:
    raw = source.read(U64.size)
    if len(raw) != U64.size:
        raise SystemExit("truncated F4RM1 header")
    return U64.unpack(raw)[0]


def load_map(path: Path) -> tuple[int, int, int, list[Segment]]:
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{path}: invalid F4RM1 magic")
        post_wrt_size = read_u64(source)
        post_r1_payload_size = read_u64(source)
        side_size = read_u64(source)
        count = read_u64(source)
        segments: list[Segment] = []
        for _ in range(count):
            raw = source.read(SEGMENT.size)
            if len(raw) != SEGMENT.size:
                raise SystemExit(f"{path}: truncated segment table")
            segments.append(Segment(*SEGMENT.unpack(raw)))
        if source.read(1):
            raise SystemExit(f"{path}: unexpected trailing map bytes")

    segments.sort(key=lambda segment: segment.source)
    expected = 0
    for segment in segments:
        if segment.length == 0 or segment.source != expected:
            raise SystemExit(
                f"{path}: source coverage gap/overlap at {expected}"
            )
        if segment.destination + segment.length > post_r1_payload_size:
            raise SystemExit(f"{path}: destination segment is out of bounds")
        expected = segment.source_end
    if expected != post_wrt_size:
        raise SystemExit(
            f"{path}: mapped {expected} of {post_wrt_size} source bytes"
        )
    return post_wrt_size, post_r1_payload_size, side_size, segments


def map_interval(
    segments: list[Segment], start: int, length: int
) -> list[tuple[int, int, int]]:
    """Return (source_offset, destination_offset, length) fragments."""
    end = start + length
    fragments: list[tuple[int, int, int]] = []
    for segment in segments:
        if segment.source_end <= start:
            continue
        if segment.source >= end:
            break
        overlap_start = max(start, segment.source)
        overlap_end = min(end, segment.source_end)
        fragments.append(
            (
                overlap_start,
                segment.destination + overlap_start - segment.source,
                overlap_end - overlap_start,
            )
        )
    if sum(fragment[2] for fragment in fragments) != length:
        raise ValueError(f"source interval {start}+{length} is not fully mapped")
    return fragments


def recipient_start(
    row: dict[str, str], default_size: int
) -> tuple[int, int]:
    if row.get("recipient_offset"):
        start = int(row["recipient_offset"])
        length = int(row.get("recipient_length") or default_size)
    else:
        start = int(row["recipient_region"]) * default_size
        length = int(row.get("recipient_length") or default_size)
    return start, length


def contiguous_destination(
    fragments: list[tuple[int, int, int]]
) -> tuple[int, int] | None:
    by_source = sorted(fragments)
    start = by_source[0][1]
    expected = start
    for _source, destination, length in by_source:
        if destination != expected:
            return None
        expected += length
    return start, expected - start


def main() -> int:
    args = parse_args()
    post_wrt_size, post_r1_size, side_size, segments = load_map(args.r1_map)
    output_rows: list[dict[str, int | str]] = []
    rejected: list[str] = []

    with args.post_wrt_edges.open(newline="", encoding="utf-8") as source:
        for edge_index, row in enumerate(csv.DictReader(source)):
            donor_start = int(row["donor_offset"])
            donor_length = int(row["donor_length"])
            recipient_source, recipient_length = recipient_start(
                row, args.recipient_size
            )
            gain = int(row.get("gain_bytes") or 0)
            try:
                donor_fragments = map_interval(
                    segments, donor_start, donor_length
                )
                recipient_fragments = map_interval(
                    segments, recipient_source, recipient_length
                )
            except ValueError as error:
                rejected.append(f"edge {edge_index}: {error}")
                continue

            mapped_recipient = contiguous_destination(recipient_fragments)
            if mapped_recipient is None:
                rejected.append(
                    f"edge {edge_index}: recipient is split/permuted by R1"
                )
                continue
            recipient_destination, _ = mapped_recipient
            if (
                recipient_length != args.recipient_size
                or recipient_destination % args.recipient_size != 0
            ):
                rejected.append(
                    f"edge {edge_index}: mapped recipient is not an aligned "
                    f"{args.recipient_size}-byte runtime region"
                )
                continue
            if args.require_contiguous_recipient and len(recipient_fragments) != 1:
                rejected.append(
                    f"edge {edge_index}: recipient spans R1 map segments"
                )
                continue

            # Replaying must reproduce the original post-WRT donor byte order.
            donor_fragments.sort(key=lambda fragment: fragment[0])
            if any(
                destination & 255
                or length < 256
                or length > 65536
                or length & (length - 1)
                for _source, destination, length in donor_fragments
            ):
                rejected.append(
                    f"edge {edge_index}: mapped donor fragments cannot be "
                    "represented by the compact runtime plan"
                )
                continue
            if any(
                destination + length > recipient_destination
                for _source, destination, length in donor_fragments
            ):
                rejected.append(
                    f"edge {edge_index}: mapped donor is noncausal after R1"
                )
                continue

            base_order = int(row.get("replay_order") or 0)
            for fragment_index, (
                source_offset,
                destination,
                length,
            ) in enumerate(donor_fragments):
                output_rows.append(
                    {
                        "recipient_region":
                            recipient_destination // args.recipient_size,
                        "recipient_offset": recipient_destination,
                        "recipient_length": recipient_length,
                        "donor_offset": destination,
                        "donor_length": length,
                        "replay_order": base_order + fragment_index,
                        "gain_bytes": gain if fragment_index == 0 else 0,
                        "post_wrt_donor_offset": source_offset,
                        "source_edge": edge_index,
                    }
                )

    output_rows.sort(
        key=lambda row: (
            int(row["recipient_region"]),
            int(row["replay_order"]),
            int(row["source_edge"]),
            int(row["post_wrt_donor_offset"]),
        )
    )
    recipient_order: dict[int, int] = {}
    for row in output_rows:
        recipient = int(row["recipient_region"])
        order = recipient_order.get(recipient, 0)
        if order > 255:
            raise SystemExit(
                f"recipient {recipient} has more than 256 donor fragments"
            )
        row["replay_order"] = order
        recipient_order[recipient] = order + 1

    args.post_r1_edges.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recipient_region",
        "recipient_offset",
        "recipient_length",
        "donor_offset",
        "donor_length",
        "replay_order",
        "gain_bytes",
        "post_wrt_donor_offset",
        "source_edge",
    ]
    with args.post_r1_edges.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"post-WRT bytes: {post_wrt_size}")
    print(f"post-R1 payload bytes: {post_r1_size}")
    print(f"post-R1 side bytes: {side_size}")
    print(f"mapped fragments: {len(output_rows)}")
    print(f"rejected edges: {len(rejected)}")
    for message in rejected[:20]:
        print(message)
    return 0 if output_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
