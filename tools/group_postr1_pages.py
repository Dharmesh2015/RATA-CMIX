#!/usr/bin/env python3
"""Group post-R1 pages into contiguous, near-target donor recipients.

The CSV is used by donor discovery. The compact F4GB file stores only the
resulting contiguous span lengths; starts are recovered by prefix sum.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"F4GB"
VERSION = 1
MIXED_CLASS = 10


@dataclass(frozen=True)
class Page:
    page_id: int
    start: int
    end: int
    post_r1_length: int
    decoded_length: int
    stream_class: int
    class_name: str
    feature_mask: int


@dataclass(frozen=True)
class Group:
    group_id: int
    first_page: int
    page_count: int
    start: int
    end: int
    decoded_length: int
    stream_class: int
    class_name: str
    feature_mask: int

    @property
    def length(self) -> int:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pages_csv", type=Path)
    parser.add_argument("groups_csv", type=Path)
    parser.add_argument("boundaries_bin", type=Path)
    parser.add_argument("--stream-size", type=int, required=True)
    parser.add_argument("--target-bytes", type=int, default=1 << 20)
    return parser.parse_args()


def read_pages(path: Path, stream_size: int) -> list[Page]:
    pages: list[Page] = []
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "pack_id", "first_page", "page_count", "post_r1_start",
            "post_r1_end", "post_r1_length", "decoded_length",
            "stream_class", "class_name", "feature_mask",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"unexpected page CSV header: {reader.fieldnames}")
        previous_start = -1
        previous_page = -1
        for row in reader:
            if int(row["page_count"]) != 1:
                raise ValueError("input must contain one physical page per row")
            page = Page(
                page_id=int(row["first_page"]),
                start=int(row["post_r1_start"]),
                end=int(row["post_r1_end"]),
                post_r1_length=int(row["post_r1_length"]),
                decoded_length=int(row["decoded_length"]),
                stream_class=int(row["stream_class"]),
                class_name=row["class_name"],
                feature_mask=int(row["feature_mask"]),
            )
            if page.page_id != previous_page + 1:
                raise ValueError("page IDs are not contiguous")
            if page.start <= previous_start or page.end <= page.start:
                raise ValueError("page offsets are not strictly increasing")
            if page.end - page.start != page.post_r1_length:
                raise ValueError("page length does not match its offsets")
            if page.end > stream_size:
                raise ValueError("page exceeds the declared post-R1 stream")
            pages.append(page)
            previous_start = page.start
            previous_page = page.page_id
    if not pages:
        raise ValueError("page CSV is empty")
    return pages


def choose_boundary(
    pages: list[Page], first: int, group_start: int, target: int
) -> int:
    desired = group_start + target
    low = first + 1
    high = len(pages)
    while low < high:
        middle = (low + high) // 2
        if pages[middle].start < desired:
            low = middle + 1
        else:
            high = middle
    candidates = [index for index in (low - 1, low)
                  if first < index < len(pages)]
    if not candidates:
        return len(pages)
    # A tie stays at or below the target to bound ordinary chunk sizes.
    return min(candidates, key=lambda index: (
        abs(pages[index].start - desired),
        pages[index].start > desired,
        pages[index].start,
    ))


def group_pages(pages: list[Page], stream_size: int, target: int) -> list[Group]:
    if target <= 0:
        raise ValueError("target size must be positive")
    groups: list[Group] = []
    first = 0
    group_start = 0
    while first < len(pages):
        next_first = choose_boundary(pages, first, group_start, target)
        group_end = pages[next_first].start if next_first < len(pages) else stream_size
        if group_end <= group_start:
            raise ValueError("grouping failed to make forward progress")
        members = pages[first:next_first]
        weights: dict[int, int] = {}
        names: dict[int, str] = {}
        decoded_length = 0
        feature_mask = 0
        for page in members:
            weights[page.stream_class] = (
                weights.get(page.stream_class, 0) + page.post_r1_length)
            names[page.stream_class] = page.class_name
            decoded_length += page.decoded_length
            feature_mask |= page.feature_mask
        dominant_class, dominant_bytes = max(
            weights.items(), key=lambda item: (item[1], -item[0]))
        page_bytes = sum(weights.values())
        if dominant_bytes * 2 < page_bytes:
            stream_class = MIXED_CLASS
            class_name = "mixed"
        else:
            stream_class = dominant_class
            class_name = names[dominant_class]
        groups.append(Group(
            group_id=len(groups),
            first_page=members[0].page_id,
            page_count=len(members),
            start=group_start,
            end=group_end,
            decoded_length=decoded_length,
            stream_class=stream_class,
            class_name=class_name,
            feature_mask=feature_mask,
        ))
        first = next_first
        group_start = group_end
    if groups[-1].end != stream_size:
        raise ValueError("groups do not cover the complete stream")
    return groups


def write_groups(path: Path, groups: list[Group]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pack_id", "first_page", "page_count", "post_r1_start",
        "post_r1_end", "post_r1_length", "decoded_length",
        "stream_class", "class_name", "feature_mask",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(fields)
        for group in groups:
            writer.writerow([
                group.group_id, group.first_page, group.page_count,
                group.start, group.end, group.length, group.decoded_length,
                group.stream_class, group.class_name, group.feature_mask,
            ])


def write_varint(output, value: int) -> None:
    if value < 0:
        raise ValueError("varints cannot be negative")
    while value >= 0x80:
        output.write(bytes(((value & 0x7F) | 0x80,)))
        value >>= 7
    output.write(bytes((value,)))


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift <= 63:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid grouped-boundary varint")


def write_boundaries(
    path: Path, groups: list[Group], stream_size: int, target: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("B", VERSION))
        write_varint(output, target)
        write_varint(output, stream_size)
        write_varint(output, len(groups))
        for group in groups:
            write_varint(output, group.length)


def verify_boundaries(path: Path, groups: list[Group]) -> None:
    data = path.read_bytes()
    if data[:4] != MAGIC or len(data) < 5 or data[4] != VERSION:
        raise ValueError("invalid grouped-boundary header")
    offset = 5
    _, offset = read_varint(data, offset)  # target
    stream_size, offset = read_varint(data, offset)
    count, offset = read_varint(data, offset)
    if count != len(groups):
        raise ValueError("group count changed during boundary roundtrip")
    position = 0
    for expected in groups:
        length, offset = read_varint(data, offset)
        if position != expected.start or length != expected.length:
            raise ValueError("group boundary changed during roundtrip")
        position += length
    if offset != len(data) or position != stream_size:
        raise ValueError("grouped-boundary payload is inconsistent")


def main() -> None:
    args = parse_args()
    pages = read_pages(args.pages_csv, args.stream_size)
    groups = group_pages(pages, args.stream_size, args.target_bytes)
    write_groups(args.groups_csv, groups)
    write_boundaries(
        args.boundaries_bin, groups, args.stream_size, args.target_bytes)
    verify_boundaries(args.boundaries_bin, groups)

    lengths = [group.length for group in groups]
    sorted_lengths = sorted(lengths)
    percentile95 = sorted_lengths[min(
        len(sorted_lengths) - 1, int(0.95 * len(sorted_lengths)))]
    print(f"pages={len(pages)}")
    print(f"groups={len(groups)}")
    print(f"target_bytes={args.target_bytes}")
    print(f"minimum_bytes={min(lengths)}")
    print(f"median_bytes={int(statistics.median(lengths))}")
    print(f"mean_bytes={sum(lengths) / len(lengths):.2f}")
    print(f"p95_bytes={percentile95}")
    print(f"maximum_bytes={max(lengths)}")
    print(f"boundary_side_data_bytes={args.boundaries_bin.stat().st_size}")
    print(f"groups_csv={args.groups_csv}")
    print(f"boundaries_bin={args.boundaries_bin}")


if __name__ == "__main__":
    main()
