#!/usr/bin/env python3
"""Compile selective post-R1 transform and predictor plans.

The tool deliberately separates discovery from production. It never marks a
transform as a winner merely because raw bytes shrink. A transform CSV contains
only decisions already accepted by exact FX4 cost/coding tests. Unlisted spans
are emitted as RAW. The resulting F4TX plan partitions the complete post-R1
stream, while the optional F4CP v4 plan carries ordered donors and selective
predictor masks.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path


MIB = 1 << 20
F4TX_HEADER = struct.Struct("<4sHHQI")
F4TX_BLOCK = struct.Struct("<QIIHBBH")
F4TX_OPERATION = struct.Struct("<BIII")
F4CP_HEADER = struct.Struct("<4sHHIIQH32s")
F4CP_DONOR = struct.Struct("<HIHB")
F4CP_SPAN = struct.Struct("<QII BB")
F4CP_SPAN_V6 = struct.Struct("<QII BBH")
TRACE_HEADER = struct.Struct("<4sHHQ")


STAGES = {
    "pmd1": 1 << 0,
    "typed": 1 << 1,
    "openzl": 1 << 1,
    "xml_body": 1 << 1,
    "scr2": 1 << 2,
    "shorthand": 1 << 2,
    "cx": 1 << 3,
    "scz": 1 << 4,
    "wct": 1 << 5,
    "scrr": 1 << 6,
    "rlz": 1 << 7,
    "approx_rlz": 1 << 7,
    "xor": 1 << 8,
    "wikinative2": 1 << 9,
    "wiki": 1 << 9,
    "phrase": 1 << 10,
    "macros": 1 << 10,
}

EXPERTS = {
    "structural": 1 << 0,
    "ppmd_escape_order": 1 << 1,
    "sparse_virtual_ppm": 1 << 2,
    "word_xml_ppm": 1 << 3,
    "residual_lstm": 1 << 4,
    "micro_diffusion": 1 << 5,
    "rare_residual": 1 << 6,
    "cts_skipcts": 1 << 7,
    "dmc": 1 << 8,
    "token_match": 1 << 9,
    # Both names select the one causal top-3 continuation specialist.
    "episodic_cache": 1 << 9,
    "context_mixer": 1 << 12,
    "oracle": 1 << 14,
    "donor_profile": 1 << 15,
    "mini_cmix": 1 << 16,
    "legacy_donor_replay": 1 << 17,
    "url_structure": 1 << 18,
    "shadow_only": 1 << 19,
}

STREAM_CLASSES = {
    "prose": 0,
    "xml": 1,
    "number": 2,
    "date": 3,
    "table": 4,
    "reference": 5,
    "url": 6,
    "identifier": 7,
    "template": 8,
    "list": 9,
    "mixed": 10,
    "auto": 10,
}


@dataclass
class Operation:
    kind: int
    a: int
    b: int = 0
    c: int = 0


@dataclass
class TransformChoice:
    offset: int
    length: int
    mask: int
    stream_class: int = 10
    physical_order: int | None = None
    references: list[int] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)


@dataclass
class ExpertChoice:
    offset: int
    length: int
    mask: int
    stream_class: int
    profile: int
    mini_model_mask: int = 0


@dataclass
class DonorChoice:
    recipient: int
    donor_offset: int
    length: int
    order: int


def parse_mask(text: str, names: dict[str, int]) -> int:
    text = text.strip()
    if not text or text.lower() in {"raw", "none", "off"}:
        return 0
    if text.lower().startswith("0x") or text.isdigit():
        return int(text, 0)
    mask = 0
    for item in text.replace("|", "+").replace(",", "+").split("+"):
        key = item.strip().lower()
        if key not in names:
            raise ValueError(f"unknown selection name: {key}")
        mask |= names[key]
    return mask


def parse_offsets(text: str) -> list[int]:
    return [int(item, 0) for item in text.replace("|", ";").split(";") if item.strip()]


def load_transform_csv(path: Path | None) -> list[TransformChoice]:
    if path is None:
        return []
    result: list[TransformChoice] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            operations: list[Operation] = []
            if row.get("operations", "").strip():
                raw = json.loads(row["operations"])
                for operation in raw:
                    name = str(operation.get("operation", operation.get("type", ""))).lower()
                    values = operation.get("parameters", [])
                    if name in {"rotate", "1"}:
                        operations.append(Operation(1, int(values[0])))
                    elif name in {"segment-rotate", "segment_rotate", "2"}:
                        operations.append(Operation(2, *(int(value) for value in values[:3])))
                    else:
                        raise ValueError(f"unknown PMD1 operation {name}")
            order_text = row.get("physical_order", "").strip()
            result.append(
                TransformChoice(
                    offset=int(row["offset"], 0),
                    length=int(row["length"], 0),
                    mask=parse_mask(row.get("stages", row.get("stage_mask", "")), STAGES),
                    stream_class=STREAM_CLASSES.get(row.get("stream_class", "mixed").lower(), 10),
                    physical_order=int(order_text, 0) if order_text else None,
                    references=parse_offsets(row.get("references", "")),
                    operations=operations,
                )
            )
    return result


def load_pmd1(path: Path, offset: int) -> TransformChoice:
    document = json.loads(path.read_text(encoding="utf-8"))
    operations: list[Operation] = []
    for item in document["hops"]:
        values = [int(value) for value in item["parameters"]]
        if item["operation"] == "rotate":
            operations.append(Operation(1, values[0]))
        elif item["operation"] == "segment-rotate":
            operations.append(Operation(2, values[0], values[1], values[2]))
        else:
            raise ValueError(f"unknown PMD1 operation {item['operation']}")
    return TransformChoice(
        offset=offset,
        length=int(document["source_size"]),
        mask=STAGES["pmd1"],
        operations=operations,
    )


def parse_mini_models(text: str, expert_mask: int) -> int:
    text = text.strip().lower()
    if not text:
        return 0x07FF if expert_mask & EXPERTS["mini_cmix"] else 0
    if text in {"all", "1-11"}:
        return 0x07FF
    if text.startswith("0x") or text.isdigit() and int(text) > 11:
        value = int(text, 0)
    else:
        value = 0
        for item in text.replace("|", "+").replace(",", "+").split("+"):
            model = int(item)
            if not 1 <= model <= 11:
                raise ValueError("mini_models entries must be 1..11")
            value |= 1 << (model - 1)
    if not 0 < value <= 0x07FF:
        raise ValueError("mini_models mask must select one or more of 11 models")
    if not expert_mask & EXPERTS["mini_cmix"]:
        raise ValueError("mini_models requires mini_cmix in experts")
    return value

def load_expert_csv(path: Path | None) -> list[ExpertChoice]:
    if path is None:
        return []
    result: list[ExpertChoice] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            gain = float(row.get("residual_gain", "0.25"))
            gain_code = {0.25: 0, 0.5: 1, 0.75: 2, 1.0: 3}.get(gain)
            if gain_code is None:
                raise ValueError("residual_gain must be 0.25, 0.5, 0.75 or 1.0")
            semantic_profile = int(row.get("profile", "0"), 0)
            if not 0 <= semantic_profile < 64:
                raise ValueError("profile must be 0..63")
            stream_class = STREAM_CLASSES[row.get("stream_class", "mixed").lower()]
            expert_mask = parse_mask(row["experts"], EXPERTS)
            result.append(
                ExpertChoice(
                    int(row["offset"], 0),
                    int(row["length"], 0),
                    expert_mask,
                    stream_class,
                    semantic_profile | (gain_code << 6),
                    parse_mini_models(row.get("mini_models", ""), expert_mask),
                )
            )
    return result


def load_donor_csv(path: Path | None, profile_bank: bool = False) -> list[DonorChoice]:
    if path is None:
        return []
    result: list[DonorChoice] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("keep", "1").strip().lower() in {"0", "false", "no"}:
                continue
            owner_field = "profile" if profile_bank else "recipient"
            if owner_field not in row or not row[owner_field].strip():
                raise ValueError(f"donor CSV requires {owner_field}")
            result.append(
                DonorChoice(
                    int(row[owner_field], 0),
                    int(row["donor_offset"], 0),
                    int(row["length"], 0),
                    int(row.get("order", "0"), 0),
                )
            )
    return result


def load_costs(path: Path | None, stream_size: int) -> list[float] | None:
    if path is None:
        return None
    with path.open("rb") as source:
        magic, version, flags, size = TRACE_HEADER.unpack(source.read(TRACE_HEADER.size))
        if magic != b"F4TC" or version != 1 or flags != 0 or size != stream_size:
            raise ValueError("cost trace does not match post-R1 stream")
        values = array.array("f")
        values.fromfile(source, stream_size)
        if sys.byteorder != "little":
            values.byteswap()
        if source.read(1):
            raise ValueError("trailing cost trace bytes")
    return list(values)


def adaptive_boundaries(stream_size: int, costs: list[float] | None,
                        base_bytes: int) -> set[int]:
    boundaries = {0, stream_size}
    for start in range(0, stream_size, base_bytes):
        end = min(stream_size, start + base_bytes)
        boundaries.add(start)
        boundaries.add(end)
        if costs is None:
            continue
        bpb = sum(costs[start:end]) / max(1, end - start)
        subdivision = 64 << 10 if bpb >= 1.2 else 256 << 10 if bpb >= 0.9 else base_bytes
        for split in range(start + subdivision, end, subdivision):
            boundaries.add(split)
    return boundaries


def build_blocks(stream_size: int, choices: list[TransformChoice],
                 costs: list[float] | None, base_bytes: int) -> list[TransformChoice]:
    boundaries = adaptive_boundaries(stream_size, costs, base_bytes)
    for choice in choices:
        if choice.offset < 0 or choice.length <= 0 or choice.offset + choice.length > stream_size:
            raise ValueError(f"invalid transform span {choice.offset}+{choice.length}")
        boundaries.add(choice.offset)
        boundaries.add(choice.offset + choice.length)
    points = sorted(boundaries)
    blocks: list[TransformChoice] = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        selected = [choice for choice in choices if choice.offset <= start and end <= choice.offset + choice.length]
        if len(selected) > 1:
            raise ValueError(f"overlapping transform choices at {start}")
        if selected:
            choice = selected[0]
            if choice.operations and (start != choice.offset or end != choice.offset + choice.length):
                raise ValueError("PMD1 operation span cannot be subdivided")
            blocks.append(
                TransformChoice(
                    start, end - start, choice.mask, choice.stream_class,
                    choice.physical_order, list(choice.references), list(choice.operations)
                )
            )
        else:
            blocks.append(TransformChoice(start, end - start, 0))
    explicit_orders = {block.physical_order for block in blocks if block.physical_order is not None}
    if len(explicit_orders) != sum(block.physical_order is not None for block in blocks):
        raise ValueError("duplicate physical_order")
    next_order = 0
    for block in blocks:
        if block.physical_order is None:
            while next_order in explicit_orders:
                next_order += 1
            block.physical_order = next_order
            next_order += 1
    blocks.sort(key=lambda item: int(item.physical_order))
    return blocks


def write_f4tx(path: Path, stream_size: int, blocks: list[TransformChoice]) -> None:
    output = bytearray(F4TX_HEADER.pack(b"F4TX", 1, 0, stream_size, len(blocks)))
    for block in blocks:
        output.extend(
            F4TX_BLOCK.pack(
                block.offset, block.length, int(block.physical_order), block.mask,
                block.stream_class, len(block.references), len(block.operations)
            )
        )
        for reference in block.references:
            output.extend(struct.pack("<Q", reference))
        for operation in block.operations:
            output.extend(F4TX_OPERATION.pack(operation.kind, operation.a, operation.b, operation.c))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def write_f4cp(path: Path, entropy_size: int, stream_digest: bytes,
               donors: list[DonorChoice], experts: list[ExpertChoice],
               profile_bank: bool = False) -> None:
    donors = sorted(donors, key=lambda item: (item.recipient, item.order, item.donor_offset))
    experts = sorted(experts, key=lambda item: item.offset)
    previous_end = 0
    for span in experts:
        if span.length <= 0 or span.mask == 0 or span.offset < previous_end or span.offset + span.length > entropy_size:
            raise ValueError(f"invalid/overlapping expert span at {span.offset}")
        previous_end = span.offset + span.length
    if profile_bank:
        first_use: dict[int, int] = {}
        for span in experts:
            if span.mask & EXPERTS["donor_profile"]:
                profile = span.profile & 63
                first_use[profile] = min(first_use.get(profile, span.offset), span.offset)
        for donor in donors:
            if donor.recipient not in first_use or not 0 <= donor.recipient < 64:
                raise ValueError(f"donor profile {donor.recipient} has no selected span")
            if donor.donor_offset + donor.length > first_use[donor.recipient]:
                raise ValueError(f"donor profile {donor.recipient} is not causal")
    version = 6 if any(span.mini_model_mask for span in experts) else (5 if profile_bank else 4)
    if version == 6 and donors and not profile_bank:
        raise ValueError("F4CP v6 donors require --donor-profile-bank")
    output = bytearray(
        F4CP_HEADER.pack(b"F4CP", version, 0, MIB, 0, entropy_size, len(donors), stream_digest)
    )
    for donor in donors:
        encoded_length = 0 if donor.length == 65536 else donor.length
        output.extend(F4CP_DONOR.pack(
            donor.recipient, donor.donor_offset, encoded_length, donor.order))
    output.extend(struct.pack("<I", len(experts)))
    for span in experts:
        if version >= 6:
            output.extend(F4CP_SPAN_V6.pack(
                span.offset, span.length, span.mask, span.stream_class,
                span.profile, span.mini_model_mask))
        else:
            output.extend(F4CP_SPAN.pack(
                span.offset, span.length, span.mask, span.stream_class, span.profile))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def write_candidate_manifest(path: Path, blocks: list[TransformChoice],
                             costs: list[float] | None, limit: int) -> None:
    ranked = []
    for block in blocks:
        if costs is None:
            bpb = 0.0
        else:
            bpb = sum(costs[block.offset:block.offset + block.length]) / block.length
        ranked.append((bpb, block))
    ranked.sort(key=lambda item: (-item[0], item[1].offset))
    fieldnames = ["candidate_id", "offset", "length", "baseline_bpb", "stages", "status"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        candidate = 0
        for bpb, block in ranked[:limit]:
            for stage in STAGES:
                if stage in {"openzl", "xml_body", "shorthand", "approx_rlz", "wiki", "macros"}:
                    continue
                writer.writerow({
                    "candidate_id": candidate,
                    "offset": block.offset,
                    "length": block.length,
                    "baseline_bpb": f"{bpb:.9f}",
                    "stages": stage,
                    "status": "unmeasured",
                })
                candidate += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stream", type=Path, help="exact post-R1 stream")
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--transform-csv", type=Path)
    parser.add_argument("--expert-csv", type=Path)
    parser.add_argument("--donor-csv", type=Path)
    parser.add_argument("--donor-profile-bank", action="store_true",
                        help="interpret donor CSV owner as reusable profile 0..63")
    parser.add_argument("--cost-trace", type=Path)
    parser.add_argument("--pmd1-json", type=Path)
    parser.add_argument("--pmd1-offset", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--base-kib", type=int, default=1024, choices=(64, 256, 1024))
    parser.add_argument("--entropy-size", type=int,
                        help="transformed entropy-stream size for F4CP spans")
    parser.add_argument("--candidate-limit", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stream_size = args.stream.stat().st_size
    digest = hashlib.sha256(args.stream.read_bytes()).digest()
    choices = load_transform_csv(args.transform_csv)
    if args.pmd1_json:
        choices.append(load_pmd1(args.pmd1_json, args.pmd1_offset))
    costs = load_costs(args.cost_trace, stream_size)
    blocks = build_blocks(stream_size, choices, costs, args.base_kib << 10)
    transform_path = args.prefix.with_suffix(".f4tx")
    write_f4tx(transform_path, stream_size, blocks)

    experts = load_expert_csv(args.expert_csv)
    donors = load_donor_csv(args.donor_csv, args.donor_profile_bank)
    plan_path = args.prefix.with_suffix(".f4cp")
    entropy_size = args.entropy_size or stream_size
    if experts or donors:
        write_f4cp(plan_path, entropy_size, digest, donors, experts, args.donor_profile_bank)
    else:
        plan_path.unlink(missing_ok=True)

    manifest_path = args.prefix.with_suffix(".candidates.csv")
    write_candidate_manifest(manifest_path, blocks, costs, args.candidate_limit)
    report = {
        "post_r1_bytes": stream_size,
        "transform_blocks": len(blocks),
        "selected_transform_blocks": sum(block.mask != 0 for block in blocks),
        "physical_reordering": any(block.physical_order != index for index, block in enumerate(blocks)),
        "expert_spans": len(experts),
        "donor_edges": len(donors),
        "donor_profile_bank": args.donor_profile_bank,
        "transform_plan": str(transform_path),
        "predictor_plan": str(plan_path) if plan_path.exists() else None,
        "candidate_manifest": str(manifest_path),
        "rule": "Only exact-cost/coding winners belong in transform/expert/donor CSV files.",
    }
    report_path = args.prefix.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
