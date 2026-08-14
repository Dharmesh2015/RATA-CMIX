#!/usr/bin/env python3
"""Write an F4CP v6 profile-bank binary plan for the no-mutation donor-profile
oracle (PostR1Experts::kDonorProfile), driven by DonorPlan::LoadExternal.

Binary format (all integers little-endian), verified against
src/donor_plan.cpp DonorPlan::LoadExternal / ReadExpertSpans / Initialize:

Header:
  magic[4]        "F4CD" (discovery, flags=1) or "F4CP" (production, flags=0)
  version u16     6  (has_mini_model_mask branch of ReadExpertSpans)
  flags   u16     0 or 1, must match magic per LoadExternal's production/
                  discovery check
  chunk_size u32  DonorPlan::kChunkSize (1 << 20)
  seed_size  u32  0 for version>=2 (per-assignment length is read instead)
  planned_size u64  stream byte length (must equal the actual input size)
  count u16       number of Assignment records
  digest[32]      unused by the loader beyond byte-count; zero-filled

count x Assignment (version>=3):
  recipient u16   in profile-bank mode this is the PROFILE SLOT (0-63), not
                  a stream region -- see Initialize()'s `profile_bank_` branch
  donor_offset u32
  length u16
  order u8

  Then (version>=4):
span_count u32
span_count x ExpertSpan (version>=6, has_mini_model_mask=true):
  offset u64
  length u32
  expert_mask u32
  stream_class u8
  profile_id u8   low 6 bits = profile slot (0-63, matches Assignment.recipient)
                  bits 6-7   = residual-gain selector:
                              0 -> g=0.25, 1 (default) -> g=0.50,
                              2 -> g=0.75, 3 -> g=1.00
  mini_model_mask u16  0 (mini-cmix expert inactive for this run)

Hard constraint (Initialize(), profile_bank_ branch): profile_id & 63 gives at
most 64 distinct profile slots per plan file. Spans must be offset-sorted and
non-overlapping (span.offset >= previous_span_end).
"""
from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

K_ORACLE = 1 << 14         # PostR1Experts::kOracle
K_DONOR_PROFILE = 1 << 15  # PostR1Experts::kDonorProfile
# kOracle is required for span_oracle_ accumulation at all (postr1_experts.cpp
# line 129: `if (evaluation_mask_ & kOracle) span_oracle_.push_back(span);`).
# kDonorProfile alone activates the expert but writes nothing comparable.
K_SPAN_MASK = K_ORACLE | K_DONOR_PROFILE

STREAM_CLASS = {
    "prose": 0, "xml": 1, "number": 2, "date": 3, "table": 4,
    "reference": 5, "url": 6, "identifier": 7, "template": 8,
    "list": 9, "mixed": 10,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recipients_csv", type=Path,
                     help="grouped recipients CSV (pack_id,...,post_r1_start,"
                          "post_r1_end,post_r1_length,...,class_name,...)")
    ap.add_argument("selection_csv", type=Path,
                     help="rows: recipient_group,donor_offset,donor_length[,order]; "
                          "multiple ordered rows may share one recipient")
    ap.add_argument("stream_bytes", type=int)
    ap.add_argument("output_plan", type=Path)
    ap.add_argument("--strength", type=int, default=1, choices=[0, 1, 2, 3],
                     help="0=g0.25 1=g0.50(default) 2=g0.75 3=g1.00")
    ap.add_argument("--discovery", action="store_true",
                     help="write F4CD (flags=1) instead of F4CP (flags=0); "
                          "both are accepted by LoadExternal identically for "
                          "this profile-bank path, discovery just marks the "
                          "plan as non-production for downstream tooling")
    args = ap.parse_args()

    groups = {}
    with args.recipients_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[int(row["pack_id"])] = {
                "start": int(row["post_r1_start"]),
                "end": int(row["post_r1_end"]),
                "length": int(row["post_r1_length"]),
                "class_name": row.get("class_name", "mixed") or "mixed",
            }

    selections: dict[int, list[tuple[int, int, int]]] = {}
    with args.selection_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            recipient = int(row["recipient_group"])
            donors = selections.setdefault(recipient, [])
            order_text = row.get("order", "").strip()
            order = int(order_text) if order_text else len(donors)
            donors.append((
                int(row["donor_offset"]),
                int(row["donor_length"]),
                order,
            ))
    if len(selections) > 64:
        raise SystemExit(
            f"{len(selections)} recipients requested but the profile-bank "
            f"mechanism caps at 64 profile slots per plan (Initialize() "
            f"masks profile_id & 63)")

    assignments = []  # (recipient=slot, donor_offset, length, order)
    spans = []        # (offset, length, expert_mask, stream_class, profile_id, mini_mask)
    for slot, recipient in enumerate(sorted(selections)):
        g = groups.get(recipient)
        if g is None:
            raise SystemExit(f"recipient group {recipient} not found in {args.recipients_csv}")
        donors = sorted(selections[recipient], key=lambda item: item[2])
        orders = [order for _offset, _length, order in donors]
        if len(set(orders)) != len(orders) or any(
                not 0 <= order <= 255 for order in orders):
            raise SystemExit(
                f"recipient {recipient}: donor order values must be unique bytes")
        for donor_offset, donor_length, order in donors:
            if donor_offset + donor_length > g["start"]:
                raise SystemExit(
                    f"causality violation: recipient {recipient} starts at "
                    f"{g['start']} but donor ends at {donor_offset + donor_length}")
            if (donor_length < 256 or donor_length > 65536 or
                    donor_length & (donor_length - 1)):
                raise SystemExit(
                    f"recipient {recipient}: donor_length={donor_length} must "
                    f"be a power of two from 256 through 65536")
            if donor_offset & 255:
                raise SystemExit(
                    f"recipient {recipient}: donor_offset={donor_offset} is not "
                    f"256-byte aligned (donor_plan.cpp WriteArchive requires "
                    f"donor_offset & 255 == 0); nearest aligned offsets are "
                    f"{donor_offset & ~255} and {(donor_offset & ~255) + 256}")
            assignments.append((slot, donor_offset, donor_length, order))
        profile_id = (slot & 0x3F) | ((args.strength & 3) << 6)
        stream_class = STREAM_CLASS.get(g["class_name"], 10)
        spans.append((g["start"], g["length"], K_SPAN_MASK,
                      stream_class, profile_id, 0))

    if len(assignments) > 65535:
        raise SystemExit("too many donor assignments for the F4CP header")

    spans.sort(key=lambda s: s[0])
    previous_end = 0
    for offset, length, *_ in spans:
        if offset < previous_end:
            raise SystemExit(
                f"overlapping spans at offset {offset} (previous span ends "
                f"at {previous_end}) -- selected recipients must not overlap")
        previous_end = offset + length

    out = bytearray()
    magic = b"F4CD" if args.discovery else b"F4CP"
    flags = 1 if args.discovery else 0
    out += magic
    out += struct.pack("<HHII", 6, flags, 1 << 20, 0)
    out += struct.pack("<Q", args.stream_bytes)
    out += struct.pack("<H", len(assignments))
    out += b"\x00" * 32

    for recipient, donor_offset, length, order in assignments:
        encoded_length = 0 if length == 65536 else length
        out += struct.pack(
            "<HIHB", recipient, donor_offset, encoded_length, order)

    out += struct.pack("<I", len(spans))
    for offset, length, expert_mask, stream_class, profile_id, mini_mask in spans:
        out += struct.pack("<QIIBBH", offset, length, expert_mask,
                            stream_class, profile_id, mini_mask)

    args.output_plan.write_bytes(bytes(out))
    print(f"wrote {args.output_plan}: {len(assignments)} assignments, "
          f"{len(spans)} spans, {len(out)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
