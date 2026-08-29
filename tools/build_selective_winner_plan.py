#!/usr/bin/env python3
"""Build exact production F4CP and F4VR plans from final winner rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path


HEADER = struct.Struct("<4sHHIIQH32s")
ASSIGNMENT_V6 = struct.Struct("<HIHB")
SPAN_V6 = struct.Struct("<QIIBBH")
F4VR_EXTERNAL = struct.Struct("<4sHHQ")
IO_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class Winner:
    region: int
    offset: int
    length: int
    mode: str
    mask: int
    mini_mask: int
    strength_code: int
    donors: tuple[tuple[int, int], ...]
    vr_min_length: int
    vr_event_count: int
    baseline: int
    candidate: int
    standalone_side: int
    standalone_net: int
    stream_class: int = 10


@dataclass(frozen=True)
class Scr2Pattern:
    code: int
    data: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_plan", type=Path)
    parser.add_argument("portfolio_selected_csv", type=Path)
    parser.add_argument("output_plan", type=Path)
    parser.add_argument("--post-r1-stream", type=Path)
    parser.add_argument("--output-vr-plan", type=Path)
    parser.add_argument("--vr-events-csv", type=Path)
    parser.add_argument(
        "--portfolio-trials-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "durable trial ledger whose gross-positive rows may become net "
            "winners after reusable F4CP profile costs are shared"
        ),
    )
    parser.add_argument(
        "--scr2-header",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "models"
        / "scr2_tokens.h",
    )
    parser.add_argument("--minimum-net-gain", type=int, default=1)
    parser.add_argument("--baseline-s1-bytes", type=int, default=0)
    parser.add_argument("--candidate-s1-bytes", type=int, default=0)
    parser.add_argument("--discovery-s1-bytes", type=int, default=0)
    return parser.parse_args()


def parse_donors(text: str) -> tuple[tuple[int, int], ...]:
    if not text or text == "-":
        return ()
    donors = []
    for item in text.split(";"):
        offset_text, length_text = item.split(":", 1)
        donors.append((int(offset_text), int(length_text)))
    return tuple(donors)


def load_cost_replay_events(
    path: Path | None,
) -> dict[tuple[int, str], list[tuple[int, int]]]:
    result: dict[tuple[int, str], set[tuple[int, int]]] = {}
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            try:
                key = (int(row["recipient_region"]), row["mode"])
                event = (int(row["event_offset"]), int(row["pattern"]))
            except (KeyError, ValueError) as error:
                raise SystemExit(
                    f"invalid virtual-replay event at CSV row {row_number}"
                ) from error
            result.setdefault(key, set()).add(event)
    return {
        key: sorted(events)
        for key, events in result.items()
    }


def varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode a negative value")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        output.append(byte)
        if not value:
            return bytes(output)


def varint_size(value: int) -> int:
    return len(varint(value))


def signed_delta(value: int) -> int:
    return value << 1 if value >= 0 else ((-(value + 1)) << 1) | 1


def archive_v7_size(
    assignments: list[tuple[int, int, int, int]],
    spans: list[tuple[int, int, int, int, int, int]],
) -> int:
    if not assignments and not spans:
        return 0
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for profile_id, offset, length, order in assignments:
        groups.setdefault(profile_id, []).append((offset, length, order))

    size = 2 + varint_size(len(groups))
    for profile_id in sorted(groups):
        donors = sorted(groups[profile_id], key=lambda item: item[2])
        size += varint_size(profile_id) + varint_size(len(donors))
        previous = 0
        for index, (offset, _length, order) in enumerate(donors):
            if order != index:
                raise ValueError("non-contiguous donor order")
            shifted = offset >> 8
            encoded = shifted if index == 0 else signed_delta(shifted - previous)
            size += varint_size(encoded) + 1
            previous = shifted

    profiles: list[tuple[int, int, int, int]] = []
    span_profiles: list[int] = []
    for _offset, _length, mask, stream_class, profile_id, mini_mask in spans:
        profile = (mask, stream_class, profile_id, mini_mask)
        try:
            index = profiles.index(profile)
        except ValueError:
            index = len(profiles)
            profiles.append(profile)
        span_profiles.append(index)

    size += varint_size(len(profiles))
    for mask, _stream_class, _profile_id, mini_mask in profiles:
        size += varint_size(mask) + 2 + varint_size(mini_mask)

    size += varint_size(len(spans))
    previous_end = 0
    for span, profile_index in zip(spans, span_profiles):
        offset, length = span[0], span[1]
        if offset < previous_end:
            raise ValueError("overlapping selected spans")
        size += varint_size(offset - previous_end)
        size += varint_size(length)
        size += varint_size(profile_index)
        previous_end = offset + length
    return size


def load_header(path: Path) -> tuple[int, int, bytes]:
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise SystemExit("candidate plan is truncated")
    magic, version, flags, chunk_size, _seed, stream_size, _count, digest = (
        HEADER.unpack_from(payload)
    )
    if magic != b"F4CD" or flags != 1 or version not in range(1, 7):
        raise SystemExit("expected an F4CD discovery plan")
    return chunk_size, stream_size, digest


def winner_from_row(row: dict[str, str], row_number: int) -> Winner:
    try:
        region = int(row["recipient_region"])
        profile_id = int(row["profile_id"])
        baseline = int(row["baseline_payload_bytes"])
        candidate = int(row["candidate_payload_bytes"])
        standalone_side = int(row["standalone_plan_bytes"])
        standalone_net = int(row["net_gain_bytes"])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid portfolio row {row_number}") from error
    expected_net = baseline - candidate - standalone_side
    if standalone_net != expected_net:
        raise SystemExit(
            f"inconsistent net gain at CSV row {row_number}: "
            f"{standalone_net} != {expected_net}"
        )
    try:
        return Winner(
            region=region,
            offset=int(row["recipient_offset"]),
            length=int(row["recipient_length"]),
            mode=row["mode"],
            mask=int(row["expert_mask"]),
            mini_mask=int(row["mini_model_mask"]),
            strength_code=(profile_id >> 6) & 3,
            donors=parse_donors(row["donors"]),
            vr_min_length=int(row.get("vr_min_length") or 0),
            vr_event_count=int(row.get("vr_event_count") or 0),
            baseline=baseline,
            candidate=candidate,
            standalone_side=standalone_side,
            standalone_net=standalone_net,
            stream_class=int(row["stream_class"]),
        )
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid portfolio row {row_number}") from error


def load_winners(path: Path, minimum_net_gain: int) -> list[Winner]:
    latest: dict[int, Winner] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            winner = winner_from_row(row, row_number)
            active = winner.mask != 0 or winner.vr_min_length != 0
            if active and row["status"] != "accepted_after_side":
                raise SystemExit(
                    f"active plan is not accepted at CSV row {row_number}"
                )
            latest[winner.region] = winner
    return [
        winner
        for _, winner in sorted(latest.items())
        if (winner.mask != 0 or winner.vr_min_length != 0)
        and winner.standalone_net >= minimum_net_gain
    ]


def winner_action_key(winner: Winner) -> tuple[object, ...]:
    return (
        winner.region,
        winner.offset,
        winner.length,
        winner.mask,
        winner.mini_mask,
        winner.strength_code,
        winner.donors,
        winner.vr_min_length,
        winner.vr_event_count,
        winner.stream_class,
    )


def load_trial_candidates(paths: list[Path]) -> list[Winner]:
    latest: dict[tuple[object, ...], Winner] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"portfolio trial ledger does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as source:
            for row_number, row in enumerate(csv.DictReader(source), start=2):
                winner = winner_from_row(row, row_number)
                active = winner.mask != 0 or winner.vr_min_length != 0
                if active and winner.baseline > winner.candidate:
                    latest[winner_action_key(winner)] = winner
    return sorted(
        latest.values(),
        key=lambda item: (item.region, item.mode, winner_action_key(item)),
    )

def parse_scr2_patterns(path: Path) -> list[Scr2Pattern]:
    text = path.read_text(encoding="utf-8")
    bytes_match = re.search(
        r"kPatternBytes\[\d+\]\s*=\s*\{(.*?)\};", text, re.S
    )
    tokens_match = re.search(
        r"kTokens\[kTokenCount \+ 1\]\s*=\s*\{(.*?)\};", text, re.S
    )
    if not bytes_match or not tokens_match:
        raise SystemExit(f"cannot parse SCR2 token header: {path}")
    packed = bytes(
        int(value, 16)
        for value in re.findall(r"0x([0-9a-fA-F]{1,2})", bytes_match.group(1))
    )
    records = re.findall(r"\{([^{}]+)\}", tokens_match.group(1))
    patterns: list[Scr2Pattern] = []
    for code, record in enumerate(records):
        values = [
            int(value)
            for value in re.findall(r"(?<![0-9A-Za-z_])(\d+)u?", record)
        ]
        if code == 0:
            continue
        if len(values) != 9:
            raise SystemExit(f"invalid SCR2 token record {code}")
        offset, length = values[4], values[5]
        data = packed[offset : offset + length]
        if len(data) != length:
            raise SystemExit(f"truncated SCR2 token record {code}")
        patterns.append(Scr2Pattern(code, data))
    if len(patterns) != 128:
        raise SystemExit(f"expected 128 SCR2 patterns, found {len(patterns)}")
    return patterns


def find_scr2_events(
    stream: bytes,
    winner: Winner,
    patterns: list[Scr2Pattern],
) -> list[tuple[int, int]]:
    by_first: dict[int, list[Scr2Pattern]] = {}
    for pattern in patterns:
        if len(pattern.data) >= winner.vr_min_length:
            by_first.setdefault(pattern.data[0], []).append(pattern)
    for choices in by_first.values():
        choices.sort(key=lambda item: (-len(item.data), item.code))

    events: list[tuple[int, int]] = []
    local = 0
    view = memoryview(stream)
    while local < winner.length:
        absolute = winner.offset + local
        match: Scr2Pattern | None = None
        for pattern in by_first.get(view[absolute], ()):
            end = absolute + len(pattern.data)
            if end > winner.offset + winner.length:
                continue
            if absolute // IO_BLOCK_BYTES != (end - 1) // IO_BLOCK_BYTES:
                continue
            if view[absolute:end] == pattern.data:
                match = pattern
                break
        if match is None:
            local += 1
            continue
        events.append((absolute, match.code))
        local += len(match.data)
    return events


def build_f4vr_payload(
    winners: list[Winner],
    stream_path: Path,
    stream_size: int,
    digest: bytes,
    header_path: Path,
    cost_events: dict[tuple[int, str], list[tuple[int, int]]],
) -> tuple[bytes, list[tuple[int, int]], dict[int, bytes]]:
    if not any(winner.vr_min_length for winner in winners):
        return b"", [], {}
    if not stream_path:
        raise SystemExit("--post-r1-stream is required for selected SCR2 winners")
    stream = stream_path.read_bytes()
    if len(stream) != stream_size:
        raise SystemExit(
            f"post-R1 stream size mismatch: {len(stream)} != {stream_size}"
        )
    if hashlib.sha256(stream).digest() != digest:
        raise SystemExit("post-R1 stream SHA-256 does not match F4CD header")

    patterns = parse_scr2_patterns(header_path)
    pattern_by_code = {pattern.code: pattern.data for pattern in patterns}
    events: list[tuple[int, int]] = []
    for winner in winners:
        if not winner.vr_min_length:
            continue
        if winner.vr_min_length == 0xFFFF:
            found = cost_events.get((winner.region, winner.mode), [])
            previous_end = winner.offset
            for offset, code in found:
                pattern = pattern_by_code.get(code)
                if pattern is None:
                    raise SystemExit(
                        f"unknown SCR2 pattern {code} in region {winner.region}"
                    )
                end = offset + len(pattern)
                if (
                    offset < winner.offset
                    or end > winner.offset + winner.length
                    or offset < previous_end
                    or offset // IO_BLOCK_BYTES != (end - 1) // IO_BLOCK_BYTES
                    or stream[offset:end] != pattern
                ):
                    raise SystemExit(
                        f"invalid cost-positive SCR2 event "
                        f"{offset}:{code} in region {winner.region}"
                    )
                previous_end = end
        else:
            found = find_scr2_events(stream, winner, patterns)
        if len(found) != winner.vr_event_count:
            raise SystemExit(
                f"SCR2 event mismatch in region {winner.region}: "
                f"{len(found)} != {winner.vr_event_count}"
            )
        events.extend(found)

    used_codes: list[int] = []
    compact: dict[int, int] = {}
    for _offset, code in events:
        if code not in compact:
            compact[code] = len(used_codes)
            used_codes.append(code)

    payload = bytearray()
    payload.append(1)
    payload.extend(struct.pack("<H", len(used_codes)))
    for code in used_codes:
        data = pattern_by_code[code]
        payload.extend(struct.pack("<H", len(data)))
        payload.extend(data)
    payload.extend(struct.pack("<I", len(events)))
    previous_end = 0
    for offset, code in events:
        if offset < previous_end:
            raise SystemExit("overlapping F4VR events")
        payload.extend(varint(offset - previous_end))
        payload.extend(varint(compact[code]))
        previous_end = offset + len(pattern_by_code[code])
    return bytes(payload), events, {
        compact[code]: pattern_by_code[code] for code in used_codes
    }


def f4cp_layout(
    winners: list[Winner],
) -> tuple[
    dict[tuple[tuple[int, int], ...], int],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int, int, int]],
] | None:
    donor_sets = sorted({winner.donors for winner in winners if winner.donors})
    if len(donor_sets) > 64:
        return None
    donor_profiles = {
        donors: profile_id for profile_id, donors in enumerate(donor_sets)
    }

    assignments: list[tuple[int, int, int, int]] = []
    for donors, profile_id in donor_profiles.items():
        for order, (offset, length) in enumerate(donors):
            if offset & 255:
                raise SystemExit(f"unaligned donor offset: {offset}")
            if length < 256 or length > 65536 or length & (length - 1):
                raise SystemExit(f"invalid donor length: {length}")
            assignments.append((profile_id, offset, length, order))

    spans: list[tuple[int, int, int, int, int, int]] = []
    previous_end = 0
    for winner in sorted(winners, key=lambda item: item.offset):
        if not winner.mask:
            continue
        if winner.offset < previous_end:
            raise SystemExit(f"overlapping winner at region {winner.region}")
        for donor_offset, donor_length in winner.donors:
            if donor_offset + donor_length > winner.offset:
                raise SystemExit(
                    f"noncausal donor for region {winner.region}: "
                    f"{donor_offset}:{donor_length}"
                )
        donor_profile = donor_profiles.get(winner.donors, 0)
        final_profile_id = (winner.strength_code << 6) | donor_profile
        spans.append(
            (
                winner.offset,
                winner.length,
                winner.mask,
                winner.stream_class,
                final_profile_id,
                winner.mini_mask,
            )
        )
        previous_end = winner.offset + winner.length
    return donor_profiles, assignments, spans


def f4cp_objective(winners: list[Winner]) -> tuple[int, int] | None:
    layout = f4cp_layout(winners)
    if layout is None:
        return None
    _profiles, assignments, spans = layout
    gross = sum(winner.baseline - winner.candidate for winner in winners)
    side = archive_v7_size(assignments, spans)
    return gross - side, side


def reusable_profile_key(winner: Winner) -> tuple[object, ...]:
    return (
        winner.mask,
        winner.mini_mask,
        winner.strength_code,
        winner.donors,
        winner.stream_class,
    )


def optimize_shared_f4cp_costs(
    initial: list[Winner], candidates: list[Winner]
) -> tuple[list[Winner], dict[str, object]]:
    selected = [winner for winner in initial if winner.mask]
    locked_regions = {winner.region for winner in initial}
    pool = [
        winner
        for winner in candidates
        if winner.mask
        and not winner.vr_min_length
        and winner.region not in locked_regions
    ]
    best_by_action: dict[tuple[object, ...], Winner] = {}
    for candidate in pool:
        key = winner_action_key(candidate)
        previous = best_by_action.get(key)
        if previous is None or candidate.candidate < previous.candidate:
            best_by_action[key] = candidate
    pool = list(best_by_action.values())

    initial_score = f4cp_objective(selected)
    if initial_score is None:
        raise SystemExit("initial winner set exceeds 64 donor profiles")
    current_score = initial_score[0]
    selected_from_trials: list[Winner] = []

    while True:
        occupied = {winner.region for winner in selected}
        families: dict[tuple[object, ...], dict[int, Winner]] = {}
        for candidate in pool:
            if candidate.region in occupied:
                continue
            family = families.setdefault(reusable_profile_key(candidate), {})
            previous = family.get(candidate.region)
            if previous is None or candidate.candidate < previous.candidate:
                family[candidate.region] = candidate

        best_delta = 0
        best_addition: list[Winner] = []
        for family in families.values():
            ranked = sorted(
                family.values(),
                key=lambda item: (
                    -(item.baseline - item.candidate),
                    item.offset,
                    item.mode,
                ),
            )
            for count in range(1, len(ranked) + 1):
                addition = ranked[:count]
                score = f4cp_objective(selected + addition)
                if score is None:
                    continue
                delta = score[0] - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_addition = addition
        if not best_addition:
            break
        selected.extend(best_addition)
        selected_from_trials.extend(best_addition)
        current_score += best_delta

    final_score = f4cp_objective(selected)
    assert final_score is not None
    stats: dict[str, object] = {
        "gross_positive_f4cp_trials": len(pool),
        "selected_after_shared_f4cp_cost": len(selected_from_trials),
        "selected_shared_f4cp_regions": sorted(
            winner.region for winner in selected_from_trials
        ),
        "initial_f4cp_net_bytes": initial_score[0],
        "optimized_f4cp_net_bytes": final_score[0],
        "shared_f4cp_incremental_net_bytes": final_score[0] - initial_score[0],
    }
    return [winner for winner in initial if not winner.mask] + selected, stats


def virtual_replay_lower_bounds(
    candidates: list[Winner],
) -> list[dict[str, object]]:
    families: dict[str, list[Winner]] = {}
    for candidate in candidates:
        if candidate.vr_min_length:
            families.setdefault(candidate.mode, []).append(candidate)
    reports: list[dict[str, object]] = []
    for mode, family in sorted(families.items()):
        gross = sum(winner.baseline - winner.candidate for winner in family)
        events = sum(winner.vr_event_count for winner in family)
        # F4VR-v1 needs at least one byte for each event gap and one byte for
        # each compact pattern ID. This excludes the pattern dictionary and
        # header, so it is an optimistic lower bound, not an estimated cost.
        event_floor = 2 * events + 7 if events else 0
        reports.append(
            {
                "mode": mode,
                "regions": len(family),
                "gross_payload_gain_bytes": gross,
                "events": events,
                "optimistic_f4vr_event_floor_bytes": event_floor,
                "optimistic_net_before_pattern_table_bytes": gross
                - event_floor,
            }
        )
    return reports


def main() -> int:
    args = parse_args()
    chunk_size, stream_size, digest = load_header(args.candidate_plan)
    accepted_winners = load_winners(
        args.portfolio_selected_csv, args.minimum_net_gain
    )
    trial_candidates = load_trial_candidates(args.portfolio_trials_csv)
    winners, amortization_stats = optimize_shared_f4cp_costs(
        accepted_winners, trial_candidates
    )
    vr_lower_bounds = virtual_replay_lower_bounds(trial_candidates)
    cost_replay_events = load_cost_replay_events(args.vr_events_csv)
    winners.sort(key=lambda item: item.offset)

    previous_end = 0
    for winner in winners:
        if winner.offset < previous_end:
            raise SystemExit(f"overlapping winner at region {winner.region}")
        if winner.offset + winner.length > stream_size:
            raise SystemExit(f"winner outside stream at region {winner.region}")
        for donor_offset, donor_length in winner.donors:
            if donor_offset + donor_length > winner.offset:
                raise SystemExit(
                    f"noncausal donor for region {winner.region}: "
                    f"{donor_offset}:{donor_length}"
                )
        previous_end = winner.offset + winner.length

    layout = f4cp_layout(winners)
    if layout is None:
        raise SystemExit("selected plan exceeds 64 donor profiles")
    donor_profiles, assignments, spans = layout

    if len(assignments) > 0xFFFF:
        raise SystemExit("F4CP-v6 supports at most 65535 assignments")

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("wb") as output:
        output.write(
            HEADER.pack(
                b"F4CP",
                6,
                0,
                chunk_size,
                max((item[2] for item in assignments), default=0),
                stream_size,
                len(assignments),
                digest,
            )
        )
        for profile_id, offset, length, order in assignments:
            encoded_length = 0 if length == 65536 else length
            output.write(
                ASSIGNMENT_V6.pack(profile_id, offset, encoded_length, order)
            )
        output.write(struct.pack("<I", len(spans)))
        for span in spans:
            output.write(SPAN_V6.pack(*span))

    f4cp_archive_bytes = archive_v7_size(assignments, spans)
    vr_path = args.output_vr_plan or args.output_plan.with_suffix(".f4vr")
    vr_payload, vr_events, vr_patterns = build_f4vr_payload(
        winners,
        args.post_r1_stream,
        stream_size,
        digest,
        args.scr2_header,
        cost_replay_events,
    )
    if vr_payload:
        vr_path.write_bytes(
            F4VR_EXTERNAL.pack(b"F4VR", 1, 0, stream_size) + vr_payload
        )
    elif vr_path.exists():
        vr_path.unlink()

    f4vr_archive_bytes = len(vr_payload)
    gross_gain = sum(winner.baseline - winner.candidate for winner in winners)
    standalone_net = sum(winner.standalone_net for winner in winners)
    combined_side = f4cp_archive_bytes + f4vr_archive_bytes
    plan_net = gross_gain - combined_side
    fell_back_to_baseline = bool(winners and plan_net <= 0)
    if fell_back_to_baseline:
        winners = []
        donor_profiles = {}
        assignments = []
        spans = []
        vr_payload = b""
        vr_events = []
        vr_patterns = {}
        f4cp_archive_bytes = 0
        f4vr_archive_bytes = 0
        gross_gain = 0
        standalone_net = 0
        combined_side = 0
        plan_net = 0
        args.output_plan.write_bytes(
            HEADER.pack(
                b"F4CP", 6, 0, chunk_size, 0, stream_size, 0, digest
            )
            + struct.pack("<I", 0)
        )
        if vr_path.exists():
            vr_path.unlink()

    s1_growth = max(
        0, args.candidate_s1_bytes - args.baseline_s1_bytes
    ) if args.baseline_s1_bytes and args.candidate_s1_bytes else None
    hutter_projected_net = (
        plan_net - s1_growth if s1_growth is not None else None
    )

    selected_csv = args.output_plan.with_suffix(
        args.output_plan.suffix + ".selected.csv"
    )
    with selected_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "region",
                "offset",
                "length",
                "mode",
                "mask",
                "mini_mask",
                "strength_code",
                "donors",
                "vr_min_length",
                "vr_event_count",
                "gross_gain",
                "standalone_side",
                "standalone_net",
            ]
        )
        for winner in winners:
            writer.writerow(
                [
                    winner.region,
                    winner.offset,
                    winner.length,
                    winner.mode,
                    winner.mask,
                    winner.mini_mask,
                    winner.strength_code,
                    ";".join(f"{o}:{n}" for o, n in winner.donors) or "-",
                    winner.vr_min_length,
                    winner.vr_event_count,
                    winner.baseline - winner.candidate,
                    winner.standalone_side,
                    winner.standalone_net,
                ]
            )

    summary = {
        "format": "F4CP-v6 external/v7 archive + F4VR-v1",
        "selected_regions": len(winners),
        "donor_profiles": len(donor_profiles),
        "donor_assignments": len(assignments),
        "expert_spans": len(spans),
        "virtual_replay_regions": sum(
            winner.vr_min_length != 0 for winner in winners
        ),
        "virtual_replay_events": len(vr_events),
        "virtual_replay_patterns": len(vr_patterns),
        "gross_payload_gain_bytes": gross_gain,
        "sum_conservative_standalone_net_bytes": standalone_net,
        "f4cp_compact_archive_bytes": f4cp_archive_bytes,
        "f4vr_compact_archive_bytes": f4vr_archive_bytes,
        "combined_archive_plan_bytes": combined_side,
        "projected_net_after_combined_plans_bytes": plan_net,
        "fell_back_to_baseline": fell_back_to_baseline,
        "discovery_s1_bytes_not_scored": args.discovery_s1_bytes or None,
        "baseline_s1_bytes": args.baseline_s1_bytes or None,
        "candidate_s1_bytes": args.candidate_s1_bytes or None,
        "s1_growth_bytes": s1_growth,
        "projected_hutter_net_bytes": hutter_projected_net,
        "shared_side_data_optimizer": amortization_stats,
        "virtual_replay_event_cost_lower_bounds": vr_lower_bounds,
        "external_f4cp_bytes": args.output_plan.stat().st_size,
        "external_f4cp_sha256": hashlib.sha256(
            args.output_plan.read_bytes()
        ).hexdigest(),
        "external_f4vr_bytes": vr_path.stat().st_size if vr_payload else 0,
        "external_f4vr_sha256": (
            hashlib.sha256(vr_path.read_bytes()).hexdigest()
            if vr_payload
            else None
        ),
        "stream_bytes": stream_size,
        "stream_sha256": digest.hex(),
        "warning": (
            "Projected gains must be confirmed by one complete compression "
            "and exact decompression; S1 growth is included but arithmetic "
            "coder whole-file effects still require final validation."
        ),
    }
    summary_path = args.output_plan.with_suffix(
        args.output_plan.suffix + ".summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
