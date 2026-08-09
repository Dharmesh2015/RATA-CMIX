#!/usr/bin/env python3
"""Beam-search page-scoped mini-cmix subsets from an F4MT trace.

This is a shortlist only. The selected F4CP-v7 plan must still be compressed
exactly, including its serialized plan bytes, before it is accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MINI_EXPERT = 1 << 16
DONOR_EXPERT = 1 << 15
STREAM_CLASSES = {
    0: "prose", 1: "xml", 2: "number", 3: "date", 4: "table",
    5: "reference", 6: "url", 7: "identifier", 8: "template",
    9: "list", 10: "mixed",
}
TRACE_DTYPE = np.dtype([("bit", "u1"), ("value", "<i2", (13,))])


@dataclass
class Choice:
    mask: int = 0
    donor: bool = False
    gain_code: int = 0
    gain_bytes: float = 0.0
    loss_bits: float = 0.0


def varint_size(value: int) -> int:
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def bit_loss(logit: np.ndarray, actual: np.ndarray) -> float:
    return float((np.logaddexp(0.0, logit) - actual * logit).sum() / math.log(2.0))


def evaluate_page(records: np.ndarray, beam_width: int) -> tuple[Choice, Choice, float]:
    actual = records["bit"].astype(np.float64)
    values = records["value"].astype(np.float64) / 2048.0
    base = values[:, 0]
    model_logits = np.clip(base[:, None] + values[:, 1:12], -4.0, 4.0)
    donor_delta = values[:, 12]
    baseline_loss = bit_loss(base, actual)

    best_without = Choice(loss_bits=baseline_loss)
    best_with = Choice(donor=True, loss_bits=baseline_loss)
    mask_cache: dict[int, np.ndarray] = {0: np.zeros_like(base)}
    choice_cache: dict[int, tuple[Choice, Choice]] = {}

    def mini_delta(mask: int) -> np.ndarray:
        found = mask_cache.get(mask)
        if found is not None:
            return found
        indexes = [index for index in range(11) if mask & (1 << index)]
        found = np.clip(model_logits[:, indexes].mean(axis=1) - base, -4.0, 4.0)
        mask_cache[mask] = found
        return found

    def test(mask: int) -> tuple[Choice, Choice]:
        cached = choice_cache.get(mask)
        if cached is not None:
            return cached
        delta = mini_delta(mask)
        local_without = Choice(mask=mask, loss_bits=baseline_loss)
        local_with = Choice(mask=mask, donor=True, loss_bits=baseline_loss)
        for gain_code in range(4):
            mini_scale = 0.0625 * (gain_code + 1) if mask else 0.0
            donor_scale = 0.25 * (gain_code + 1)
            if mask:
                loss = bit_loss(base + np.clip(mini_scale * delta, -1.5, 1.5), actual)
                if loss < local_without.loss_bits:
                    local_without = Choice(mask, False, gain_code, 0.0, loss)
            correction = mini_scale * delta + donor_scale * donor_delta
            loss = bit_loss(base + np.clip(correction, -1.5, 1.5), actual)
            if loss < local_with.loss_bits:
                local_with = Choice(mask, True, gain_code, 0.0, loss)
        result = (local_without, local_with)
        choice_cache[mask] = result
        return result

    donor_only_without, donor_only = test(0)
    del donor_only_without
    best_with = donor_only

    # Every model is eligible on its own. Then exhaustively search subsets of
    # the page's strongest individual candidates. This captures arbitrary
    # non-prefix selections without evaluating all 2,047 masks per page.
    ranked_models: list[tuple[float, int]] = []
    for model in range(11):
        without, with_donor = test(1 << model)
        if without.loss_bits < best_without.loss_bits:
            best_without = without
        if with_donor.loss_bits < best_with.loss_bits:
            best_with = with_donor
        ranked_models.append(
            (min(without.loss_bits, with_donor.loss_bits), model))
    ranked_models.sort()
    shortlist = [
        model for _loss, model in ranked_models[:max(1, min(11, beam_width))]
    ]
    for local_mask in range(1, 1 << len(shortlist)):
        mask = 0
        for local_model, model in enumerate(shortlist):
            if local_mask & (1 << local_model):
                mask |= 1 << model
        without, with_donor = test(mask)
        if without.loss_bits < best_without.loss_bits:
            best_without = without
        if with_donor.loss_bits < best_with.loss_bits:
            best_with = with_donor

    best_without.gain_bytes = (baseline_loss - best_without.loss_bits) / 8.0
    best_with.gain_bytes = (baseline_loss - best_with.loss_bits) / 8.0
    return best_without, best_with, baseline_loss / 8.0


def expert_mask(choice: Choice) -> int:
    return (
        (MINI_EXPERT if choice.mask else 0)
        | (DONOR_EXPERT if choice.donor else 0)
    )


def profile_key(row: dict[str, str], choice: Choice) -> tuple[int, int, int, int]:
    return (
        expert_mask(choice),
        int(row["stream_class"]),
        choice.gain_code << 6,
        choice.mask,
    )


def profile_definition_size(profile: tuple[int, int, int, int]) -> int:
    mask, _stream_class, _profile_id, mini_mask = profile
    return varint_size(mask) + 2 + varint_size(mini_mask)


def v7_plan_cost(
    rows: list[dict[str, str]],
    choices: list[Choice],
    selected: list[int] | set[int],
    fixed: int,
    donor_cost: int,
) -> int:
    ordered = sorted(selected)
    if not ordered:
        return 0
    donor_used = any(choices[index].donor for index in ordered)
    # donor_cost is the exact v7 prefix through the donor groups. With no
    # donor bank, version + flags + zero group count costs three bytes.
    size = donor_cost if donor_used else 3
    profiles: dict[tuple[int, int, int, int], int] = {}
    for index in ordered:
        key = profile_key(rows[index], choices[index])
        if key not in profiles:
            profiles[key] = len(profiles)
    size += varint_size(len(profiles))
    size += sum(profile_definition_size(profile) for profile in profiles)
    size += varint_size(len(ordered))
    previous_end = 0
    for index in ordered:
        row = rows[index]
        offset = int(row["offset"])
        length = int(row["length"])
        if offset < previous_end:
            raise ValueError("selected expert spans overlap")
        size += varint_size(offset - previous_end)
        size += varint_size(length)
        size += varint_size(profiles[profile_key(row, choices[index])])
        previous_end = offset + length
    return fixed + size


def span_cost(previous_end: int, row: dict[str, str], choice: Choice) -> int:
    del previous_end
    # Exact standalone v7 cost. Shared selection normally pays less because
    # the profile definition and framing are amortized over many spans.
    donor_prefix = 23 if choice.donor else 3
    profile = profile_key(row, choice)
    return (
        donor_prefix
        + varint_size(1)
        + profile_definition_size(profile)
        + varint_size(1)
        + varint_size(int(row["offset"]))
        + varint_size(int(row["length"]))
        + varint_size(0)
    )


def select(
    rows: list[dict[str, str]],
    choices: list[Choice],
    fixed: int,
    donor_cost: int,
    min_net: float,
) -> tuple[list[int], float, int]:
    positive = [
        index for index, choice in enumerate(choices)
        if choice.gain_bytes > 0.0
    ]
    if not positive:
        return [], 0.0, 0

    by_profile: dict[tuple[int, int, int, int], set[int]] = {}
    for index in positive:
        by_profile.setdefault(profile_key(rows[index], choices[index]), set()).add(index)

    def evaluate(candidate: set[int]) -> tuple[float, int]:
        gain = sum(choices[index].gain_bytes for index in candidate)
        side = v7_plan_cost(rows, choices, candidate, fixed, donor_cost)
        return gain - side, side

    selected: set[int] = set()
    score = 0.0
    side = 0

    # Register a shared profile only when all of its useful pages collectively
    # pay for registration and assignments. Then add any individually useful
    # pages whose profile is already present.
    while True:
        best_set = selected
        best_score = score
        best_side = side
        candidates = list(by_profile.values())
        candidates.extend({index} for index in positive if index not in selected)
        for addition in candidates:
            candidate = selected | addition
            if candidate == selected:
                continue
            candidate_score, candidate_side = evaluate(candidate)
            if candidate_score > best_score + 1.0e-12:
                best_set = candidate
                best_score = candidate_score
                best_side = candidate_side
        if best_set == selected:
            break
        selected, score, side = best_set, best_score, best_side

    # A removal pass catches pages that became unprofitable after neighboring
    # profile choices changed gap and profile-index widths.
    while selected:
        best_set = selected
        best_score = score
        best_side = side
        removals = [{index} for index in selected]
        removals.extend(selected & group for group in by_profile.values())
        for removal in removals:
            if not removal:
                continue
            candidate = selected - removal
            candidate_score, candidate_side = evaluate(candidate)
            if candidate_score > best_score + 1.0e-12:
                best_set = candidate
                best_score = candidate_score
                best_side = candidate_side
        if best_set == selected:
            break
        selected, score, side = best_set, best_score, best_side

    if score < min_net:
        return [], 0.0, 0
    return sorted(selected), score, side

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("spans", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gross-output", type=Path)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--min-net", type=float, default=1.0)
    parser.add_argument("--fixed-plan-cost", type=int, default=0)
    parser.add_argument("--donor-metadata-cost", type=int, default=23)
    parser.add_argument("--baseline-archive", type=int, required=True)
    parser.add_argument("--target", type=int, default=197360)
    args = parser.parse_args()

    with args.trace.open("rb") as source:
        header = source.read(8)
    if header != b"F4MT\x01\x00\x0b\x00":
        raise SystemExit("invalid F4MT v1 trace")
    trace_bytes = args.trace.stat().st_size - 8
    if trace_bytes < 0 or trace_bytes % TRACE_DTYPE.itemsize:
        raise SystemExit("truncated F4MT records")
    records = np.memmap(args.trace, mode="r", dtype=TRACE_DTYPE, offset=8)
    with args.spans.open(newline="", encoding="utf-8-sig") as source:
        rows = [row for row in csv.DictReader(source) if int(row["bits"]) > 0]
    expected = sum(int(row["bits"]) for row in rows)
    if expected != len(records):
        raise SystemExit(f"trace/span mismatch: {len(records)} records, {expected} bits")

    no_donor: list[Choice] = []
    with_donor: list[Choice] = []
    cursor = 0
    baseline_loss = 0.0
    for page, row in enumerate(rows, 1):
        bits = int(row["bits"])
        without, with_profile, page_baseline = evaluate_page(
            records[cursor:cursor + bits], args.beam)
        no_donor.append(without)
        with_donor.append(with_profile if with_profile.loss_bits < without.loss_bits else without)
        baseline_loss += page_baseline
        cursor += bits
        if page % 50 == 0:
            print(f"searched {page}/{len(rows)} pages", flush=True)

    gross_no_donor = [choice.gain_bytes for choice in no_donor
                      if choice.gain_bytes > 0.0]
    gross_with_donor = [choice.gain_bytes for choice in with_donor
                        if choice.gain_bytes > 0.0]

    # Preserve every gross-positive page/mask for full-stream optimization.
    # A page is not discarded merely because it cannot pay a standalone span
    # record; shared mask profiles and assignment runs are charged later.
    gross_output = args.gross_output or args.output.with_name(
        args.output.stem + ".gross.csv"
    )
    gross_output.parent.mkdir(parents=True, exist_ok=True)
    gross_fields = [
        "offset", "length", "scenario", "profile_key", "stream_class",
        "gain_code", "residual_gain", "donor", "mini_mask",
        "mini_models", "gross_gain_bytes", "standalone_span_bytes",
        "standalone_net_bytes", "status",
    ]
    gross_rows = 0
    with gross_output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=gross_fields)
        writer.writeheader()
        for row, mini_choice, combined_choice in zip(
            rows, no_donor, with_donor
        ):
            emitted: set[tuple[int, bool, int]] = set()
            for scenario_name, choice in (
                ("mini_only", mini_choice),
                ("mini_and_optional_donor", combined_choice),
            ):
                key = (choice.mask, choice.donor, choice.gain_code)
                if choice.gain_bytes <= 0.0 or key in emitted:
                    continue
                emitted.add(key)
                models = "+".join(
                    str(model + 1) for model in range(11)
                    if choice.mask & (1 << model)
                )
                standalone = span_cost(0, row, choice)
                writer.writerow({
                    "offset": row["offset"],
                    "length": row["length"],
                    "scenario": scenario_name,
                    "profile_key": (
                        f"{choice.mask:03x}:{int(choice.donor)}:"
                        f"{choice.gain_code}:{row['stream_class']}"
                    ),
                    "stream_class": STREAM_CLASSES[int(row["stream_class"])],
                    "gain_code": choice.gain_code,
                    "residual_gain": 0.25 * (choice.gain_code + 1),
                    "donor": int(choice.donor),
                    "mini_mask": f"0x{choice.mask:03x}",
                    "mini_models": models,
                    "gross_gain_bytes": f"{choice.gain_bytes:.9f}",
                    "standalone_span_bytes": standalone,
                    "standalone_net_bytes": (
                        f"{choice.gain_bytes - standalone:.9f}"
                    ),
                    "status": "gross_positive_global_candidate",
                })
                gross_rows += 1

    selected_a, net_a, side_a = select(
        rows, no_donor, args.fixed_plan_cost, 0, args.min_net)
    selected_b, net_b, side_b = select(
        rows, with_donor, args.fixed_plan_cost,
        args.donor_metadata_cost, args.min_net)
    if net_b > net_a:
        selected, choices, estimated_net, side = selected_b, with_donor, net_b, side_b
        scenario = "mini_and_optional_donor"
    else:
        selected, choices, estimated_net, side = selected_a, no_donor, net_a, side_a
        scenario = "mini_only"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "offset", "length", "experts", "stream_class", "profile",
        "residual_gain", "mini_models", "oracle_gain_bytes", "status",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index in selected:
            row = rows[index]
            choice = choices[index]
            experts = []
            if choice.mask:
                experts.append("mini_cmix")
            if choice.donor:
                experts.append("donor_profile")
            models = "+".join(
                str(model + 1) for model in range(11)
                if choice.mask & (1 << model)
            )
            writer.writerow({
                "offset": row["offset"],
                "length": row["length"],
                "experts": "+".join(experts),
                "stream_class": STREAM_CLASSES[int(row["stream_class"])],
                "profile": 0,
                "residual_gain": 0.25 * (choice.gain_code + 1),
                "mini_models": models,
                "oracle_gain_bytes": f"{choice.gain_bytes:.6f}",
                "status": "candidate_exact_test_required",
            })

    projected = args.baseline_archive - estimated_net
    summary = {
        "trace_records": len(records),
        "pages": len(rows),
        "scenario": scenario,
        "selected_pages": len(selected),
        "baseline_modeled_bytes": baseline_loss,
        "side_data_bytes": side,
        "estimated_net_saving_bytes": estimated_net,
        "baseline_archive_bytes": args.baseline_archive,
        "projected_archive_bytes": projected,
        "target_bytes": args.target,
        "passes_projection": projected < args.target,
        "mini_positive_pages_before_metadata": len(gross_no_donor),
        "mini_positive_gross_bytes": sum(gross_no_donor),
        "mini_max_page_gross_bytes": max(gross_no_donor, default=0.0),
        "optional_donor_positive_pages_before_metadata": len(gross_with_donor),
        "optional_donor_positive_gross_bytes": sum(gross_with_donor),
        "optional_donor_max_page_gross_bytes": max(gross_with_donor, default=0.0),
        "gross_candidate_catalog": str(gross_output),
        "gross_candidate_rows": gross_rows,
        "rule": "projection is a shortlist; only exact F4CP archive size is accepted",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())