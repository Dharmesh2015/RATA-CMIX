#!/usr/bin/env python3
"""Evaluate causal residual corrections from an FXOT-v1 trace.

The per-bit oracle is deliberately labelled as noncausal: it is only a ceiling.
Every deployable result is trained on an earlier chronological split and scored
on later records. No current/future bit is included in its feature vector.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<4sHHQQQ")
RECORD = np.dtype([
    ("base_p", "<u2"), ("final_p", "<u2"),
    ("ppmd", "u1"), ("lstm", "u1"), ("fxcm", "u1"),
    ("flags", "u1"), ("ppm_meta", "u1"), ("match", "u1"),
], align=False)
LN2 = math.log(2.0)
DELTA_GRID = np.asarray([-2.0, -1.0, -0.5, -0.25, 0.0,
                         0.25, 0.5, 1.0, 2.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--baseline-total", type=int, default=108_492_825)
    parser.add_argument("--target-total", type=int, default=100_000_000)
    parser.add_argument("--entropy-bytes", type=int, default=587_138_826)
    parser.add_argument("--sample-bits", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=923)
    parser.add_argument("--code-overhead-bytes", type=int, default=4096)
    parser.add_argument("--evolution-bits", type=int, default=250_000)
    parser.add_argument("--output-model", type=Path,
                        default=Path("residual_oracle_model.json"))
    return parser.parse_args()


def logits_from_probability(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1.0 / 65536.0, 65535.0 / 65536.0)
    return np.log(p) - np.log1p(-p)


def dequantize_logit(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float64) * (16.0 / 255.0) - 8.0


def bit_loss(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.logaddexp(0.0, z) - y * z) / LN2


def unpack(records: np.ndarray) -> tuple[np.ndarray, ...]:
    flags = records["flags"]
    y = (flags & 1).astype(np.float64)
    bitpos = ((flags >> 2) & 7).astype(np.int64)
    stream = ((flags >> 5) & 7).astype(np.int64)
    order = (records["ppm_meta"] >> 3).astype(np.int64)
    escape = (records["ppm_meta"] & 7).astype(np.int64)
    base_z = logits_from_probability(records["base_p"] / 65536.0)
    final_z = logits_from_probability(records["final_p"] / 65536.0)
    return (y, bitpos, stream, order, escape, base_z, final_z,
            dequantize_logit(records["ppmd"]),
            dequantize_logit(records["lstm"]),
            dequantize_logit(records["fxcm"]),
            records["match"].astype(np.float64))


def lagged(values: np.ndarray, distance: int) -> np.ndarray:
    result = np.zeros_like(values)
    if distance < len(values):
        result[distance:] = values[:-distance]
    return result


def prior_mean(values: np.ndarray, window: int) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    end = np.arange(len(values), dtype=np.int64)
    begin = np.maximum(0, end - window)
    count = np.maximum(1, end - begin)
    return (prefix[end] - prefix[begin]) / count


def features(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (y, bitpos, stream, order, escape, base_z, final_z,
     ppmd_z, lstm_z, fxcm_z, match) = unpack(records)
    n = len(records)
    x = np.zeros((n, 33), dtype=np.float32)
    x[:, 0] = 1.0
    x[:, 1] = np.clip(ppmd_z - final_z, -6.0, 6.0)
    x[:, 2] = np.clip(lstm_z - final_z, -6.0, 6.0)
    x[:, 3] = np.clip(fxcm_z - final_z, -6.0, 6.0)
    x[:, 4] = np.clip(final_z - base_z, -4.0, 4.0)
    x[:, 5] = order / 31.0
    x[:, 6] = escape / 7.0
    x[:, 7] = np.log1p(match) / math.log(256.0)
    x[np.arange(n), 8 + bitpos] = 1.0
    x[np.arange(n), 16 + stream] = 1.0
    q = 1.0 / (1.0 + np.exp(-np.clip(final_z, -20.0, 20.0)))
    residual = y - q
    x[:, 24] = np.minimum(np.abs(final_z), 8.0) / 8.0
    x[:, 25] = np.std(
        np.stack((ppmd_z, lstm_z, fxcm_z, final_z)), axis=0) / 4.0
    x[:, 26] = lagged(residual, 1)
    x[:, 27] = lagged(residual, 2)
    x[:, 28] = lagged(residual, 4)
    x[:, 29] = prior_mean(np.abs(residual), 8)
    x[:, 30] = prior_mean(np.abs(residual), 32)
    x[:, 31] = lagged(y, 1)
    x[:, 32] = ((records["flags"] >> 1) & 1).astype(np.float32)
    return x, y.astype(np.float32), final_z.astype(np.float64)


def contiguous_tail_sample(data: np.memmap, begin: int, end: int,
                           count: int) -> np.ndarray:
    count = min(count, max(0, end - begin))
    if count == 0:
        return np.empty(0, dtype=RECORD)
    return np.asarray(data[end - count:end])


def fit_linear(x: np.ndarray, y: np.ndarray, z: np.ndarray,
               epochs: int) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    lr = 0.08
    for epoch in range(epochs):
        q = 1.0 / (1.0 + np.exp(-np.clip(z + x @ w, -20.0, 20.0)))
        grad = x.T @ (q - y) / len(y) + 1.0e-5 * w
        w -= lr * grad
        lr *= 0.82
    return w


def evolve(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
           population: int, generations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best = w.copy()
    best_loss = float(bit_loss(z + x @ best, y).sum())
    sigma = 0.08
    for _ in range(generations):
        candidates = best + rng.normal(0.0, sigma,
            size=(population, len(best)))
        candidates[0] = best
        losses = np.empty(population, dtype=np.float64)
        for i, candidate in enumerate(candidates):
            losses[i] = bit_loss(z + x @ candidate, y).sum()
        winner = int(np.argmin(losses))
        if losses[winner] < best_loss:
            best, best_loss = candidates[winner].copy(), float(losses[winner])
        sigma *= 0.90
    return best


def evaluate_range(data: np.memmap, begin: int, end: int,
                   weights: np.ndarray | None = None,
                   context_delta: np.ndarray | None = None,
                   chunk: int = 2_000_000) -> tuple[float, float, float]:
    core_bits = final_bits = candidate_bits = 0.0
    for start in range(begin, end, chunk):
        records = np.asarray(data[start:min(end, start + chunk)])
        (y, bitpos, stream, order, _escape, base_z, final_z,
         _ppmd, _lstm, _fxcm, _match) = unpack(records)
        core_bits += float(bit_loss(base_z, y).sum())
        final_bits += float(bit_loss(final_z, y).sum())
        correction = 0.0
        if weights is not None:
            x, _, _ = features(records)
            correction = x @ weights
        elif context_delta is not None:
            order_band = np.minimum(order // 4, 7)
            context = bitpos + 8 * order_band + 64 * stream
            correction = context_delta[context]
        candidate_bits += float(bit_loss(final_z + correction, y).sum())
    return core_bits, final_bits, candidate_bits


def train_context_grid(sample: np.ndarray) -> np.ndarray:
    (y, bitpos, stream, order, _escape, _base_z, final_z,
     _ppmd, _lstm, _fxcm, _match) = unpack(sample)
    context = bitpos + 8 * np.minimum(order // 4, 7) + 64 * stream
    table = np.zeros((512, len(DELTA_GRID)), dtype=np.float64)
    for column, delta in enumerate(DELTA_GRID):
        table[:, column] = np.bincount(context,
            weights=bit_loss(final_z + delta, y), minlength=512)
    return DELTA_GRID[np.argmin(table, axis=1)]


def oracle_bits(data: np.memmap, begin: int, end: int,
                chunk: int = 2_000_000) -> tuple[float, float]:
    baseline = oracle = 0.0
    for start in range(begin, end, chunk):
        records = np.asarray(data[start:min(end, start + chunk)])
        y, *_middle, final_z, _p, _l, _f, _m = unpack(records)
        losses = np.stack([bit_loss(final_z + d, y) for d in DELTA_GRID])
        baseline += float(bit_loss(final_z, y).sum())
        oracle += float(np.min(losses, axis=0).sum())
    return baseline, oracle


def main() -> int:
    args = parse_args()
    with args.trace.open("rb") as source:
        raw = source.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise SystemExit("truncated FXOT header")
    magic, version, record_size, stream_size, limit_bytes, _ = HEADER.unpack(raw)
    if magic != b"FXOT" or version != 1 or record_size != RECORD.itemsize:
        raise SystemExit("unsupported FXOT trace")
    record_count = (args.trace.stat().st_size - HEADER.size) // RECORD.itemsize
    data = np.memmap(args.trace, mode="r", dtype=RECORD,
                     offset=HEADER.size, shape=(record_count,))
    train_end = record_count * 70 // 100
    validation_begin = record_count * 85 // 100
    train_sample = contiguous_tail_sample(
        data, 0, train_end, args.sample_bits)
    if len(train_sample) < 1024:
        raise SystemExit("FXOT trace is too short for chronological training")
    x, y, z = features(train_sample)
    weights = fit_linear(x, y, z, args.epochs)

    evolution_sample = contiguous_tail_sample(
        data, train_end, validation_begin, args.evolution_bits)
    if len(evolution_sample) != 0:
        ex, ey, ez = features(evolution_sample)
        weights = evolve(weights, ex, ey, ez, args.population,
                         args.generations, args.seed)
    context_delta = train_context_grid(train_sample)

    core, baseline, linear = evaluate_range(
        data, validation_begin, record_count, weights=weights)
    _, _, contextual = evaluate_range(
        data, validation_begin, record_count, context_delta=context_delta)
    oracle_base, oracle = oracle_bits(data, validation_begin, record_count)
    heldout_bytes = max(1.0, (record_count - validation_begin) / 8.0)
    linear_gain_bpb = (baseline - linear) / heldout_bytes
    context_gain_bpb = (baseline - contextual) / heldout_bytes
    oracle_gain_bpb = (oracle_base - oracle) / heldout_bytes

    linear_model_bytes = 16 + 2 * len(weights)
    context_model_bytes = 16 + len(context_delta)
    linear_deployment_bytes = args.code_overhead_bytes + linear_model_bytes
    context_deployment_bytes = args.code_overhead_bytes + context_model_bytes
    linear_projected_total = (
        args.baseline_total -
        linear_gain_bpb * args.entropy_bytes / 8.0 +
        linear_deployment_bytes)
    context_projected_total = (
        args.baseline_total -
        context_gain_bpb * args.entropy_bytes / 8.0 +
        context_deployment_bytes)
    candidates = [
        ("baseline", float(args.baseline_total), 0.0, 0),
        ("linear", linear_projected_total, linear_gain_bpb,
         linear_deployment_bytes),
        ("context_grid", context_projected_total, context_gain_bpb,
         context_deployment_bytes),
    ]
    best_mode, projected_total, best_gain_bpb, deployment_bytes = min(
        candidates, key=lambda item: item[1])
    required_bpb = ((args.baseline_total - args.target_total) * 8.0 /
                    args.entropy_bytes)

    model = {
        "format": "FX4 residual-linear-v2",
        "feature_count": len(weights),
        "weights": [float(v) for v in weights],
        "context_delta": [float(v) for v in context_delta],
        "training_records": int(len(train_sample)),
        "evolution_records": int(len(evolution_sample)),
        "heldout_records": int(record_count - validation_begin),
        "estimated_linear_model_bytes": int(linear_model_bytes),
        "estimated_context_model_bytes": int(context_model_bytes),
        "assumed_code_overhead_bytes": int(args.code_overhead_bytes),
    }
    args.output_model.write_text(
        json.dumps(model, separators=(",", ":")), encoding="ascii")
    serialized_json_bytes = args.output_model.stat().st_size

    report = {
        "stream_size": stream_size,
        "trace_limit_bytes": limit_bytes,
        "records": record_count,
        "heldout_core_bits": core,
        "heldout_cmix_obias_bits": baseline,
        "bitlstm32_heldout_gain_bytes": (core - baseline) / 8.0,
        "noncausal_grid_oracle_gain_bpb": oracle_gain_bpb,
        "noncausal_oracle_can_reach_target": oracle_gain_bpb >= required_bpb,
        "causal_context_grid_gain_bpb": context_gain_bpb,
        "causal_linear_evolved_gain_bpb": linear_gain_bpb,
        "causal_best_fraction_of_oracle": (
            max(0.0, best_gain_bpb) / max(oracle_gain_bpb, 1.0e-12)),
        "required_gain_bpb_for_target": required_bpb,
        "projected_linear_total_bytes": linear_projected_total,
        "projected_context_grid_total_bytes": context_projected_total,
        "projected_full_total_bytes": projected_total,
        "best_deployable_mode": best_mode,
        "best_deployment_bytes": deployment_bytes,
        "target_total_bytes": args.target_total,
        "serialized_diagnostic_json_bytes": serialized_json_bytes,
        "meets_projected_target": projected_total <= args.target_total,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
