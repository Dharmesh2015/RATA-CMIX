#!/usr/bin/env python3
"""Train the CPU-only decoder-first FXRL96 residual model from an FXOT trace.

The model is baseline anchored: all correction outputs start at zero. Training
uses chronological splits and reset-aligned 256-byte sequences. The exported
blob is row-int8/fp16 and the final report scores that quantized model, including
the fact that a Hutter model asset is normally paid once in S1 and once in S2.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
from pathlib import Path

import numpy as np

from residual_lstm96_blob import (CONFIDENCE, CONTEXTS, HIDDEN, INPUT,
                                  LINEAR, OUTPUT, dequantize_rows,
                                  inspect_blob, write_blob)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "train_residual_lstm96.py needs CPU PyTorch; the final C++ codec does not"
    ) from error


HEADER = struct.Struct("<4sHHQQQ")
# v1: 10 bytes, no transformer6m field (record_size == RECORD_V1.itemsize).
# v2: adds a trailing transformer_logit byte (0xff = no active transformer
# for that bit -- small S1 helper streams keep the online LSTM only); written
# whenever the tree that produced the trace was built with FX4_TRANSFORMER6M.
RECORD_V1 = np.dtype([
    ("base_p", "<u2"), ("final_p", "<u2"),
    ("ppmd", "u1"), ("lstm", "u1"), ("fxcm", "u1"),
    ("flags", "u1"), ("ppm_meta", "u1"), ("match", "u1"),
], align=False)
RECORD_V2 = np.dtype(RECORD_V1.descr + [("transformer", "u1")], align=False)
LN2 = math.log(2.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output_blob", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--reset-bytes", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--valid-batches", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=923)
    parser.add_argument("--correction-limit", type=float, default=2.0)
    parser.add_argument("--baseline-total", type=int, default=108_492_825)
    parser.add_argument("--target-total", type=int, default=100_000_000)
    parser.add_argument("--entropy-bytes", type=int, default=587_138_826)
    parser.add_argument("--code-overhead-bytes", type=int, default=8192)
    parser.add_argument("--asset-copies", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def probability_logit(values: np.ndarray) -> np.ndarray:
    probability = np.clip(values.astype(np.float32),
                          1.0 / 65536.0, 65535.0 / 65536.0)
    return np.log(probability) - np.log1p(-probability)


def dequantized_logit(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) * (16.0 / 255.0) - 8.0


def stream_batch(data: np.memmap, starts: np.ndarray,
                 sequence_bytes: int) -> np.ndarray:
    records = []
    count = (sequence_bytes + 1) * 8
    for start in starts:
        bit_start = int(start) * 8
        records.append(np.asarray(data[bit_start:bit_start + count]))
    return np.stack(records, axis=0).reshape(len(starts), sequence_bytes + 1, 8)


def tensor_features(records: np.ndarray) -> tuple[torch.Tensor, ...]:
    y = (records["flags"] & 1).astype(np.float32)
    bit_position = ((records["flags"] >> 2) & 7).astype(np.int64)
    stream = ((records["flags"] >> 5) & 7).astype(np.int64)
    order = (records["ppm_meta"] >> 3).astype(np.float32)
    escape = (records["ppm_meta"] & 7).astype(np.float32)
    base_z = probability_logit(records["base_p"].astype(np.float32) / 65536.0)
    final_z = probability_logit(records["final_p"].astype(np.float32) / 65536.0)
    ppmd_z = dequantized_logit(records["ppmd"])
    lstm_z = dequantized_logit(records["lstm"])
    fxcm_z = dequantized_logit(records["fxcm"])
    match = records["match"].astype(np.float32)
    override = ((records["flags"] >> 1) & 1).astype(np.float32)

    source = slice(0, -1)
    target = slice(1, None)
    source_y = y[:, source]
    source_final_z = final_z[:, source]
    source_p = 1.0 / (1.0 + np.exp(-np.clip(source_final_z, -20.0, 20.0)))
    byte_input = np.zeros((*source_y.shape[:2], INPUT), dtype=np.float32)
    byte_input[:, :, 0] = 1.0
    byte_input[:, :, 1:9] = source_y * 2.0 - 1.0
    byte_input[:, :, 9:17] = np.clip(source_final_z / 8.0, -1.0, 1.0)
    byte_input[:, :, 17:25] = (source_y - source_p) * 4.0
    byte_input[:, :, 25] = np.clip(
        np.mean(ppmd_z[:, source] - source_final_z, axis=2) / 6.0, -1.0, 1.0)
    byte_input[:, :, 26] = np.clip(
        np.mean(lstm_z[:, source] - source_final_z, axis=2) / 6.0, -1.0, 1.0)
    byte_input[:, :, 27] = np.clip(
        np.mean(fxcm_z[:, source] - source_final_z, axis=2) / 6.0, -1.0, 1.0)
    byte_input[:, :, 28] = np.mean(order[:, source], axis=2) / 31.0
    byte_input[:, :, 29] = np.mean(escape[:, source], axis=2) / 7.0
    byte_input[:, :, 30] = (
        np.log1p(np.max(match[:, source], axis=2)) / math.log(256.0))
    byte_input[:, :, 31] = stream[:, source, 0].astype(np.float32) / 7.0

    tz = final_z[:, target]
    tb = base_z[:, target]
    tp = ppmd_z[:, target]
    tl = lstm_z[:, target]
    tf = fxcm_z[:, target]
    torder = order[:, target]
    tescape = escape[:, target]
    tmatch = match[:, target]
    current = np.empty((*tz.shape, LINEAR), dtype=np.float32)
    current[..., 0] = np.clip((tp - tz) / 6.0, -1.0, 1.0)
    current[..., 1] = np.clip((tl - tz) / 6.0, -1.0, 1.0)
    current[..., 2] = np.clip((tf - tz) / 6.0, -1.0, 1.0)
    current[..., 3] = torder / 31.0
    current[..., 4] = tescape / 7.0
    current[..., 5] = np.log1p(tmatch) / math.log(256.0)
    stacked = np.stack((tp, tl, tf, tz), axis=-1)
    current[..., 6] = np.minimum(1.0, np.std(stacked, axis=-1) / 4.0)
    current[..., 7] = np.clip((tz - tb) / 4.0, -1.0, 1.0)

    target_stream = stream[:, target]
    context = (bit_position[:, target] +
               8 * np.minimum((torder / 4).astype(np.int64), 7) +
               64 * target_stream).astype(np.int64)
    confidence = np.digitize(np.abs(tz),
        np.asarray([0.5, 1.0, 2.0, 3.0, 4.0, 6.0], dtype=np.float32))
    tensors = (byte_input, current, context, confidence, tz,
               y[:, target], override[:, target])
    return tuple(torch.from_numpy(np.ascontiguousarray(value))
                 for value in tensors)


class ResidualLstm96(nn.Module):
    def __init__(self, correction_limit: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(INPUT, HIDDEN, batch_first=True)
        self.output = nn.Linear(HIDDEN, OUTPUT)
        self.linear = nn.Parameter(torch.zeros(OUTPUT, LINEAR))
        self.context = nn.Parameter(torch.zeros(CONTEXTS))
        self.gate = nn.Parameter(torch.ones(CONFIDENCE))
        self.correction_limit = correction_limit
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, byte_input: torch.Tensor, current: torch.Tensor,
                context: torch.Tensor, confidence: torch.Tensor,
                override: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(byte_input)
        recurrent_delta = self.output(recurrent)
        linear_delta = torch.sum(current * self.linear[None, None, :, :], dim=-1)
        gate = torch.clamp(self.gate, 0.0, 2.0)[confidence]
        correction = gate * (recurrent_delta + linear_delta) + self.context[context]
        correction = torch.clamp(correction,
                                 -self.correction_limit, self.correction_limit)
        return correction * (1.0 - override)


def loss_bits(logit: torch.Tensor, bit: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(
        logit, bit, reduction="sum") / LN2


def evaluate(model: ResidualLstm96, data: np.memmap, starts: np.ndarray,
             sequence_bytes: int) -> tuple[float, float, int]:
    baseline = candidate = 0.0
    examples = 0
    model.eval()
    with torch.no_grad():
        for start in starts:
            values = tensor_features(stream_batch(
                data, np.asarray([start]), sequence_bytes))
            byte_input, current, context, confidence, final_z, bit, override = values
            correction = model(byte_input, current, context, confidence, override)
            baseline += float(loss_bits(final_z, bit))
            candidate += float(loss_bits(final_z + correction, bit))
            examples += sequence_bytes
    return baseline, candidate, examples


def quantized_copy(model: ResidualLstm96) -> ResidualLstm96:
    result = copy.deepcopy(model).cpu()
    with torch.no_grad():
        result.lstm.weight_ih_l0.copy_(torch.from_numpy(dequantize_rows(
            model.lstm.weight_ih_l0.detach().cpu().numpy())))
        result.lstm.weight_hh_l0.copy_(torch.from_numpy(dequantize_rows(
            model.lstm.weight_hh_l0.detach().cpu().numpy())))
        combined_bias = (model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0)
        combined_bias = combined_bias.detach().cpu().numpy().astype("<f2").astype(np.float32)
        result.lstm.bias_ih_l0.copy_(torch.from_numpy(combined_bias))
        result.lstm.bias_hh_l0.zero_()
        result.output.weight.copy_(torch.from_numpy(dequantize_rows(
            model.output.weight.detach().cpu().numpy())))
        result.output.bias.copy_(torch.from_numpy(
            model.output.bias.detach().cpu().numpy().astype("<f2").astype(np.float32)))
        result.linear.copy_(torch.from_numpy(dequantize_rows(
            model.linear.detach().cpu().numpy())))
        result.context.copy_(torch.from_numpy(
            model.context.detach().cpu().numpy().astype("<f2").astype(np.float32)))
        result.gate.copy_(torch.from_numpy(
            torch.clamp(model.gate, 0.0, 2.0).detach().cpu().numpy()
            .astype("<f2").astype(np.float32)))
    return result


def export(model: ResidualLstm96, path: Path, args: argparse.Namespace) -> int:
    model = model.cpu()
    with torch.no_grad():
        return write_blob(
            path,
            wih=model.lstm.weight_ih_l0.numpy(),
            whh=model.lstm.weight_hh_l0.numpy(),
            bias=(model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0).numpy(),
            wout=model.output.weight.numpy(),
            bout=model.output.bias.numpy(),
            wlinear=model.linear.numpy(),
            context_delta=model.context.numpy(),
            confidence_gate=torch.clamp(model.gate, 0.0, 2.0).numpy(),
            correction_limit=args.correction_limit,
            reset_bytes=args.reset_bytes,
        )


def main() -> int:
    args = arguments()
    if args.reset_bytes <= 0:
        raise SystemExit("--reset-bytes must be positive")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with args.trace.open("rb") as source:
        raw_header = source.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise SystemExit("truncated FXOT header")
    magic, version, record_size, stream_size, limit_bytes, declared = \
        HEADER.unpack(raw_header)
    if magic != b"FXOT" or version not in (1, 2):
        raise SystemExit("unsupported FXOT trace")
    record_dtype = RECORD_V2 if version == 2 else RECORD_V1
    if record_size != record_dtype.itemsize:
        raise SystemExit("unsupported FXOT trace")
    record_count = (args.trace.stat().st_size - HEADER.size) // record_dtype.itemsize
    if declared and declared != record_count:
        raise SystemExit("FXOT declared record count does not match the file")
    byte_count = record_count // 8
    if byte_count < args.reset_bytes * 20:
        raise SystemExit("trace is too short for chronological LSTM-96 training")
    data = np.memmap(args.trace, mode="r", dtype=record_dtype,
                     offset=HEADER.size, shape=(record_count,))
    train_end = byte_count * 70 // 100
    validation_begin = byte_count * 85 // 100
    train_blocks = (train_end - args.reset_bytes - 1) // args.reset_bytes
    if train_blocks <= 0:
        raise SystemExit("no reset-aligned training blocks")

    model = ResidualLstm96(args.correction_limit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    model.train()
    for step in range(args.steps):
        starts = rng.integers(0, train_blocks, size=args.batch,
                              endpoint=False) * args.reset_bytes
        values = tensor_features(stream_batch(data, starts, args.reset_bytes))
        byte_input, current, context, confidence, final_z, bit, override = values
        correction = model(byte_input, current, context, confidence, override)
        loss = functional.binary_cross_entropy_with_logits(
            final_z + correction, bit)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            model.gate.clamp_(0.0, 2.0)
        if (step + 1) % 100 == 0:
            print(f"step={step + 1} loss_bits={float(loss) / LN2:.8f}",
                  flush=True)

    first_valid = ((validation_begin + args.reset_bytes - 1) //
                   args.reset_bytes) * args.reset_bytes
    available = max(0, (byte_count - first_valid - args.reset_bytes - 1) //
                    args.reset_bytes)
    valid_count = min(args.valid_batches, available)
    if valid_count == 0:
        raise SystemExit("no untouched reset-aligned validation blocks")
    starts = first_valid + np.arange(valid_count, dtype=np.int64) * args.reset_bytes
    float_base, float_candidate, examples = evaluate(
        model, data, starts, args.reset_bytes)
    quantized = quantized_copy(model)
    quant_base, quant_candidate, _ = evaluate(
        quantized, data, starts, args.reset_bytes)
    if abs(float_base - quant_base) > 1.0e-3:
        raise SystemExit("internal baseline mismatch during quantized evaluation")
    model_bytes = export(quantized, args.output_blob, args)
    blob_info = inspect_blob(args.output_blob)

    gain_bpb = (quant_base - quant_candidate) / examples
    projected_payload_saving = gain_bpb * args.entropy_bytes / 8.0
    deployment_bytes = args.code_overhead_bytes + args.asset_copies * model_bytes
    projected_total = args.baseline_total - projected_payload_saving + deployment_bytes
    required_gain_bpb = ((args.baseline_total - args.target_total +
                          deployment_bytes) * 8.0 / args.entropy_bytes)
    report = {
        "format": "FXRL96-v1",
        "trace_stream_bytes": int(stream_size),
        "trace_limit_bytes": int(limit_bytes),
        "training_steps": args.steps,
        "validation_bytes": examples,
        "float_validation_gain_bytes": (float_base - float_candidate) / 8.0,
        "quantized_validation_gain_bytes": (quant_base - quant_candidate) / 8.0,
        "quantized_gain_bpb": gain_bpb,
        "model_bytes": model_bytes,
        "asset_copies_in_score": args.asset_copies,
        "assumed_code_overhead_bytes": args.code_overhead_bytes,
        "deployment_bytes": deployment_bytes,
        "projected_payload_saving_bytes": projected_payload_saving,
        "projected_total_bytes": projected_total,
        "baseline_total_bytes": args.baseline_total,
        "target_total_bytes": args.target_total,
        "required_gain_bpb_including_deployment": required_gain_bpb,
        "net_positive_vs_baseline": projected_total < args.baseline_total,
        "projected_target_met": projected_total <= args.target_total,
        "blob": blob_info,
    }
    report_path = args.report or args.output_blob.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                           encoding="ascii")
    if args.checkpoint:
        torch.save({"state_dict": model.state_dict(), "report": report},
                   args.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
