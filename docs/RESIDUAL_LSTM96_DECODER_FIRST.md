# Decoder-first Residual LSTM-96

## Purpose

The 100 MB target cannot be produced by writing a decompressor for an archive
that does not yet exist. The useful reverse order is instead:

1. Freeze the decoder's model format and state transition.
2. Prove an identity model is reversible and archive-neutral.
3. Train a compressor-side candidate for that exact decoder.
4. Reject it unless chronological held-out saving pays all Hutter costs.
5. Only then run the expensive full control and candidate compressions.

This implementation contains no LLM and no GPU path. Runtime inference is C++
on one CPU core. The offline trainer explicitly selects CPU PyTorch.

## Implemented model

`FXRL96-v1` is applied after the accepted BitLSTM32 terminal correction:

```text
cmix-obias probability
  -> accepted BitLSTM32
  -> byte-step LSTM-96 correction
  -> arithmetic coder
```

The LSTM advances once per completed byte and emits eight corrections for the
next byte. That is much cheaper than a per-bit LSTM-96. A small current-bit
linear head uses PPMd, LSTM, FXCM, match, order, escape, disagreement and the
existing BitLSTM32 correction. A 512-entry context table and seven confidence
gates provide the contextual ensemble and no-change behavior.

The model is baseline anchored: recurrent output, linear output and context
table start at zero. A zero correction returns the incoming integer probability
without a logit round trip. Override bits remain unchanged.

The blob uses row-wise symmetric int8 weights, fp16 scales/tables, a fixed
256-byte state reset and CRC-32. The current identity asset is 53,406 bytes.
It is counted once with S1 and once inside S2, in addition to code growth.

## Phase A: decoder parity first

From WSL:

```bash
cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix-discovery
python3 tools/make_residual_lstm96_identity_blob.py \
  models/residual_lstm96_identity.blob
make clean
make CC=clang++-17 record RESIDUAL_LSTM96=1
export KH_BITLSTM32="$PWD/models/bitlstm32/refit_golden256_fp16.blob"
export KH_RESIDUAL_LSTM96="$PWD/models/residual_lstm96_identity.blob"
```

Compress and decompress a small representative input with the identity model.
The restored SHA-256 must match. Its entropy payload must also match the same
binary with `KH_RESIDUAL_LSTM96` unset, because all corrections are zero.

## Phase B: trace the accepted 108 MB-class model

Do not trace a donor/SCR2/virtual-replay experiment. Trace the continuous
accepted cmix-obias path:

```bash
make clean
make CC=clang++-17 record ORACLE_TRACE=1
export KH_BITLSTM32="$PWD/models/bitlstm32/refit_golden256_fp16.blob"
export FX4_ORACLE_TRACE=/root/fx4run/obias_100mb.fxot
export FX4_ORACLE_TRACE_BYTES=100000000
export FX4_STOP_AFTER_ORACLE_TRACE=1
./cmix -e /root/fx4run/enwik9 /root/fx4run/oracle_probe
```

This cleanly stops after 100,000,000 logical post-R1 bytes. The trace is about
8 GB because it stores ten bytes for each of eight coded bits.

## Phase C: train and gate LSTM-96

```bash
python3 tools/train_residual_lstm96.py \
  /root/fx4run/obias_100mb.fxot \
  /root/fx4run/residual_lstm96.blob \
  --threads 1 \
  --steps 3000 \
  --baseline-total 108492825 \
  --target-total 100000000 \
  --code-overhead-bytes 8192
```

Training uses the first 70% chronologically, leaves 70-85% for future tuning,
and reports on untouched reset-aligned blocks after 85%. The report is for the
quantized model, not the larger float training model.

Keep the model only when all are true:

```text
quantized_validation_gain_bytes > 0
net_positive_vs_baseline = true
small exact roundtrip passes
measured payload saving > 2 * model bytes + executable growth
```

`projected_target_met` is deliberately strict. Reaching 100,000,000 from
108,492,825 requires about 0.115718 bpb before deployment cost. The LSTM-96
must earn more than that after its asset is counted twice; otherwise it can be
a useful step toward 108 MB or 106 MB, but it is not the complete 100 MB path.

## Phase D: build the candidate S1

```bash
RESIDUAL_LSTM96_BLOB=/root/fx4run/residual_lstm96.blob \
  bash ./build_obias_record.sh
```

The script prints the packed core, wrapper, BitLSTM32 asset, Residual LSTM-96
asset and total S1 size. During `-e`, the LSTM-96 blob is appended to archive9;
the self-extracting decoder verifies and loads it before decoding payload bits.

Only after a representative exact test beats the score gate should two full
multi-day Hutter runs be made: one untouched cmix-obias control and one LSTM-96
candidate.
