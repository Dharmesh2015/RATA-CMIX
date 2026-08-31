# Residual oracle workflow

This workflow measures whether a small CPU-only causal correction can improve
the cmix-obias probability enough to justify integration. It does not try to
recompress the already-random archive, and it never uses the current or a future
bit as a deployable feature.

## Execution order

Do not begin with two full multiday runs. Use these gates:

1. Build the accepted cmix-obias record configuration and pass a small exact
   roundtrip.
2. Generate a 100 MB chronological trace from a normal raw-enwik9 `-e` run.
3. Measure the noncausal correction ceiling and causal held-out gain.
4. Integrate a residual head only when its projected saving exceeds executable
   and model growth.
5. Reproduce the full cmix-obias control, then run the full candidate.

This establishes the 108 MB architecture before claiming an improvement while
avoiding a full run for a candidate that cannot approach the target.

## Generate a 100 MB trace

Run from native WSL ext4:

```bash
make clean
make record ORACLE_TRACE=1
export KH_BITLSTM32="$PWD/models/bitlstm32/refit_golden256_fp16.blob"
export FX4_ORACLE_TRACE=/root/fx4run/obias_100mb.fxot
export FX4_ORACLE_TRACE_BYTES=100000000
export FX4_STOP_AFTER_ORACLE_TRACE=1
./cmix -e /root/fx4run/enwik9 /root/fx4run/oracle_probe
```

The command begins with raw enwik9 and follows the normal article-order,
PHDA9/WRT and R1 pipeline. The trace begins at the canonical post-R1 arithmetic
coding stream; small dictionary/order helper runs are ignored by the default
100 MB minimum.

After 100,000,000 logical post-R1 bytes, the encoder closes the trace, records
its exact record count, removes the intentionally incomplete payload, and exits
before Hutter archive packaging. Do not combine this diagnostic stop with SCR2,
virtual replay, donor plans or another size-changing post-R1 mode.

Each 10-byte record contains exact pre/post-BitLSTM32 probabilities, quantized
PPMd/LSTM/FXCM logits, the actual bit, bit position, stream class, PPM order,
escape depth and match length. A 100 MB prefix therefore produces an
approximately 8 GB trace.

## Analyze on CPU

```bash
python3 tools/optimize_residual_oracle.py /root/fx4run/obias_100mb.fxot \
  --baseline-total 108492825 \
  --target-total 100000000 \
  --code-overhead-bytes 4096 \
  --output-model /root/fx4run/residual_oracle_model.json
```

The analyzer reports:

- The existing BitLSTM32 contribution.
- A noncausal per-bit correction-grid ceiling.
- A causal context table.
- A baseline-anchored causal linear correction using PPMd/LSTM/FXCM
  disagreement, prior residuals, recent error, confidence and causal stream
  state.
- Chronological held-out gain and a full-file projection after compact model
  bytes plus the supplied code-overhead estimate.

The first 70% trains the correction, the next 15% refines continuous parameters,
and the final 15% remains untouched for evaluation. The noncausal oracle is only
an upper bound.

## Acceptance gates

Moving 108,492,825 to 100,000,000 using the 587,138,826-byte entropy stream
requires about 0.116 bpb of net held-out gain.

- If the noncausal grid oracle is below the required gain, this correction
  family cannot reach 100 MB.
- If the oracle has ample headroom but causal models capture little of it, test
  one compact decoder-first Residual-LSTM96 experiment. See
  `RESIDUAL_LSTM96_DECODER_FIRST.md`.
- If a causal model projects below the control after executable/model cost,
  integrate it in deterministic C++ and run a small exact roundtrip.
- Only then spend the time on full control and candidate Hutter runs.

FineZip's useful lesson here is probability calibration and online adaptation.
Its 8B Llama runtime, GPU memory and multiday 10 MB arithmetic-coding path are
not suitable for the Hutter constraints.
