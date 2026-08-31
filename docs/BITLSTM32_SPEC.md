# KH_BITLSTM32 — per-coded-bit LSTM32 correction head: exact specification

Model: `shuffle_v2_lstm32_seq64_all_no_xml` (continuous variant), 11,489
parameters. Trained by `tools/head_refit/continuous_shuffled_lstm.py` on the res_v3
per-bit residual trace of the golden run; features built by
`tools/head_refit/full_stream_head.py` (`FeatureState.build` + `scan_bytes`) via
`tools/head_refit/shuffled_blocks.py`. This document is the C++ implementation
contract for `src/models/bitlstm32-head.cpp`; parity between the two was
verified end-to-end (see RESULTS.md).

## Where the head sits in the pipeline

Per coded bit, in `Encoder::Encode` / `Decoder::Decode` (and ONLY there — the
res_v3 trace recorded exactly these bits; `EncodeRawBit` / `ObserveKnownBit`
never touch the head):

```
base_p = Discretize(p_->Predict())        // u16, 1..65535 (the trace's final_p)
p_coded = head.Adjust(base_p, stage1, 25, m1raw, override)   // if enabled
... arithmetic-code the bit with p_coded ...
head.Observe(bit)                          // advance causal state
```

- `t = ln(p1/(1-p1))`, `p1 = base_p / 65536` (the training-time clamp to
  `[1e-6, 1-1e-6]` can never bind since `base_p >= 1`).
- The head output `delta` is **logit-additive**: `p_new = sigmoid(t + delta)`,
  re-discretized with the coder's own formula `(unsigned)(1 + 65534*p_new)`,
  clamped to `[1, 65535]`.
- On byte-mixer override bits (`Predict()` returned exactly 0/1;
  trace flag bit1; ~0.02%..6% of bits depending on stream position) the
  correction is **not applied** (`p_coded = base_p`): training masked these
  bits' loss (`w=0`), so the head is unconstrained there. Their features and
  their effect on all causal state ARE still processed (training did the same).
- All features derive from `base_p` (never from `p_coded`), so the feature
  stream is identical to the trace the head was trained on; enabling the head
  changes only what the coder codes with.
- Everything the head consumes is available at decode time: the stage-1 row,
  m1raw and base_p come from the same `Predict()` call the decoder makes, and
  every windowed feature uses only already-decoded bits/bytes.

## Network

```
x    : f32[92]  features (below)
xin  : f32[93] = concat(x, previous_bit)   # previous coded bit; 0.5 for bit 0
h0   = SiLU(W_in xin + b_in)               # W_in [32,93]
h,c  = LSTMCell_32(h0; h,c)                # PyTorch gate order i,f,g,o
delta = W_out h + b_out                    # [1,32]
logit = t + delta
```

Recurrent state (`h`, `c`) is **zeroed whenever `bit_index % 64 == 0`**
(bit_index = count of coded bits so far). This reproduces training exactly:
seq64 TBPTT ran independent 64-bit sequences with fresh zero state, and every
sequence start fell on a global multiple of 64 (blocks are 2^20 bits, aligned).
`previous_bit` is NOT reset at sequence starts — training fed the true
previous stream bit there (0.5 only at stream start).

Weight blob: `tools/export_bitlstm32_blob.py` (header `KHBL32\x01\0`,
nin/hid/seq_reset, then f32: in_proj.weight [32][93], in_proj.bias,
weight_ih_l0 [128][32], weight_hh_l0 [128][32], bias_ih_l0, bias_hh_l0,
out.weight [32], out.bias; 45,980 bytes). Env `KH_BITLSTM32=<blob>` enables;
unset ⇒ byte-identical output to golden-256 (gate-proven).

## The 92 features (exact order)

Notation: `i` = current coded-bit index (0-based), all windows cover bits
`j < i` only (strictly causal), `den = max(min(i, W), 1)`, T = 12.203.

`E[25]` — the "usable expert" row: stage-1 mixer inputs `layers_[1].Inputs()`
indices 0..22 (the 23 mixer_0 outputs) and 24 (byte-mixer/LSTM stretched
input), plus `m1raw` (raw `mixer_1_[0].Mix()` pre-Logistic). The dead fxcm
slot 23 is dropped. **Each value is round-tripped through IEEE f16
(round-to-nearest-even)** — the trace stored f16 and training saw that
quantization — then `nan→0, +inf→T, -inf→-T`.

| idx | feature | definition |
|---|---|---|
| 0–24 | experts | `E[0..24]` as above |
| 25–32 | bitpos | one-hot of `i % 8` (bit position in byte, 0 = MSB) |
| 33 | tl | `t / T` |
| 34 | tl² | `(t/T)^2` |
| 35 | sign(t) | −1/0/+1 (0 occurs iff base_p == 32768) |
| 36–40 | resm | trailing mean of `res_j = bit_j − p1_j` over W ∈ {32, 256, 2048, 16384, 131072}, × 4 |
| 41–46 | surm | trailing mean of `sur_j = −log2(max(P_j(actual bit), 1e−6))` over W ∈ {16, 64, 512, 4096, 65536, 1048576} |
| 47 | regime₁ | `(surm[64] − surm[4096]) × 4` |
| 48 | regime₂ | `(surm[512] − surm[1048576]) × 4` |
| 49 | shock clock | `log1p(i − last_shock)/14`; `last_shock` = latest `j<i` with `sur_j > 4`, initial value 0 |
| 50 | qmean | trailing mean of `1[sur_j > 4]` over W=256, × 8 |
| 51 | sd | `(t − E[24]) / T` |
| 52 | sdabs | trailing mean of `abs(t_j − E_j[24])` over W=256, / T |
| 53–59 | erel | for experts k ∈ {0,5,11,17,22,23,24}: trailing mean over W=512 of `−log2(bit_j ? pe : 1−pe)`, `pe = clamp(sigmoid(E_j[k]), 1e−6, 1−1e−6)` |
| 60 | mean(E) | over the 25 values |
| 61 | std(E) | **unbiased** (divisor 24) |
| 62 | range | `max(E) − min(E)` |
| 63 | sign agreement | fraction of k with `sign(E[k]) == sign(t)` |
| 64–68 | sorted E | ranks 0, 6, 12, 18, 24 of ascending sort |
| 69–74 | xml | **always 0** (feature_set all_no_xml zeroes the 6 XML columns) |
| 75–82 | ctx8 | previous 8 completed coded-stream bytes / 255, oldest→newest, zero-padded before 8 bytes exist; constant across the 8 bits of a byte |
| 83–87 | zone one-hot | `bytepos = i/8` (f32) bucketized against byte boundaries {541126651, 554726452, 571499539, 586459321} (torch right=False: `bp ≤ b[z]`) |
| 88 | zone fraction | `(bp − zlo[z]) / (zhi[z] − zlo[z])`, bounds as f32 from {0, …, 587138826} |
| 89 | progress | `i / 4697110608` (f32, clamped ≤ 1) |
| 90–91 | phase | `sin/cos(2π·10·progress)` |

Zone/progress constants are the enwik9 coded-stream geometry (587,138,826
post-preprocessor bytes). On other inputs the features remain computable
(zone 0, small progress) but are off the training distribution.

## Numerics / determinism

- Window sums use double accumulators with subtract-oldest ring buffers
  (training used f32 cumsum on GPU; the difference is far below the f16 input
  quantization already present, and parity was verified: 96.8% of bits produce
  the identical discretized probability, 99.9% within ±1/65536, aggregate
  entropy delta 0.008 bits over the first 2^20 bits of the golden-256 trace).
- Gate activations use an 8-lane Cephes expf (fixed coefficients/order);
  feature-side transcendentals are scalar libm.
- The head TU is compiled `-ffp-model=precise` with `noinline` entry points, so
  encode and decode execute identical machine code — determinism across the
  two directions does not depend on fast-math or inlining context.
- All state is explicitly initialized; no uninitialized reads.

## Training provenance

- Old trace (170-cell golden, enc.2074020.0.res): 10 epochs, fresh corpus-wide
  1-Mbit block permutation each epoch, within-block 64-bit-sequence shuffle,
  AdamW lr 1e-3→1e-4 hyperbolic, grad-clip 1.0, loss masked on override bits.
  Online net-KB curve ended at 527.8 KB; frozen re-evaluation on 12 seeded
  blocks: 0.498% of cmix bits = 530.2 KB scaled (measured on the OLD trace).
- Refit (this branch): `tools/head_refit/refit_continuous_lstm32.py` — same
  protocol on the golden-256 trace (enc.1566312.0.res), init from the old
  checkpoint, lr 5e-4, plus a true held-out set (block_index % 25 == 0 never
  trained, scored frozen). Numbers in RESULTS.md.
