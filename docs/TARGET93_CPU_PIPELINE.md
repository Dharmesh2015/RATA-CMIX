# Target93 connected CPU pipeline

`target93` is one connected research build, not a collection of profiles.
The 93 MB value is a target and is not a result claimed by this tree.

## Frozen modeled stream

The build deliberately preserves cmix-lex's canonical modeled stream:

```text
enwik9
  -> cmix-lex article order
  -> PHDA9/WRT
  -> payload_lex/R1
  -> 587,138,826-byte, 205-symbol modeled stream
```

Consequences:

* Existing post-R1 group boundaries remain valid.
* Existing donor offsets still name the same bytes, although no donor action
  is accepted unless a later warm full-prefix trial pays all side data.
* M3/M5, IDHOIST, SCR2 substitution, PMD1 and other byte-changing transforms
  are not in this build. Enabling one invalidates every stored group offset,
  donor plan and probability trace.

## Connected predictor

```text
canonical post-R1 stream
  -> PPMd order 25, 14 GB stable file-backed arena
  -> FXCM v26 plus DeepMix SCMA/ORDP/state-map fixes
  -> DeepMix GrammarMatch causal channel
  -> frozen 6M FX2 transformer replacing the online byte LSTM
  -> contextual mixers and SSE
  -> tiny online ESN + normalized-LMS residual expert
  -> baseline/expert likelihood gate
  -> arithmetic coder
```

The transformer is a replacement, matching the supplied FX2 architecture.
It is not added beside LSTM-170/200. Its weights were trained on this exact
205-symbol stream, so no retraining is needed for this candidate. Small helper
streams that cannot run the transformer retain the original 1x200 online LSTM.

The ESN/NLMS expert has eight reservoir states and eight adaptive readout
weights. It has no model file or side data. Encoder and decoder recreate it
causally. An exact Bayesian two-expert mixture starts with 99% weight on the
accepted predictor and resets only that gate at deterministic 1 MiB logical
boundaries. Reservoir and NLMS state remain continuous. The resulting
worst-case redundancy is about one byte over the canonical stream, before
finite-precision rounding. It uses no per-bit `exp()` or `log()`.

No CUDA, cuBLAS, cuDNN, PyTorch or GPU code is linked or executed.

## Intentionally excluded

* **M3/M5:** changes the stream to a 256-symbol product and makes the supplied
  transformer and all canonical group metadata incompatible.
* **cmix-obias/BitLSTM32:** trained around the online LSTM distribution, not
  around the replacement transformer.
* **DeltaMemory8:** the existing 5 MiB trace measured a 1,033-byte loss.
* **Token n-gram:** only about 3.2 bytes of chronological holdout gain, below
  code-size cost.
* **Donor/SCR2/mini-cmix discovery:** no warm full-prefix winner has paid its
  complete plan and binary overhead. Discovery must be off for a Hutter run.
* **Archive-to-archive CNN/delta:** independently arithmetic-coded 109 MB and
  93 MB files avalanche after a probability difference. Model the common
  pre-coder sequence or probabilities instead.

## Build S1

From Windows CMD:

```bat
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && chmod +x build_and_construct_comp.sh && ./build_and_construct_comp.sh"
```

Output:

```text
fx4-cmix/run/cmix
```

S1 embeds the packed C++ executable, compressed dictionary, compressed article
order, frozen transformer and header. No Python program, discovery plan or
external runtime model is needed.

## Before a full run

`target93` is the default make target. In that build the inactive donor,
virtual-replay, SCR2 and post-R1 transform implementations are not linked, and
their per-byte checks are compiled out.

Do not use a raw 1 MiB or raw 20 MiB `-c` sample as the acceptance gate for
this architecture. The frozen transformer is meaningful only after the exact
article-order, WRT and R1 pipeline has produced the canonical 205-symbol
stream. When testing is authorized, use the same `-e` path intended for the
full submission and treat an early 20 MiB projection only as a screening
estimate.

Then require:

1. transformer activation is nonzero;
2. compression and decompression use identical model files;
3. exact round trip;
4. S1 plus projected S2 improves the current accepted score;
5. peak RSS remains below 10 GB on the official accounting path.

The bounded projection is a screening estimate only. A Hutter claim requires a
complete `-e` run and standalone extraction of the resulting `archive9`.

## Disk and training

No new transformer training is required for this frozen-stream candidate.
Runtime disk budget is approximately:

```text
14.0 GB  PPM arena
 1.0 GB  enwik9
 0.6 GB  canonical post-R1 temporary stream
 1-3 GB  archive/helper/copy headroom
```

Use at least 25 GB free native WSL ext4 space; 30 GB is safer. A 21 GB volume
can fit but leaves little room for duplicate temporary files and filesystem
metadata.

Retraining becomes mandatory if the modeled stream changes. The supplied FX2
recipe says its training-data extraction needs roughly 300 GB and training was
performed for about 30 hours on eight RTX 5090 GPUs. That training cost is not
part of Hutter runtime, but the final model is charged in both S1 and S2.

## Realistic target

The supplied FX2 transformer reports a 96,996,198-byte payload family and CPU
inference within its stated 50-hour single-core budget. That is the defensible
first target for this connected build. Reaching 93 MB total requires a further
multi-megabyte payload gain after paying the model twice; DeepMix contexts and
the ESN/NLMS expert are hypotheses, not evidence of that gain.
