# Target93 CPU pipeline

`target93` is the consolidated CPU-only research path. The 93 MB figure is a
measurement target, not a result claimed by this source tree. A result is
accepted only after complete `-e` compression, standalone archive extraction,
SHA-256 equality with enwik9, and Hutter score accounting.

## Data and model path

Compression runs the following deterministic sequence:

```text
enwik9
  -> article order
  -> PHDA9
  -> altxs M3 PHDA9 densification
  -> WRT dictionary transform
  -> altxs M5 payload_sim order and sealed inverse data
  -> PPMd order 25 (14 GB file-backed arena)
  -> accepted online byte LSTM-256
  -> FXCM v26 and the existing context/match models
  -> contextual mixers and SSE
  -> BitLSTM32 terminal correction
  -> arithmetic coder
```

Decompression performs the exact inverse. M3 and M5 side data is entropy-coded
inside the product stream. BitLSTM32 is embedded in both S1 and S2 because the
compressor and standalone decompressor both need the same file.

No CUDA, cuBLAS, cuDNN, PyTorch, or GPU code is linked into this target.

## Build one self-contained S1

From Windows CMD:

```bat
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && chmod +x build_and_construct_comp.sh && ./build_and_construct_comp.sh"
```

The output is:

```text
fx4-cmix/run/cmix
```

That single file contains the packed core, compressed dictionary, compressed
article order, transformer model, BitLSTM32 model, and versioned header.

Build optional trace-screened ablations with:

```bat
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && TOKEN_NGRAM=1 ./build_and_construct_comp.sh"
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && DELTA_MEMORY=1 ./build_and_construct_comp.sh"
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && TRANSFORMER_EXPERIMENT=1 ./build_and_construct_comp.sh"
```

They are not defaults. Keep one only when its archive saving exceeds its S1
growth, S2 model copy, and runtime cost.

## Full Hutter run

After building S1:

```bat
powercfg /setactive SCHEME_MIN && wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/tools/run_hutter_ext4.sh /mnt/d/mywork/myideas/latestcompressor/enwik9 /mnt/d/mywork/myideas/latestcompressor/archive9
```

The script copies enwik9 and S1 to native WSL ext4 under `/root/fx4run`, pins
the single-core run, creates `archive9`, and exports the archive and timing log.
It does not use the GPU.

## Score gate

Always report both values:

```text
S1 = run/cmix bytes
S2 = archive9 bytes
S  = S1 + S2
```

The BitLSTM model is present once in S1 and once in S2. The optional
2,930,652-byte transformer is also paid twice when enabled. A smaller entropy
payload is not a Hutter improvement if model/code costs erase the gain. If
`93 MB` means total Hutter score, S2 must be correspondingly below 93 MB.

## Measured limitations

The supplied transformer was trained on the exact 205-byte vocabulary of the
pre-R1 WRT stream. The final M3+M5 product has a 256-byte vocabulary because
its small sealed inverse stream can contain every byte. An earlier integration
silently skipped the transformer on the real stream while still packaging its
2,930,652-byte model. That is now fixed:

* `target93` is the lean M3/M5 baseline and does not package the transformer.
* `target93_transformer` is an explicit ablation.
* The ablation maps the main 205-byte vocabulary to the frozen model and uses
  PPMd passthrough plus a deterministic model reset for M5-only bytes.
* FXOT traces mark passthrough bytes inactive, so activation is measurable.

The existing 5 MiB FXOT-v2 trace contains 40,000,000 real coded-bit records.
It found zero transformer activations in the old build. It also screened the
current tiny online experts:

```text
DeltaMemory8:  -1,033 bytes over the traced 5 MiB
Token n-gram:    +191 bytes full sample, +3.2 bytes chronological holdout
```

DeltaMemory8 is rejected. The token n-gram stays experimental because its
holdout gain does not yet pay executable growth.

A dense float16 PPM prior trace would require roughly:

```text
576 million tokens * 205 probabilities * 2 bytes = about 236 GB
```

That exceeds the requested 100 GB workspace. Retraining should therefore use
one of these bounded workflows:

1. Stream PPM distributions through a FIFO directly into the trainer.
2. Store only top-k probabilities plus a residual-mass value per token.
3. Train on deterministic article shards and delete each prior shard after its
   optimizer checkpoint is committed.

Do not train on future bytes or ship an uncounted model. The final model asset
must be included in both S1 and S2.

## Discovery policy

Continue discovery, but only after one complete target93 baseline exists. Run
these exact ablations before resuming donor or page-level searches:

1. M3+M5 with the accepted online LSTM and no transformer.
2. Transformer on the old 205-byte stream without M3+M5.
3. M3+M5 plus the vocabulary-adapted supplied transformer.
4. M3+M5 plus a transformer retrained on that stream.
5. Candidate 4 plus any expert that first wins on chronological FXOT holdout.

Only candidate 4 can fairly test whether the combined architecture approaches
the target. Donor, SCR2, mini-cmix, shadow LSTM, CNN, URL, and knot specialists
remain discovery tools. None belongs in S1 until a warm full-prefix trial saves
more archive bytes than complete plan/model/binary overhead.

The reference gap from 108,492,825 bytes to 93,000,000 bytes is 15,492,825
bytes. No current trace-screened side expert closes a material fraction of that
gap; 93 MB remains a research target, not a result.
