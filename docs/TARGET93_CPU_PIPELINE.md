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
  -> pretrained CPU transformer6m (replaces the online byte LSTM)
  -> FXCM v26 and the existing context/match models
  -> contextual mixers and SSE
  -> BitLSTM32 terminal correction
  -> arithmetic coder
```

Decompression performs the exact inverse. M3 and M5 side data is entropy-coded
inside the product stream. The transformer and BitLSTM32 assets are embedded in
both S1 and S2, because the compressor and standalone decompressor both need
the same files.

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

Build the optional n-gram ablation with:

```bat
wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && TOKEN_NGRAM=1 ./build_and_construct_comp.sh"
```

It is not the default. Keep it only when its archive saving exceeds its S1
growth and runtime cost.

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

The 2,930,652-byte transformer and 23,002-byte BitLSTM model are present once
in S1 and once in S2. A smaller entropy payload is not a Hutter improvement if
those model costs erase the gain. If `93 MB` means total Hutter score, the S2
payload must be correspondingly below 93 MB.

## Current limitation

The supplied transformer weights were trained for the earlier fx2 transformed
stream, not the final M3+M5 stream. They are valid deterministic weights and
the integration is reversible, but they are not expected to realize the full
potential of M3+M5 without retraining.

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
2. Transformer on the old stream without M3+M5.
3. M3+M5 plus the supplied transformer.
4. M3+M5 plus a transformer retrained on that stream.
5. Candidate 4 plus `TOKEN_NGRAM=1`.

Only candidate 4 can fairly test whether the combined architecture approaches
the target. Donor, SCR2, mini-cmix, shadow LSTM, CNN, URL, and knot specialists
remain discovery tools. None belongs in S1 until a warm full-prefix trial saves
more archive bytes than complete plan/model/binary overhead.

