# cmix-obias integration

This tree keeps the existing FX4 selective post-R1 discovery system and ports
the configuration-of-record architecture published with `cmix-obias`.

## Record core

The default `make record` build enables:

- the 256-cell byte LSTM (`horizon=128`, learning rate `0.03`);
- the coded-bit-only BitLSTM32 terminal correction head;
- the constant output-bias prior gate (`0.15`);
- the imported flat/aligned LSTM and mixer hot paths;
- the existing FX4 LSTM-to-FXCM bridge and 14 GB disk-backed PPM path;
- archive transport for the 23,002-byte BitLSTM32 FP16 model.

The terminal head is called only by `Encoder::Encode` and `Decoder::Decode`.
Raw/known-byte observation paths do not train it. Encoder and decoder use the
same precise-floating-point translation unit.

## Selective discovery

`make selective` adds donor plans, post-R1 experts, mini-cmix, virtual replay,
SCR2/post-R1 transforms and donor discovery. These remain plan-gated; compiling
them does not apply them globally.

## Building later in WSL

Do not omit the model asset from size accounting. A non-PGO record build is:

```bash
bash ./build_obias_record.sh
```

For a PGO build, point `PROFILE_INPUT` at a representative raw sample:

```bash
PROFILE_INPUT=/root/fx4run/sample.in bash ./build_obias_record.sh
```

For the discovery-capable binary:

```bash
BUILD_SELECTIVE=1 bash ./build_obias_record.sh
```

The generated `run/` directory contains `cmix` and `bitlstm32.blob`. Run the
compressor with `KH_BITLSTM32` set to that blob. The `-e` path appends the blob
to `archive9`, and decompression extracts it before decoding dictionary/order
assets.

The optional decoder-first CPU residual candidate is documented in
`RESIDUAL_LSTM96_DECODER_FIRST.md`. It is not enabled in the accepted record
build unless `RESIDUAL_LSTM96_BLOB` is supplied.

## Validation status

The record plus optional LSTM-96 candidate builds successfully with clang-17.
A 4 KiB identity-head test produced the same 793-byte entropy archive with the
head disabled and enabled; decompression restored an identical SHA-256. A
trained LSTM-96, full `-e` round trip, peak RSS, runtime, final S1, archive9 and
combined Hutter score have not yet been measured.

The branchless ContextMap rewrite and wholesale FXCM model-prediction reorder
from the experimental upstream speed work were deliberately not imported.
They alter pointer/state-update structure and model ordering and require an
independent archive-compatibility experiment. Safe prefetch/table/locality
changes are present.
