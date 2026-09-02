# FX4 tools

This branch ships one production path (`target93`, built by `../build.sh` /
`../build_and_construct_comp.sh`) plus the small set of scripts needed to run
that build on a Google Cloud VM and package it for
[HutterPrizeJudgingAssistant](https://github.com/jabowery/HutterPrizeJudgingAssistant).
Nothing here builds, links, or drives donor discovery, SCR2, mini-cmix,
shadow LSTM-200, or any other research profile -- that code does not exist
on this branch; it lives only on `exp/selective-discovery`.

## Scripts

- `run_google_cloud_hutter.sh ENWIK9 [RUN_DIR] [CPU]` -- copies the built
  `cmix` (S1) and `enwik9` into a fresh run directory, pins execution to one
  CPU core, disables GPU/threaded-math environment variables, and runs the
  full `-e` compression. Writes `archive9`, `archive9.sha256`, and
  `compression.log`. Refuses to reuse an existing run directory.
- `monitor_hutter_run.sh RUN_DIR [INTERVAL_SECONDS]` -- run from a second
  session while compression is in progress. Reports process CPU/elapsed
  time/RSS, post-R1 entropy position, current streamed payload size, a
  linear archive9 size projection, `ppm.temp` size, and host memory. The
  projection is not a score prediction.
- `create_judging_entry.sh ARCHIVE9 OUTPUT_ROOT [ENTRY_NAME]` -- packages a
  completed `archive9` plus a from-scratch source tarball (single top-level
  directory, `install.sh`/`build.sh`/`comp9.args`/complete source, a
  `SOURCE_MANIFEST.sha256`) and `submission/entry.env` into
  `OUTPUT_ROOT/Entries/ENTRY_NAME/`, matching the directory layout
  `HutterPrizeJudgingAssistant`'s `ENTRANT_INSTRUCTIONS.md` requires for a
  self-extracting entry. Validates CR-byte-free scripts and an
  exact-19-byte `comp9.args` before packaging.

## Standalone diagnostic tools (not part of the build)

These compile independently (`clang++ -O2 -std=c++17 tools/NAME.cpp -o NAME`)
and are never invoked by `build.sh`, `build_and_construct_comp.sh`, or the
run scripts above. Useful for inspecting intermediate pipeline state by
hand.

- `byte_vocab.cpp` -- reports the distinct byte-value count (vocabulary
  size) of an arbitrary file, with the present/absent byte ranges in hex.
- `emit_r1_map.cpp` -- given a post-WRT stream, emits the R1 payload_lex
  side data and the recipient/offset map using `src/r1_reorder_transform.h`
  directly.

## What is not here

Donor discovery, SCR2, virtual replay, mini-cmix, shadow LSTM-200,
residual/BitLSTM32/obias heads, token n-gram and delta-memory correction
experts, ALTXS M3/M5 streams, and every script or tool that drove or
measured them were removed from this branch along with `research/`,
`results/`, `run/`, notebooks, and Python training/discovery programs. That
work continues on `exp/selective-discovery`; nothing here can enable it --
the flags and source files it depends on do not exist in this tree's
`makefile` or `src/`.
