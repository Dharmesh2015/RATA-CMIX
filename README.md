# FX4-CMIX Google Cloud Hutter Release

CPU-only enwik9 candidate, built on fx2-cmix-transformer (Vladimer Ivanov,
Kaido Orav, Byron Knoll). This is a technical diff against that baseline; the
checked-in configuration is one deterministic codec with no runtime feature
flags that change its probabilities.

The fx4-cmix work described below is by Dharmesh Patel and Naveen Bijalwan.
Portions of the implementation were developed with assistance from AI coding
tools.

## What this adds to fx2-cmix-transformer

1. **A 2x200 online LSTM expert alongside the frozen transformer.** Two
   stacked 200-cell layers, trained online with backpropagation through
   time over a 128-byte horizon, feeding the outer mixer four channels:

   - the expert's own probability;
   - an expected-byte hint;
   - a disagreement channel between the expert and the transformer;
   - a gate on transformer uncertainty.

   Every one of them is additive. The expert contributes inputs the mixer
   weighs; it never rewrites the mixed prediction.

2. **GrammarMatch** (fx-deepmix, Halvor Yttredal, July 28, 2026), a model
   that predicts two Wikipedia-specific structural patterns directly from
   the post-WRT stream: piped-link labels that extend their target's byte
   image, and in-article title recurrences. It is exported twice -- as a
   mixer channel, and as its own layer-0 mixer context keyed on which
   grammar family fired. The context is the larger half of its value: a
   channel the mixer can weigh differently per family is worth
   substantially more than the same channel weighed uniformly. ECHO-family
   arming is enabled at a one-byte title-match threshold.

3. **v22++, an additive-only enhancement of Kaido Orav's fxcm_v22** (the
   code's own name for it: `v22++592`, `FXCM_V22PP592_NAME`). The v22 core
   -- its 431 original outputs, 12 mixers, `SparseMatchModel`, and the
   `lstmpr`/`lstmex` bridge to the LSTM/transformer stack -- is
   byte-for-byte unchanged; nothing in v22++ writes to v22's own internal
   probability path. Adopting fxcm_v26 wholesale was tried first and
   measured worse than the v22 baseline, traced to v26 disturbing v22's own
   proven internal state; a separate earlier attempt that modified v22's
   existing contexts in place regressed the same way. v22++ is additive
   specifically to avoid that failure mode. Two things ride alongside the
   untouched core:

   - Eight **stationary maps** over byte/symbol/word/codeword-oriented
     contexts, each exporting two complementary predictions to the outer
     mixer -- adapted from the two-signal stationary-map idea in
     fxcm_v26/fx-deepmix, computed and exported independently of v22's own
     state.
   - **90 additional signals** v22's own internals already computed at
     every position but had discarded (suppressed via `prediction_index--`
     in the original), now exported to the outer mixer instead of thrown
     away. This part owes nothing to v26 -- it is v22's own computation,
     merely no longer wasted.

   Both are purely additive exports: the outer mixer is free to learn zero
   weight for either if it does not complement the transformer/LSTM stack,
   and neither can alter v22's own internal predictions.

4. **Disk-backed PPM.** The order-25 model's 14,000 MiB sub-allocator is
   mapped to `ppm.temp` rather than held in anonymous RAM, which is what
   lets an allocator that size run inside the judging memory limit.
   Residency is held down by dropping resident pages on a fixed byte
   cadence with `MADV_DONTNEED`.

   This is output-neutral. Clearing present page-table entries on a
   `MAP_SHARED` mapping changes where bytes are held, not what the model
   reads back. The mapping's address never moves, which matters because
   PPMD stores raw pointers into its own heap -- `pText`, `UnitsStart`,
   `LoUnit`, `HiUnit`, the free lists and every tree node -- so eviction
   must preserve addresses. `MADV_RANDOM` is applied at map time, since
   readahead on a pointer-chased tree fetches pages that are never read,
   and `ppm.temp` is opened `O_NOATIME` to keep repeated page-fault reads
   from generating inode metadata writes.

5. **Recompressed transformer weights.** The shipped model moves to the v5
   tensor container, which entropy-codes the Q4 symbols against per-tensor
   histograms and a causal column-local count derived from the tensor's own
   shape. No side table is stored: the decoder reconstructs that state from
   the shape and the symbols it has already read.

   | File | Bytes |
   | --- | ---: |
   | `6m-q4-fp32.tfwc2` (previous) | 2,930,652 |
   | `6m-q4-fp32.tfwc5` (shipped) | 2,902,452 |
   | Saved | 28,200 |

   This is lossless and it is verified as lossless, not asserted. The v5
   CRC32 covers the compressed bytes rather than the decoded tensors, so it
   proves the file arrived intact and proves nothing about whether the
   decoder reconstructs the right weights -- a decoder whose probability
   model drifted from the encoder's would pass the checksum and hand the
   transformer silently wrong values. The check that does bind is
   behavioural: coding the same 16,384-byte prefix with the previous file
   and with this one produces byte-identical output, archive 1,710 and
   payload 1,673 both ways, with all 572 models loading.

   Recompressed with the tooling in `pysrc/`:

       python -m pysrc.weights_compress decompress 6m-q4-fp32.tfwc2 tmp.bin
       python -m pysrc.weights_compress compress5   tmp.bin 6m-q4-fp32.tfwc5

   Those two commands run in the training environment -- `pysrc/`
   imports torch -- not in the judged build, which never runs Python.
   The C++ loader reads the v5 container and nothing else.

6. **Scr2Match**, SCR2's derived pattern table used as a probability expert
   rather than a stream transform. The transform form was tried and
   rejected: it substitutes patterns with marker and escape bytes drawn from
   values the source alphabet does not use, which changes the post-WRT
   vocabulary the frozen transformer was trained on and is refused outright.
   As a model instead, the stream is untouched: for every pattern it tracks
   how many leading bytes currently match the tail of already-decoded
   history -- one match-length counter per pattern, no history buffer, no
   rescan. When several patterns tied at the longest match agree on the next
   byte, that byte is an expectation and the per-bit probability is a
   learned confidence; on genuine disagreement, or when nothing has matched
   far enough, it stays silent. Exported as an additive mixer channel plus
   its own layer-0 context keyed on the matching pattern, how far it
   matched, and whether several patterns agreed -- the same shape as
   GrammarMatch. Idle emits exactly 0.5.

7. **A zero-side-data morphology specialist** for literal WRT words the
   dictionary has not seen before. A causal hash table keyed on the
   lower-cased trailing bytes of the word in progress learns which byte
   follows a given suffix and how reliably. No side file, no retraining --
   the table starts empty and fills as decoding proceeds. Additive channel
   plus its own mixer context; abstains at probability 0.5 whenever it has
   no live expectation for the current bit.

8. **A zero-side-data causal donor specialist**, a multi-reference episodic
   match expert over already-decoded history. It never replays donor bytes
   into PPMd, the transformer, the LSTM expert, or FXCM -- a rejected
   prediction from it cannot disturb any other model's state, only the
   mixer's own weighting of this one channel. Matches are discovered live
   from decoded bytes; nothing is stored about donor offsets or source
   pages, so there is no side file and no external asset either side of the
   codec needs.

The checked-in configuration is exactly:

    v22p + 521 + T1 + T2 + T2-E + T2-ED + T2-EDG + Stationary
        + GrammarMatch + the grammar mixer context + GM_ARM
        + Scr2Match + MorphologyMatch + CausalDonor

which is 575 mixed models. Every setting above is compiled in at a fixed
value. There is no environment variable and no build flag on this branch that
changes a probability.

See [the architecture guide](docs/ARCHITECTURE.md) for the full model list,
S1/S2 layout, and build details.

## Status

- Branch: release/google-cloud-hutter. Further optimization work continues
  on exp/gch-optimization; its settled baseline is merged here as each piece
  lands, its development history and in-progress experiments are not
- Platform: Linux x86-64, Ubuntu 20.04 (focal)
- Toolchain: clang++-17, LTO (`-flto=thin`), profile-guided (PGO), UPX 5.1.1
- GPU: not used or linked
- Main entropy stream: 586,459,321-byte, 205-symbol post-WRT stream
- PPMd: order 25, 14,000 MiB disk-backed heap (`ppm.temp`)
- Online LSTM expert: 2 layers x 200 cells, BPTT horizon 128
- Mixed models: 575
- Frozen transformer: 6M CPU model, embedded in both S1 and archive9
- Hutter form: self-extracting

target93 is the candidate name and research target. This repository does not
claim a measured 93 MB archive. A complete judged compression and decompression
run is still required before making a score claim.

## Result

Not yet measured. Update after a complete, judged compression and
decompression run against the real enwik9.

| Item | Value |
| --- | ---: |
| Previous record `L` | `TBD` bytes |
| `archive9` | `TBD` bytes |
| `cmix` | `TBD` bytes |
| Total `S = archive9 + cmix` | `TBD` bytes |
| Improvement `1 - S/L` | `TBD` |
| Bytes below previous record | `TBD` |
| Margin above 1% threshold | `TBD` |

## Platform

| Metric | Value |
| --- | --- |
| Machine type | `n4d-highmem-2` (2 vCPU, 16 GiB RAM) |
| OS | Ubuntu 20.04.6 LTS (focal), `ubuntu-2004-focal-v20240731` |
| Storage | GCE persistent disk, 100 GB |
| Geekbench 5 `T` used for timing | `TBD` |

## Run Measurements

Compression run:

| Metric | Value |
| --- | ---: |
| Wall time | `TBD` |
| User + system CPU time | `TBD` |
| Maximum resident set size | `TBD` |
| Exit status | `TBD` |

Verified full decompression run:

| Metric | Value |
| --- | ---: |
| Wall time | `TBD` |
| User + system CPU time | `TBD` |
| Maximum resident set size | `TBD` |
| Exit status | `TBD` |

## Production Pipeline

    enwik9
      -> article reorder
      -> PHDA9
      -> WRT
      -> PPMd + FXCM + frozen transformer + 2x200 online LSTM expert
         + GrammarMatch + Scr2Match + MorphologyMatch + CausalDonor
      -> arithmetic coder
      -> self-extracting archive9

The predictors are mixed together; they are not serial compressors. See
[the architecture guide](docs/ARCHITECTURE.md) for the exact model and archive
layout.

## Google Cloud Quick Start

Use an Ubuntu 20.04 (focal) x86-64 VM with no GPU, at least 16 GiB RAM, and a
local disk with at least 100 GB capacity -- matching the judging image
(`ubuntu-2004-focal-v20240731`, `ubuntu-os-cloud`) and resource limits in
[ENTRANT_INSTRUCTIONS.md](https://github.com/jabowery/HutterPrizeJudgingAssistant/blob/main/ENTRANT_INSTRUCTIONS.md).
For example:

    gcloud compute instances create fast-vm-decomp \
      --zone=us-central1-b \
      --machine-type=n4d-highmem-2 \
      --boot-disk-size=100GB \
      --boot-disk-type=hyperdisk-balanced \
      --image=ubuntu-2004-focal-v20240731 \
      --image-project=ubuntu-os-cloud

Then, on the VM:

    sudo ./install.sh
    ./build_and_construct_comp.sh
    ./tools/run_google_cloud_hutter.sh /data/enwik9 /data/fx4run 0

From a second SSH session:

    ./tools/monitor_hutter_run.sh /data/fx4run 60

The run script pins the codec to one CPU, disables common GPU and threaded math
runtimes, and refuses to reuse an existing run directory.

## How S1 Is Built

`build_and_construct_comp.sh` is the single source of truth for turning this
source tree into `cmix` (S1); `build.sh` runs the same steps inside the
judging harness's offline, read-only-`/entry` sandbox. Both do:

1. `make target93` -- clang++-17, `-flto=thin`, and, when
   `pgo/default.profdata` is present, `-fprofile-use`. Research profiles and
   feature switches live only on `exp/selective-discovery`; this branch
   compiles one fixed configuration.
2. Strip the binary and UPX-pack it (`--ultra-brute`, verified with `upx -t`).
3. Run the packed binary against itself to compress `dictionary/english.dic`
   and the article-order file, then decompress each back and `cmp` against
   the original -- a real reversibility check, not just a build check.
4. Append the compressed dictionary, compressed article order, the frozen
   transformer weights, and a small header after the packed executable. See
   [the architecture guide](docs/ARCHITECTURE.md#s1-layout) for the exact
   byte layout.

### The PGO Profile

`pgo/default.profdata` is generated once, ahead of time, and committed --
`build.sh` runs offline and can't profile a fresh run itself. To regenerate
it:

    make pgo-instrumented
    ./cmix_pgo_instrumented -e prof_input/input   profile_out_1
    ./cmix_pgo_instrumented -e prof_input/input2  profile_out_2
    make pgo-merge
    make target93   # picks up pgo/default.profdata automatically

`prof_input/` holds the exact inputs the committed profile was trained on, so
the profile is reproducible from what's in the source package. A profile
generated with a different clang++-17 build than the one `install.sh`
provisions (for example, a distro-patched package instead of the
`apt.llvm.org` build) can leave some functions' profile data unmatched at
build time -- harmless (LLVM falls back to default heuristics for just that
function, per-function, never a build failure), but for full effect the
profile should be regenerated with the same toolchain `install.sh` installs.

## Verify archive9

Run decompression in a clean directory that does not contain enwik9:

    mkdir /data/fx4decode
    cp /data/fx4run/archive9 /data/fx4decode/
    cd /data/fx4decode
    /usr/bin/time -v taskset -c 0 ./archive9
    cmp /data/enwik9 enwik9_uncompressed
    sha256sum /data/enwik9 enwik9_uncompressed

## Build A Judging Entry

After a successful full round trip:

    ./tools/create_judging_entry.sh \
      /data/fx4run/archive9 /data/fx4-entry FX4

This creates:

    /data/fx4-entry/Entries/FX4/
      entry.env
      archive9
      fx4-cmix-source.tar.gz

The source archive has exactly one top-level directory and contains only the
production C++ codec, required assets (dictionary, transformer weights, the
committed PGO profile and its generation inputs), licenses, documentation and
build inputs -- copied by an explicit allowlist in the script, not a directory
walk, so nothing else in a developer checkout can reach a submission.

## Alpha Judging Assistant

    git clone https://github.com/jabowery/HutterPrizeJudgingAssistant.git
    cd HutterPrizeJudgingAssistant
    cp -a /data/fx4-entry/Entries/FX4 Entries/
    cp /data/enwik9 ./enwik9
    ./judging_assistance.sh \
      --serial \
      --runtime-exec-policy process-tree \
      --work-root /mnt/large-disk/HutterPrizeJudging \
      Entries/FX4 ./enwik9

process-tree is required because S1 and archive9 execute a helper image
extracted from their own already-counted bytes to decode the embedded
dictionary and article order. The complete descendant tree remains within the
judge's resource accounting.

See [the cloud and judging guide](docs/GOOGLE_CLOUD_AND_JUDGING.md) before
starting the multiday run.

## Official References

- [HutterPrizeJudgingAssistant](https://github.com/jabowery/HutterPrizeJudgingAssistant)
- [Entrant instructions](https://github.com/jabowery/HutterPrizeJudgingAssistant/blob/main/ENTRANT_INSTRUCTIONS.md)
- [Hutter Prize detailed rules](https://www.hutter1.net/prize/hrules.htm)
