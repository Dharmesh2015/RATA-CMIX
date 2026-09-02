# FX4-CMIX Google Cloud Hutter Release

CPU-only enwik9 candidate, built on the cmix-lex lineage (Kaido Orav, Byron
Knoll, Ibrahim Marcouch). This is a technical diff against that baseline; the
checked-in configuration is one deterministic codec with no runtime feature
flags that change its probabilities.

## What changed from cmix-lex

1. **Frozen transformer replaces the online byte LSTM on the main stream.**
   A 12-layer, width-192, ~6M-parameter CPU transformer, quantized for AVX2
   int4/int8 inference, predicts the canonical 587,138,826-byte, 205-symbol
   payload_lex/R1 stream in place of cmix-lex's online LSTM. Small embedded
   helper streams (dictionary, article order) keep an online 200-cell LSTM
   instead, since their vocabularies don't fit the frozen model.
2. **GrammarMatch.** A new model that predicts two Wikipedia-specific
   structural patterns directly from the post-WRT stream: piped-link labels
   that extend their target's byte image, and in-article title recurrences.
3. **DeepMix contexts.** Four additional FXCM context maps (article-order
   and secondary-symbol signals) plus a deterministic overflow fix for
   `ContextMap3`'s update (a `U32` counter could wrap during a long one-run
   and invert a confident prediction).
4. **ESN/NLMS correction** and a **contextual specialist corrector** layered
   on top of the mixed prediction.
5. **PPM storage**, unchanged in algorithm from cmix-lex's stable
   `ppm.temp` mmap discipline, retuned to an 8,704 MiB RSS purge trigger.

See [the architecture guide](docs/ARCHITECTURE.md) for the full model list,
S1/S2 layout, and build details.

## Status

- Branch: release/google-cloud-hutter
- Platform: Linux x86-64, Ubuntu 22.04
- GPU: not used or linked
- Main entropy stream: canonical 587,138,826-byte post-R1 stream
- PPMd: order 25, 14,000 MiB file-backed heap
- PPM RSS purge trigger: 8,704 MiB
- Frozen transformer: 6M CPU model, embedded in both S1 and archive9
- Hutter form: self-extracting

target93 is the candidate name and research target. This repository does not
claim a measured 93 MB archive. A complete judged compression and decompression
run is still required before making a score claim.

## Production Pipeline

    enwik9
      -> article reorder
      -> PHDA9
      -> WRT
      -> payload_lex/R1
      -> PPMd + FXCM v26 + frozen transformer + retained CPU experts
      -> arithmetic coder
      -> self-extracting archive9

The predictors are mixed together; they are not serial compressors. See
[the architecture guide](docs/ARCHITECTURE.md) for the exact model and archive
layout.

## Google Cloud Quick Start

Use an Ubuntu 22.04 x86-64 VM with no GPU, at least 16 GiB RAM, and a local
SSD volume with at least 150 GB capacity.

    sudo ./install.sh
    ./build_and_construct_comp.sh
    ./tools/run_google_cloud_hutter.sh /data/enwik9 /data/fx4run 0

From a second SSH session:

    ./tools/monitor_hutter_run.sh /data/fx4run 60

The run script pins the codec to one CPU, disables common GPU and threaded math
runtimes, and refuses to reuse an existing run directory.

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
production C++ codec, required assets, licenses, documentation and build
inputs.

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
