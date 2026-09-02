# FX4-CMIX Google Cloud Hutter Release

This branch is the production-only, CPU-only FX4-CMIX candidate for enwik9.
It removes the donor discovery engine, SCR2, URL/post-R1 portfolio experts,
shadow LSTM-200, mini-cmix, residual traces, alternate M3/M5 streams and their
archive formats.

The checked-in configuration is one deterministic codec. There are no runtime
feature flags that change its probabilities.

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
inputs. Historical research/, results/, run/, notebooks, Python files,
discovery scripts and transformer benchmarks are excluded.

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
