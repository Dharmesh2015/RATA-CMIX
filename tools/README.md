# FX4 tool status

The connected `target93` build has one production path.

Supported production tools:

* `../build_and_construct_comp.sh` builds and packages the self-contained S1.
* `run_hutter_ext4.sh` runs exact `-e` compression on native WSL ext4.
* `package_accepted_ext4.sh` packages an already accepted core.
* `check_hutter_progress.sh` observes a running full-file job.

`validate_1mb_ext4.sh` is retained only as a legacy raw-helper regression. It
does not exercise the frozen transformer and cannot accept or reject
`target93`.

Other shell, CMD and C++ tools in this directory are historical research
artifacts. They are not compiled, packaged or called by the production path.
The Python discovery and transformer-training programs were removed from this
branch; the frozen transformer model is used directly and no training runtime
is part of S1.

## Target93 selective first-20 test

`run_target93_selective20_ext4.sh` is the bounded, resumable donor/SCR2
gate for the current transformer architecture. It rebuilds the exact
target93 predictor in its discovery forks, tests the first 20 grouped
post-R1 recipients from a warm prefix, and never uses Python.

The three phases are:

1. donor singletons and arithmetic-cost SCR2 virtual replay;
2. exact donor bundles of depth 2 through 7;
3. combinations formed from measured phase-1/phase-2 candidates.

Every trial is charged its complete standalone archive-plan bytes. The final
summary also subtracts the stripped-core growth of the smallest production
feature set needed by the selected winners. No discovery engine is linked
into that production core.

```bash
bash tools/run_target93_selective20_ext4.sh /path/to/enwik9
```

Durable output is written to `results/target93_selective20`. Rerunning the
same command resumes incomplete ledgers and reuses completed phases.

## Full target93 selective run

The full ext4 script performs the complete CPU-only workflow:

1. builds and packages the canonical target93 core and discovery core;
2. resumes all three warm post-R1 discovery phases;
3. builds compact F4CP donor/mini-cmix plans and F4VR SCR2 plans in C++;
4. charges plan bytes and actual packaged S1 growth;
5. selects baseline, F4CP, F4VR, or their combination;
6. runs -e on native WSL ext4 and verifies an exact round trip by default.

The submitted compressor does not contain the discovery engine. Selected
plans are embedded in archive9, and decompression needs no external plan.
The 14,000 MiB PPMd heap is file-backed; the default RSS purge trigger is
8,704 MiB and can be changed with FX4_PPM_RSS_MB.

From Windows CMD:

    tools\run_target93_selective_full.cmd

Optional explicit paths:

    tools\run_target93_selective_full.cmd ..\enwik9 ..\archive9_target93

In a second Windows CMD window:

    tools\monitor_target93_selective.cmd 300

The monitor reports discovery progress and, during the final entropy pass,
live BPB, projected payload, projected archive9, projected Hutter S1 + S2,
RAM, and active-time ETA. Projections below 2% are marked unstable.
