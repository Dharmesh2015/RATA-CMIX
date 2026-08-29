# Winner-First Selective Discovery

This isolated branch searches for post-R1 improvements without changing the
accepted warm PPMd, LSTM-170, FXCM v26, mixer, or SSE state.

## Discovery S1 versus scored S1

The last packaged discovery executable is 514,907 bytes. It deliberately
contains the search engine and every candidate expert so one resumable run can
measure them. It is not a Hutter submission candidate.

The accepted production S1 remains 441,024 bytes. The 73,883-byte difference is
discovery machinery, not archive debt that a selected model must recover. After
discovery, rebuild without `DONOR_DISCOVERY` and compile out every unselected
expert before comparing `S1 + S2`.

A matched selective release-core compile measured the knot implementation alone:

```text
                         topology off   topology on   delta
raw stripped core             456,656       468,944  12,288
UPX --ultra-brute core         176,132       178,868   2,736
```

Therefore topology may enter the scored S1 only if its aggregate archive saving
exceeds 2,736 bytes plus the exact compact plan cost. Otherwise `TOPOLOGY=0`
removes its code and tables at compile time.

## Selection contract

For every grouped recipient, discovery executes three complete warm runs from
raw 1,000,000,000-byte enwik9:

1. **Individual:** test donor singletons, each of the 11 mini-cmix models,
   local shadow LSTM-200, URL specialist, SCR2 virtual replay, topology
   recurrence, and the causal CNN separately.
2. **Donor beam:** rank up to 24 donor atoms using all exact singleton
   outcomes, including near-losses, then exactly code ordered 2-7 bundles.
   This preserves interactions where individually weak donors win together.
3. **Combine:** beam-search exact combinations from actions with positive gross
   payload gain. Every combined candidate still pays its complete side data.

There are no predefined cross-expert combinations. Each trial records a
conservative standalone plan charge, while Phase 3 may amortize a reusable
profile inside an exact same-recipient combination. Rejected branches never
alter the warm primary predictor.

Each exact branch owns private expert state and a cloned count-only arithmetic
encoder. Rejected branches never modify the accepted predictor. At the end of a
recipient, the result is either the baseline or exactly one winning
combination.

## SCR2 virtual replay

SCR2 is not applied as a global byte transform. The original six trials replayed
every matching occurrence above a fixed pattern length. Through recipient 250
that produced 174,713 bytes of gross payload saving, but required 1,628,320
bytes of standalone event metadata. It produced no accepted net winner.

Discovery now runs one cost-positive SCR2 trial per recipient. It records the
exact warm predictor probability for every bit, scores all overlapping SCR2
matches, performs metadata-aware weighted interval and pattern pruning, and
then exact-codes only the retained events. Original bytes are still observed by
PPMd, LSTM, FXCM, and every normal model, so subsequent probabilities remain
identical to the baseline.

Selected event offsets and pattern IDs are written durably to
search.individual.portfolio_vr_events.csv. The final plan builder validates
those events against the exact post-R1 stream and embeds the normal F4VR plan.
Patterns are shared across selected recipients and the final combined side cost
is recalculated before acceptance.

## Last durable result

The interrupted individual run completed recipients 0 through 250. After each
standalone candidate paid its complete side data, accepted winners were:

    donor          0
    mini-cmix      0
    shadow LSTM200 0
    URL            0
    topology       0
    causal CNN     0
    old SCR2       0

Topology, causal CNN, LSTM-200, mini-cmix, and URL are therefore retained in the
research source but disabled in the default resume. The default launcher catches
up only the new cost-positive SCR2 trial, including already completed regions.
Set FX4_WINNER_SCR2_COST_ONLY=0 only when deliberately rerunning the legacy
donor and combination phases.

## Donor search

Phase 1 tests causal donor singletons at strengths 0.25, 0.50, 0.75, and 1.00.
Phase 2 uses only positive singleton donors, searches ordered bundles up to
seven donors, and measures every retained bundle with exact arithmetic coding.
Donor effects are not assumed additive.

The bounded search defaults are:

```text
donors per bundle: 7
donor atom pool:    24
beam width:         8
exact candidates:   48 per recipient
combination depth:  8
```

They can be changed with `FX4_WINNER_PLANNED_DONORS`,
`FX4_WINNER_PORTFOLIO_DONOR_ATOMS`,
`FX4_WINNER_PORTFOLIO_BEAM_WIDTH`,
`FX4_WINNER_PORTFOLIO_MAX_CANDIDATES`, and
`FX4_WINNER_PORTFOLIO_MAX_DEPTH`.

## Run or resume

From Windows CMD:

```bat
D:\mywork\myideas\latestcompressor\fx4-cmix-discovery\run_selective_discovery.cmd
```

The run is foreground, single-core, WSL-ext4, and does not use the GPU. Rerun
the same command after interruption. Completed recipient rows are durable; the
interrupted phase rebuilds the exact warm prefix and resumes at the next
unfinished recipient.

Monitor from a second Windows CMD:

```bat
D:\mywork\myideas\latestcompressor\fx4-cmix-discovery\monitor_selective_discovery.cmd
```

## Durable outputs

The default directory is:

```text
results/selective_discovery_winner_first
```

Each phase has its own exact trial and selected ledgers:

```text
search.individual.portfolio_trials.csv
search.individual.portfolio_selected.csv
search.donor_beam.portfolio_trials.csv
search.donor_beam.portfolio_selected.csv
search.combine.portfolio_trials.csv
search.combine.portfolio_selected.csv
```

Final artifacts are:

```text
current_winners.f4cp
current_winners.f4vr
current_winners.f4cp.selected.csv
current_winners.f4cp.summary.json
SHA256SUMS
```

The F4CP plan carries donor assignments and selected probability-expert spans.
The F4VR plan carries selected SCR2 replay events. Both are embedded into the
compressed archive during a production run, so decompression needs no external
plan file.

## Acceptance accounting

Every individual trial records:

```text
candidate payload + complete standalone side data
```

This is a conservative screening score. Gross payload winners remain available
to the final shared-cost optimizer even when they lose their standalone plan.
The optimizer reads every durable `portfolio_trials.csv`, groups identical
donor/expert profiles, and tests the compact F4CP cost after each reusable
profile is charged once. A selected combination must still beat its warm
baseline after complete side data. The final builder shares donor profiles,
expert profiles, and SCR2 patterns, recalculates compact F4CP plus F4VR bytes,
and reports:

- gross payload gain
- combined compact plan cost
- net archive gain
- S1 growth
- projected Hutter `S1 + S2` gain

The summary also gives an optimistic lower bound for every SCR2 family. F4VR-v1
needs at least one byte for the event gap and one byte for the pattern ID per
event, before its header or pattern table. If gross payload gain is below that
floor, dictionary sharing or smaller varints cannot turn the family into a
winner; the occurrence representation itself must change.

These are discovery projections. A Hutter result is proven only after one full
selected-plan compression, archive-embedded plan accounting, exact full
decompression and SHA-256 match, peak RSS below 10 GB, and a lower total
`S1 + S2` than the cmix-lex reference.

## Historical G5 evidence status

The old cumulative campaign reported `855,173 -> 854,810` bytes for G0-G5 with
donor `4195584:1024`. Keep the `854,810` archive as a historical artifact, but
do not count the claimed 363-byte saving as an accepted result:

- the 855,173-byte no-plan baseline archive was not preserved;
- the ledger says that baseline was reused from an earlier G3 run instead of
  being encoded by the same binary as the G5 candidate;
- preserved G5 control, fixed, mini, and virtual-replay calibration archives
  all have the same 854,810-byte size and identical SHA-256, even with those
  actions disabled;
- the current exact in-process warm replay of the same donor gives 118,039
  baseline bytes and 118,039 candidate bytes, then loses 25 bytes of side data.

Therefore the reproducible result for that donor is currently zero gross gain,
not 363 bytes. Reopening G5 requires a same-source, same-binary, same-prefix A/B
pair with only the embedded donor plan changed.

## Experimental topology recurrence expert

`kTopologyRecurrence` is a post-R1 probability expert inspired by Gauss words
and chord interlacement. It never transforms or reorders bytes. Phase A now
tests it independently at four strength settings for every grouped recipient.
Existing durable recipients are revisited with topology-only catch-up trials,
so their donor, mini-cmix, LSTM-200, URL, and SCR2 work is not repeated.

The bounded causal state implements the useful low-overhead parts of the knot
proposal:

- canonical A-B-A recurrence patterns over the last 16 token events
- distance, nesting, disjoint, and interlacement contexts
- a topology-copy prediction from the previous matching recurrence state
- gating by topology confidence and FXCM disagreement

The tables are rebuilt from already decoded bytes and are reset only for the
selected recipient span. No graph, knot code, donor bytes, or learned table is
serialized. F4CP stores one reusable expert profile plus compact selected spans.
Reidemeister rewriting and braid-word transforms are intentionally excluded:
they need reversible transformation data and have no measured benefit yet.

The previous region-0 prototype lost 198 bytes after side data. The completed
expert is therefore discovery-only and compile-gated. It survives into a final
release only when aggregate exact savings exceed its compact-plan bytes and its
measured 2,736-byte packed-core cost.

## Experimental causal CNN expert

The CPU-CNN proposal is implemented as a deliberately small post-R1 correction
expert, not as a replacement for PPMd, LSTM-170, FXCM, or the accepted mixer.
It has eight integer channels, six causal dilated layers evaluated once per
byte, twelve retained features, and eight online-trained bit-position heads.
The convolution is deterministic fixed arithmetic; only the tiny heads adapt,
once per 64 bytes.

Each selected recipient resets its private CNN state. Zero-initialized heads
exactly reproduce the warm baseline until training finds a correction, and the
existing causal regret gate suppresses corrections without sufficient measured
gain. Rejected trials cannot mutate any primary predictor state.

The retired TinySSM F4CP bit is reused, so the mask width and plan format do not
grow. No weights, activations, checkpoint, or disk file are serialized. A
selected recipient pays only the existing shared expert-profile and span
records. Disk may hold discovery ledgers or resumable research checkpoints, but
a final decoder reconstructs the CNN causally from decoded bytes.

A matched release-core build with the discovery engine excluded measured:

```text
                         causal CNN off   causal CNN on   delta
UPX --ultra-brute core           178,868         180,472   1,604
```

`FX4_CAUSAL_CNN=0` removes the implementation. It may enter the scored S1 only
when aggregate exact archive savings exceed 1,604 bytes plus the final compact
span metadata. Existing durable individual rows receive CNN-only catch-up
trials; donor, mini-cmix, LSTM-200, URL, SCR2, and topology trials are not
repeated.

## Memory-safe winner-first resume (2026-08-24)

The stopped run at recipient 98 was an RSS failure, not a bad ledger. The
discovery build still selected the legacy fork-search PPM mode, which used a
private 14 GB mapping and disabled the normal cmix-lex MADV_DONTNEED
residency cadence. Winner-first search is in-process and does not fork
predictors, so it now selects the shared disk-backed mapping at runtime and
retains the 64 KiB eviction cadence. Legacy fork search still receives
MAP_PRIVATE.

Additional output-neutral discovery reductions:

- count-only range-coder clones retain the interval and byte count, not a copy
  of the complete archive buffer
- post-R1 count, DMC, URL, and adaptive-mixer tables allocate only when their
  mask needs them
- direct mini-cmix and shadow-LSTM trials no longer update unused generic gain
  tables
- SCR2-only trials do not construct an inactive specialist
- the four byte-identical URL gain trials are represented by one trial
- glibc uses one arena and trims freed branch pages after each recipient

Exact warm region-0 validation:

    canonical post-R1 region:       0..1,048,219
    baseline payload:               104,950 bytes
    shared candidate rows checked:  53
    candidate mismatches:           0
    trials after duplicate removal: 54
    maximum RSS:                    8,468,836 KiB (8.08 GiB)
    swaps:                          0
    major page faults:              13,220
    wall time including S1 setup:   20:36.26
    exit status:                    0 (clean region-limit pause)

The failed campaign had reached about 13.4 million KiB RSS with no swap.
Recipients 0 through 98 remain durable and compatible; the next normal run
replays them and starts new trials at recipient 99.

Verified discovery S1:

    S1 bytes:   514,907
    SHA-256:    6036696c445d22953184d4141232444a560f9957904b4fe302e592df3a5b1d09
    dictionary: 100,514 bytes, exact decode
    order:      199,145 bytes, exact decode
