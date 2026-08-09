# FX4 Selective Donor Handoff

## Protected incumbent

The canonical post-R1 region-421 result is the acceptance floor, not a disposable experiment:

| Field | Value |
|---|---:|
| Recipient offset | 441,450,496 |
| Recipient length | 1,048,576 |
| Baseline archive | 198,857 bytes |
| Seven-donor archive | 197,309 bytes |
| Compact F4CP v7 plan | 25 bytes |
| Accounted total | **197,334 bytes** |
| Hard ceiling | 197,360 bytes |

Never replace this path unless the candidate archive plus all side data is below 197,334 bytes. Never accept a result above 197,360 bytes.

The exact acceptance manifest, hashes, plan, and donor list are in `results/accepted_region421_197360/`.

## Accepted donor bundle

All offsets are in the canonical 587,138,826-byte post-R1 stream. Replay order matters.

| Order | Donor offset | Length |
|---:|---:|---:|
| 0 | 407,573,504 | 4,096 |
| 1 | 407,581,696 | 2,048 |
| 2 | 407,775,744 | 2,048 |
| 3 | 407,787,520 | 4,096 |
| 4 | 407,793,664 | 4,096 |
| 5 | 407,799,808 | 4,096 |
| 6 | 407,810,048 | 2,048 |

The seven donors total 22,528 bytes. They are reconstructed from already decoded post-R1 offsets; the bytes are not embedded.

## Production architecture

Use three strictly selective layers:

1. **Exact donor replay** may be used only when full-prefix validation shows a net win after its compact v7 edge metadata and all downstream state effects. Replay mutates PPMD, LSTM, FXCM, ByteMixer, and mixer adaptation, so an isolated recipient win is not sufficient production evidence.
2. **Page-scoped donor probability specialist** may consume donor-derived probability features without mutating the main predictor. It is the preferred way to retain a local page win without damaging following probabilities.
3. **Selective 11-model mini-cmix mask** may enable any subset of the complementary direct, indirect, word, and match predictors per span. Profiles are shared across pages and retained only when aggregate savings exceed the shared profile definition, assignment metadata, and executable growth.

Outside selected spans, optional corrections must be exactly neutral. Nothing is enabled globally.

The F4CP v7 codec delta-codes 256-byte-aligned offsets, power-of-two lengths, shared profiles, span references, and the 11-bit mini-model mask. Always use its exact encoded size; do not use the obsolete `7 + 7N` estimate.

## Multiday search

From Windows CMD:

```bat
cd /d D:\mywork\myideas\latestcompressor\fx4-cmix
run_claude_donor_multiday.cmd 25 7
```

`25` is the number of new exact trials in this batch and `7` is the pinned WSL CPU. Increase the first value after confirming stable runtime. Run the identical command again to resume. Completed rows are durable.

The search covers all 559 complete MiB of the canonical post-R1 stream and uses page boundaries. It retains positive, near-positive, and losing trials for later replacement/addition searches. It explores single donors, local length/offset refinements, ordered combinations, and beam-search bundles up to eight donors.

Durable outputs are written under `results/claude_donor_multiday/`:

```text
page.winner_trials.csv
page.winner_positive.csv
page.winner_near.csv
page.winner_selected.csv
page.winner_marginals.csv
cost_accounting.csv
discovery.log
```

The immutable seed files are under `results/donor_multiday_seed/`. Do not delete or regenerate them during a resumed search.

## Acceptance rules

1. Preserve every trial row, including losses and near-losses within 64 bytes; a losing donor can become useful in an ordered bundle.
2. Rank by actual archive change minus exact F4CP v7 metadata, not gross recipient saving.
3. Re-evaluate ordered combinations because donor effects are not additive.
4. Run leave-one-out marginal checks before retaining a bundle.
5. A profile reused by multiple recipients pays its definition once; assignments still pay their encoded cost.
6. Keep a local probability-only win even when exact replay harms downstream state.
7. Promote exact replay only after full-prefix validation. Require a complete roundtrip only for promoted production plans, not discovery trials.
8. Count executable growth for the final Hutter score. The research-only specialist and 11-model paths are not production wins until total full-file archive savings exceed their binary cost.

The immediate objective is to discover more region-421-like bundles while keeping `197,334` intact. The longer-term production plan can combine independent selective winners; it must never trade away an accepted local gain merely to simplify the next region.
