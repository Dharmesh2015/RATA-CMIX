# FX4 Source-Diverse Donor Handoff

## Objective

Find net-positive post-R1 donor profiles outside accepted source group G388.
Discovery must not replay donor bytes into PPMd, LSTM, FXCM, match models, or
the main mixer.

## Accepted State

- Keep canonical `C421-D388-7` as a separate regression result.
- The novel-source campaign excludes G388 and defers grouped probes G421/G422.
- Donor bytes initialize only the page/group-scoped probability specialist.
- The specialist and its small mixer reset at each selected span; base model
  state continues exactly as in the warm-prefix baseline.
- A proxy score is never a win. Retain only exact warm-prefix payload savings
  that exceed compact F4CP metadata.

## Prepared Search Space

- 531 page-aligned groups cover all 587,138,826 post-R1 bytes.
- 523 recipients have at least one non-G388 causal candidate.
- 347 distinct new source groups are represented.
- 6,671 candidate windows are source-labelled.
- The first eight candidates for every recipient with eight available sources
  come from eight distinct source groups.

Current proxy source hubs begin with G40, G136, G2, G16, G115, G13, G1,
G246, G44 and G39. They require exact testing.

## Commands

Regenerate the cheap source-diverse campaign after ranker changes:

```bat
prepare_diverse_donor_campaign.cmd
```

Run or resume 100 exact breadth trials over the first 64 priority recipients:

```bat
run_claude_diverse_donor_multiday.cmd 100 7 64
```

Increase the third argument only after each tier is exhausted:

```bat
run_claude_diverse_donor_multiday.cmd 100 7 128
run_claude_diverse_donor_multiday.cmd 100 7 256
run_claude_diverse_donor_multiday.cmd 100 7 523
```

The same command resumes durable CSV ledgers. Do not delete the result folder.

## Breadth Policy

Each recipient tests ordered source-diverse prefixes of 1, 2, 4 and 8 donors.
This crosses the whole stream before expensive length/local/beam refinement.
Leave-one-out marginal attribution runs only after a bundle wins.

After breadth discovery, build a separate deep campaign from:

1. Net-positive selected bundles.
2. Positive single donors.
3. Near-loss bundles within 64 bytes.
4. Sources with positive leave-one-out marginal gain.

Then run length refinement and beam depth up to eight only for those recipients.

## Result Files

- `results/diverse_group_campaign/donor_source_rankings.csv`
- `results/diverse_group_campaign/donor_window_rankings.csv`
- `results/diverse_group_campaign/candidate_source_map.csv`
- `results/claude_donor_grouped_diverse/page.winner_trials.csv`
- `results/claude_donor_grouped_diverse/page.winner_selected.csv`
- `results/claude_donor_grouped_diverse/exact_source_evidence.csv`
- `results/claude_donor_grouped_diverse/exact_positive_bundles.csv`

`bundle_net_gain_involved` is diagnostic and must not be summed across source
groups because one bundle appears in every source represented by that bundle.
