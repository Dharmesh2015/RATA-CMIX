# Claude cumulative donor/SCR2 handoff

Date: 2026-08-12

## Objective

Find additional page-aligned post-R1 donor/recipient wins without giving back
earlier wins, then validate the cumulative plan through the normal Hutter
pipeline.

The campaign is sequential:

```text
G3 candidate  -> test prefix G0..G3 (4 grouped recipients)
G4 candidate  -> test prefix G0..G4 (5 grouped recipients)
...
G421 candidate -> test prefix G0..G421 (422 grouped recipients)
```

For every new group, the reference is the best accepted cumulative plan from
all earlier groups. It is never the stock baseline again after a prior action
has won.

## Authoritative files

```text
Repository:
D:\mywork\myideas\latestcompressor\fx4-cmix

Canonical post-R1 stream:
/root/fx4crawler_98090/enwik9.post_r1.bin
size:   587,138,826 bytes
SHA256: 7826ff63dedd526c119dda08e6e044be8fa8f6e89a55f3d6b1f3447cdfc5c1ce

Grouped recipients:
results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv
results/donor_multiday_seed/enwik9.grouped_1mib_boundaries.f4gb

Existing donor rankings:
results/claude_donor_grouped_full/all_group_edges.csv
results/selective_group_campaign/enwik9.grouped_selective_candidates.csv
results/diverse_group_campaign/enwik9.grouped_diverse_candidates.csv

Campaign state:
results/cumulative_donor_campaign/campaign_state.json
results/cumulative_donor_campaign/trial_ledger.csv
results/cumulative_donor_campaign/accepted_actions.csv
```

## Current exact evidence

### Profile-only isolated wins

The donor bytes initialize only the bounded post-R1 donor probability
specialist. They are **not** replayed through PPMd, LSTM, FXCM, match models or
the main mixer.

| Case | Baseline | Archive | Charged donor coordinates | All-in | Net | Roundtrip |
|---|---:|---:|---:|---:|---:|---|
| C421-D388-7 | 198,857 | 198,781 | 25 | 198,806 | +51 | exact |
| G3-D2-2 | 126,072 | 126,012 | 25 | 126,037 | +35 | exact |

Records:

```text
results/donor_collections/C421-D388-7/profile_only_result.json
results/donor_collections/G3-D2-2/profile_only_result.json
results/donor_collections/profile_only_index.csv
```

These are isolated recipient validations, not full-prefix acceptance.

### G3 cumulative status

An initial first-six-groups run timed out at 90 minutes. A later run of the
same causal profile-only plan completed successfully. Its self-contained
98-byte F4CP plan contains coordinates only and embeds no donor payload:

```
baseline:  855,173 bytes
candidate: 855,230 bytes, including the embedded F4CP
net:           -57 bytes
```

No decompression was run because compression lost. The profile-only trace
confirmed that groups before and after G3 were unchanged. This six-group run
covers more than the requested G0..G3 prefix, so G3 is closed as a cumulative
rejection. Do not rerun the unchanged G3-D2-2 plan; move to G4.

```text
results/donor_collections/G3-D2-2/profile_only_warm_timeout.json
results/donor_collections/G3-D2-2/warm_first6/g3_warm_analysis.json
results/donor_collections/G3-D2-2/warm_first6/assisted.time
```

### Rejected paths

Do not retry these unchanged:

* First-six donor-profile plan: 855,173 -> 855,230, loss 57 including plan.
* First-six direct/full-state donor replay: 855,173 -> 855,920, loss 747.
* G389-D388-7 continuous profile: 528,106 -> 528,154, loss 48.
* G390-D388-7 continuous profile: 764,795 -> 764,840, loss 45.
* One 64 KiB G389 profile: loss 34-39 bytes depending on gate.
* Global SCR2: rejected. SCR2 is allowed only as a selected group action.
* Shadow-oracle results are not acceptance evidence until recalibrated against
  the two exact profile-only results above.

## Coordinate rules

The grouped CSV is authoritative. For group `k`:

```text
prefix_start = 0
prefix_end   = row[k].post_r1_end
prefix_size  = prefix_end
```

Important examples:

| Group | Start | End | Prefix groups | Prefix bytes |
|---:|---:|---:|---:|---:|
| G3 | 3,146,668 | 4,195,514 | 4 | 4,195,514 |
| G4 | 4,195,514 | 5,243,475 | 5 | 5,243,475 |
| G421 | 441,379,615 | 442,438,432 | 422 | 442,438,432 |

The historical `C421` is not the same as grouped `G421`:

```text
C421: 441,450,496 .. 442,499,072
G421: 441,379,615 .. 442,438,432
```

C421 crosses 60,640 bytes into grouped G422. Therefore:

* A grouped-G421 action is tested through 422 groups.
* The exact historical C421 span requires a prefix through G422, or 423
  grouped recipients, so the complete span exists.

Never apply raw-stream donor offsets to an SCR2/F4PT-transformed stream. Build
a new donor plan against the transformed bytes or test SCR2 and donor modes
separately.

### Donor-profile bank limit

The current F4CP v6 profile-bank implementation has exactly 64 reusable
profile slots. The low six bits of `profile_id` select slot 0..63; the upper
two bits encode residual-gain strength. Never reuse or wrap a slot for a
different donor profile.

Until the archive format is deliberately revised, keep at most the 64 profiles
with the largest cumulative all-in gains. A 65th useful profile must either
replace a lower-value profile after an exact re-evaluation, share an identical
donor profile, or wait for a versioned format extension whose archive and
executable cost is counted.

## Non-negotiable acceptance rules

1. Stop other WSL cmix processes before every exact trial.
2. Use the canonical post-R1 stream and page-aligned group boundaries.
3. Donors in a production plan must be causal: donor end <= recipient start.
4. Donor payload bytes are regenerated from previously decoded data. Do not
   embed donor bytes.
5. Count all F4CP/F4PT plan bytes in the archive comparison.
6. Compare completed archives produced by the same binary/configuration.
7. Decompress only when candidate bytes < reference bytes.
8. Accept only after `cmp` and SHA-256 match.
9. If a candidate loses, retain the previous cumulative plan unchanged.
10. Never replace an earlier accepted action merely because a later local
    candidate looks better. Compare the complete cumulative plans.
11. Keep PPMd, LSTM and FXCM primary state untouched by donor profiles.
12. No external model, graph, donor payload or metadata may be omitted from
    Hutter accounting.

## Build

Use a production selective build. Do **not** enable the research bootstrap:

```bash
cd /root
rm -rf fx4_cumulative
mkdir -p fx4_cumulative
cp -a /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/. fx4_cumulative/
cd /root/fx4_cumulative
make clean
make cmix -j$(nproc) \
  DONOR=1 POSTR1=1 MINI_CMIX=1 VIRTUAL_REPLAY=1 \
  POSTR1_TRANSFORM=1 OUT=cmix_cumulative
```

This build supports F4CP donor/profile spans, selective mini-cmix spans and
self-contained F4PT post-R1 transforms including selective SCR2. Raw post-R1
prefix tests can be inverse-transformed exactly. No optional path is active
without a supplied plan.

## Campaign algorithm

### Stage A: cheap screening

For each group `Gk`:

1. Rank causal donor windows by phrase/token overlap and stream class.
2. Test single donors at 256 B through 64 KiB.
3. Keep positives and near-losses within 64 bytes.
4. Beam-search ordered combinations up to seven donors.
5. Test the donor probability specialist, never full predictor replay.
6. Test selective SCR2 on the complete page-aligned group as a separate mode.
7. Keep RAW/reference, RAW+donor and selective-SCR2 candidates.
8. Combine SCR2+donor only after rebuilding donor offsets on F4PT bytes.

Do not run a cumulative prefix for every losing donor. That is quadratic and
would take months. Use isolated coding plus a calibrated shadow-loss pass to
shortlist candidates. Exact cumulative coding is reserved for candidates whose
projected gain exceeds plan cost by at least 32 bytes (prefer 64 bytes).

Before trusting shadow loss, reproduce the sign and approximate magnitude of
the C421 and G3 exact profile-only gains. Otherwise fix the diagnostic.

### Stage B: exact cumulative gate

For a shortlisted action on `Gk`:

1. Extract prefix `[0, post_r1_end[k])`.
2. Build reference F4CP/F4TX from all accepted actions before Gk.
3. Build each candidate by adding exactly one new Gk action.
4. Cache reference output by binary, input and plan SHA256 values.
5. Compress all shortlisted candidate modes.
6. Choose the smallest completed archive.
7. If it is not smaller, record the loss and keep reference.
8. If smaller, decompress that winner only and verify exact equality.
9. Append the action to accepted_actions.csv and regenerate cumulative plans.

For SCR2 or any byte-changing transform, also test through at least the next
group because primary model history changes. A donor-profile-only action does
not mutate primary state, but its exact cumulative archive still must win.

### Stage C: periodic full Hutter gates

Do not run full enwik9 after every group. Run it after eight accepted actions,
50 additional screened groups, or 64 KiB projected cumulative saving.

Build plans for the complete 587,138,826-byte post-R1 stream, then run:

```bash
cd /root/fx4_cumulative
rm -f archive9 archive9.partial ppm.temp
/usr/bin/time -v env \
  FX4_DONOR_PLAN=/root/fx4_campaign/accepted_full.f4cp \
  taskset -c 7 ./cmix_cumulative -e enwik9 archive9
```

Add `FX4_POSTR1_TRANSFORM_PLAN=...` only when an accepted F4TX plan exists.
If F4TX changes the entropy stream, the accompanying F4CP must have been
rebuilt and hashed against those transformed coordinates. Never combine an
unmapped raw-stream F4CP with F4TX.

First compare compression size, including serialized plans and executable
growth. Decompress full enwik9 only if the candidate improves:

```text
S = archive9 + compressed executable + any required external bytes
```

## G3 closed; first task G4

G3-D2-2 lost its cumulative warm test by 57 bytes, so the accepted reference
still has no donor action.

For G4, use:

```text
prefix length: 5,243,475 (G0..G4)
recipient:      4,195,514 : 1,047,961
reference:      no accepted prior donor/SCR2 actions
modes:          RAW, RAW+donor profile, selective SCR2
```

Screen donor windows and SCR2 cheaply first. Exact-code only shortlisted G4
candidates against the completed RAW reference. If a candidate wins, decompress
that winner and record it; otherwise skip decompression and move to G5.

## Then G5, G6, ...

For each Gk, use prefix G0..Gk. Its reference includes every action accepted
for earlier groups:

```text
reference accepted through G(k-1)
reference + Gk donor
reference + Gk selective SCR2
reference + Gk donor/SCR2 only if transformed offsets were rebuilt
```

Repeat in group order. Every candidate, including losses, goes into the trial
ledger. Only exact winners enter accepted actions.

## Required outputs

For every exact trial record group, prefix end, recipient coordinates, mode,
donor sequence, transform and plan hashes, reference/candidate bytes, net gain,
time/RSS/faults, and status. Record decompression and SHA256 only for winners.
Never delete loss rows.

## Ready-to-paste Claude instruction

```text
Read D:\mywork\myideas\latestcompressor\fx4-cmix\CLAUDE_CUMULATIVE_DONOR_HANDOFF.md completely.

Continue the cumulative page-aligned post-R1 donor/SCR2 campaign. G3-D2-2 is
already closed: its first-six-groups causal run completed at 855,230 bytes
against 855,173, a 57-byte loss including its self-contained 98-byte F4CP.
Decompression was correctly skipped. Do not rerun that unchanged plan.

Start at G4 using exactly G0..G4 (5,243,475 bytes), then process G5 and later
groups in order. The reference for each group must include every earlier
cumulatively accepted action. Screen donor profiles and selective page-group
SCR2, but perform expensive cumulative prefix compression only for strong
projected winners. Decompress only a smaller candidate, then require cmp and
SHA-256. Never replay donor bytes through PPMd/LSTM/FXCM. Never apply raw donor
offsets to transformed bytes. Persist every win, loss, timeout and error in
results/cumulative_donor_campaign. Periodically build a full-stream plan and
run the normal -e Hutter compression; run full decompression only after an
all-in compression win.

Do not call isolated C421 or G3 results full-prefix wins. Note that historical
C421 crosses into grouped G422; grouped G421 itself uses the first 422 groups.
```
