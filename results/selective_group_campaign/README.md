# Selective post-R1 campaign

This campaign preserves the accepted fixed-interval result and uses the
page-aligned 531-group map only for discovery.

## Accepted regression case

- Canonical recipient: offset 441,450,496, length 1,048,576.
- Accepted donor source group: 388.
- Seven donor windows: 22,528 bytes total.
- Accepted entropy payload: 197,304 bytes.
- Accounted payload plus compact F4CP v7 plan: 197,334 bytes.
- Hard gate: 197,360 bytes.

The canonical interval overlaps page-aligned group 421 by 987,936 bytes and
group 422 by 60,640 bytes. Grouped 421 or 422 results are probes; neither may
replace the canonical regression result by itself.

## Generated evidence

- region421_coordinate_map.csv: both coordinate systems and source group.
- region421_like_recipients.csv: recipients similar to grouped 421/422.
- accepted_source_recipients.csv: recipients most similar to source group 388.
- enwik9.grouped_selective_candidates.csv: exact seven donor windows injected
  ahead of ordinary ranked candidates for the selected probe recipients.
- enwik9.grouped_selective_candidates.csv.recipients.csv: bounded exact-test
  order for Claude.
- campaign.json: machine-readable provenance and rules.

The first direct source-group probes are groups 529, 389, 506, 390, 524, 528,
403, 391, 432, 522, 523 and 521. The first canonical-recipient analogues are
groups 450, 456, 453, 441, 454 and 467. These are screening ranks, not measured
compression gains.

## Resume exact donor discovery

From Windows CMD:

    cd /d D:\mywork\myideas\latestcompressor\fx4-cmix
    run_claude_grouped_donor_multiday.cmd 25 7

The ledger is isolated under results/claude_donor_grouped_selective. Repeating
the command resumes it. The runner now honors FX4_WINNER_MAX_REGIONS and the
recipient filter instead of traversing every group unintentionally.

## Build the selective research executable

Inside WSL:

    cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix
    make clean
    make selective -j$(nproc) OUT=cmix_selective

This compiles F4CP donor profiles, the 11-model mini-cmix expert and F4VR
virtual replay. All remain plan-gated. No plan means no selective prediction or
transform is activated.

## Selective SCR2 rule

Use SCR2 patterns only through F4VR virtual replay. The original bytes are
replayed through PPMd, LSTM, FXCM and the mixers, so following predictor state
is unchanged. build_virtual_replay_plan.py now accepts:

    --groups-csv GROUPS.csv
    --group ID
    --minimum-group-net BYTES
    --fixed-plan-bytes BYTES
    --group-report REPORT.csv

A group is retained only when its exact marginal removed arithmetic cost pays
for its event records and shared pattern definitions. Global SCR2 remains
rejected.

## Selective mini-cmix rule

search_mini_subset_trace.py still searches arbitrary subsets of the 11
complementary models. Adjacent mini-only groups with the same model mask,
stream class and gain are now serialized as one F4CP span, reducing side data.
Its result remains an oracle shortlist until exact archive coding confirms the
gain.

## Acceptance

1. Count F4CP/F4VR bytes and executable growth.
2. Keep the 197,334 canonical case untouched.
3. Do not call a screen score or oracle loss a compression win.
4. Full-prefix validation is required before a donor edge is used in enwik9.
5. Roundtrip and peak RSS checks remain mandatory for production.
