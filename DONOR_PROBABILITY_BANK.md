# Donor Probability Bank
## Canonical acceptance gate

Canonical region 421 (offset 441450496, length 1048576) keeps the proven seven
ordered donors. The exact entropy payload is **197,304 bytes**; the verified
archive with its five-byte header is **197,309 bytes**. The compact F4CP v7
plan costs **25 bytes**, producing an accounted all-in incumbent of
**197,334 bytes**. **197,360 bytes** is the hard no-regression gate.

The historical 98,090 and 99,182 archives were measured on a different
raw/wrapped input and are not valid gates for this post-R1 path.

The first selective page oracle covered 599 complete pages in region 421:

```text
mini-cmix forced globally:       4.442767-byte loss
donor specialist forced:        1.485402-byte loss
combined forced:                6.093859-byte loss
perfect page oracle, no costs: 15.767000-byte gain
net-positive planned pages:     0
```

Therefore region 421 keeps the exact seven-donor replay. The 11-predictor
mini-cmix and donor probability specialist remain optional, page-scoped
experts for other post-R1 recipients. They are disabled in the default build.

## Purpose

Turn exact donor evidence into a small causal probability expert without storing
one donor/profile ID for every recipient page.

The accepted PPMd, LSTM, FXCM and mixer state stays continuous. Donors only
build immutable 4/8/16/32-byte continuation tables. Their output is a bounded
logit correction, so a miss is neutral and cannot reset the primary models.

## Exact evidence

`donor_winner_search.cpp` now writes:

```text
<ledger>.winner_marginals.csv
```

For the best raw winning profile in each recipient, every donor is removed in
turn from an identical warm predictor state:

```text
marginal_gain = payload_without_donor - payload_with_full_profile
```

Only positive marginal gain is evidence that the donor helped that profile.
The ledger is durable and resumes without repeating completed removals.

Summarize reusable donors with:

```bash
bash tools/find_multi_recipient_donors.sh \
  results/page.winner_marginals.csv 4665 2 \
  > results/multi_recipient_donors.csv
```

The report contains two deliberately optimistic bounds:

* `explicit_net_upper_bound`: gain minus one registration and ideal activation
  bitmap entropy.
* `causal_no_id_upper_bound`: gain minus one registration, assuming a causal
  selector can reproduce all useful activations without page metadata.

Neither bound is a production archive result.

## Selection sequence

1. Finish exact marginal collection for all raw-positive profiles.
2. Rank donors by held-out positive marginal, not trial frequency or proxy
   phrase score.
3. Greedily select `K = 2, 4, 8` donors. Charge each offset/length once.
4. Re-run combinations after each addition because donor effects are not
   additive.
5. Train the selector on one set of pages and validate it on disjoint pages.
6. Keep a bank only when exact held-out payload gain exceeds registration,
   selector, executable and archive framing costs.

## Compact page plan

External F4CP v5/v6 stores each donor sequence once as a reusable profile; the archive writer compacts accepted plans into v7. A page
span refers to profile `0..63`; it does not repeat all donor offsets. Both the
plan builder and decoder reject a profile whose donor bytes have not already
been decoded before its first use.

Example donor bank CSV:

```csv
profile,donor_offset,length,order,keep
0,407573504,4096,0,1
0,407581696,2048,1,1
0,407775744,2048,2,1
```

Select only pages whose estimated gain pays for their span record and shared
profile bank:

```bash
python3 tools/select_postr1_page_experts.py \
  results/page421.span_oracle.csv results/page421.selected.csv \
  --span-source results/page421.oracle.experts.csv \
  --min-net-bytes 1 --fixed-plan-cost 7 --profile-bank-cost 49
```

Build the version-5 profile plan:

```bash
python3 tools/build_postr1_portfolio.py enwik9.post_r1.bin \
  --prefix results/enwik9.selective \
  --expert-csv results/page.selected.csv \
  --donor-csv results/donor_profiles.csv \
  --donor-profile-bank
```

Cross-entropy selection is only a shortlist. For region 421, retain a plan
only when the all-in archive is below 197334 bytes after plan cost (and never above the 197360-byte hard gate). Run
decompression only for an exact winner.
## Causal selector with no recipient IDs

Use only decoder-visible state:

```text
page/pack stream class
first 64-256 decoded bytes
recent 4/8/16/32-byte context hits
candidate agreement and confidence
base versus donor logit disagreement
rolling donor log-loss gain
```

At each bit, look up at most the best two donor profiles. Feed one aggregate
probability plus confidence, agreement and longest-hit bucket into a small
fixed-array contextual mixer. Update its gate online after the decoded bit.
The encoder and decoder therefore make the same decision without side data.

A donor at a later post-R1 offset cannot be used causally for an earlier page
unless compression order is changed and restoration metadata is stored. Start
with earlier donors only.

## Production gate

Count all of:

```text
donor offset/length registrations
any activation or ordering metadata
archive framing
compressed executable growth
payload change
compression and decompression time
peak RSS
```

Require exact roundtrip and net total-size improvement. Old donor ledgers must
not be combined with a new predictor binary; changing the mini-cmix expert
changes both baseline and donor marginal costs.