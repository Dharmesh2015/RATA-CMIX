# Donor collection results

This directory contains one auditable record per exactly coded donor-recipient
winner. Proxy rankings and estimated gains do not belong here.

Each collection directory contains:

* `result.txt`: human-readable payload, side-data, and net accounting.
* `result.json`: the same data using schema `FX4_DONOR_COLLECTION_V1`.
* `donors.csv`: ordered donor offsets, lengths, and ends.
* Optional exact-cost reports for selective transforms such as SCR2 virtual
  replay.

The accounting rule is:

```text
all_in_bytes = assisted_payload_bytes + F4CP_plan_bytes + other_plan_bytes
net_saving_bytes = isolated_baseline_bytes - all_in_bytes
```

A collection remains marked `full_prefix_validation` until it wins after the
normal causal prefix in the complete post-R1 stream. Isolated results must not
be presented as full-file savings.

Regenerate every collection report after updating `accepted_exact.csv`:

```bash
python3 tools/write_donor_collection_reports.py \
  results/postr1_opportunity_ledger/accepted_exact.csv \
  results/donor_collections \
  --augment-csv results/donor_collections/scr2_validation.csv
```

SCR2 is accepted only as virtual replay. Both coder sides observe every
expanded original byte through `ObserveKnownByte`, so the base PPMd, LSTM,
FXCM, match, and mixer histories remain the same as the corresponding donor
run. A physical SCR2 token stream is not accepted here.
