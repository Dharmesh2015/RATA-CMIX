# FX4 Post-R1 Opportunity Handoff

## Accepted result

Keep `C421-D388-7` unchanged:

- Recipient: `[441450496, 442499072)`
- Seven donor windows: 22,528 bytes from source group G388
- Baseline archive: 198,857 bytes
- Accepted payload: 197,309 bytes
- Compact plan: 25 bytes
- All-in result: 197,334 bytes
- Exact isolated round trip: passed
- Full-prefix validation: still required

The authoritative row is `results/postr1_opportunity_ledger/accepted_exact.csv`.

## Next trials

Use `results/postr1_opportunity_ledger/validation_queue.csv` in order.

1. `donor_D388_7`: reuse the already validated seven-window profile on 32 new recipients.
2. `donor_new_source`: refine each proposed source into an ordered 4-8 window bundle before exact testing.
3. `scr2_virtual_replay_oracle`: test only arithmetic-cost-positive macro events. Never feed SCR2 tokens into the base predictor.
4. `mini11_oracle_only`: evaluate subsets, but do not enable the 11-model mini-cmix unless cumulative savings pay selection metadata and binary growth.

The first D388 recipients are G389, G390, G391, G403, G524, G509, G528, G522, G432, G525, G523, and G508.

## State rule

All modes are recipient scoped. Donor bytes update only the donor specialist. Virtual replay emits its event but passes each expanded original byte through PPMd, LSTM, FXCM, match models, and the main mixer exactly once. At the recipient end, clear only the specialist/event state.

Do not replay donor bytes into the base Predictor. Do not replace the post-R1 bytes seen by the base models. This preserves subsequent probabilities.

## Evidence labels

- `exact_isolated_roundtrip`: measured payload and plan; full-prefix gate remains.
- `validated_source_proxy`: D388 is proven on C421, recipient is still estimated.
- `uncalibrated_source_proxy`: similarity ranking only.
- SCR2 raw saving: exact structural byte count, not entropy saving.
- Mini11 high bound: C421 class-prior oracle, not a deployable saving.

Only write a production F4CP edge when exact warm-prefix payload gain exceeds profile, assignment, span, executable, and transform metadata.

## Regenerate

Run from `fx4-cmix`:

```bat
python tools\build_postr1_opportunity_ledger.py --groups results\donor_multiday_seed\enwik9.grouped_1mib_recipients.csv --accepted results\accepted_region421_197360\acceptance.json --accepted-edges results\accepted_region421_197360\donor_edges.csv --accepted-source-recipients results\selective_group_campaign\accepted_source_recipients.csv --diverse-candidates results\diverse_group_campaign\candidate_source_map.csv --scr2-encoded ..\special_scanner\postr1_structural_codec\enwik9.scr2.bin --scr2-meta ..\special_scanner\postr1_structural_codec\enwik9.scr2.meta --mini-oracle results\mini_subset_20260809\page421_v6_oracle.span.csv --output results\postr1_opportunity_ledger
```
