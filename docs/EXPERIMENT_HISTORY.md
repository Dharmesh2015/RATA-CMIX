# FX4 experiment ledger

This file preserves the high-signal results from the removed generated
`results/` and `research/` trees. Values from different modeled streams are
not directly comparable.

## Target93 selective discovery correction

The historical donor discovery constructors used `Predictor(vocab)`, which
left the frozen transformer disabled even in a transformer-compiled binary.
Those ledgers therefore do not establish donor or SCR2 gains over target93.
The discovery API now propagates the transformer''s runtime enable flag into
all parent and forked predictors. New target93 donor/SCR2 evidence must come
from `tools/run_target93_selective20_ext4.sh` or an equivalent corrected
build.

The production transformer link also omits the uncompressed training-weight
loader. The compressed `.tfwc2` loader now supplies the two shared tensor
accessors directly; model bytes and predictions are unchanged.

## Retained conclusions

| Experiment | Measured result | Production decision |
| --- | ---: | --- |
| cmix-lex canonical reference | 109,190,109-byte S2; 109,650,047 total | Reference |
| supplied FX2 transformer | 96,996,198-byte payload family | CPU transformer replacement is the strongest available path below 100 MB |
| cmix-obias family | about 108,492,825 total | Better reference than cmix-lex, but tied to online-LSTM features |
| SCR2 global transform | 22,045,576 raw bytes removed; final payload +46 bytes and core +8,208 bytes | Reject globally |
| isolated C421 seven-donor replay | 198,857 -> 197,309; 25-byte plan; 197,334 all-in | Research-only; not a warm full-prefix accepted win |
| early-prefix G5 donor trial | 363-byte local/prefix saving | Did not remain a production full-stream win |
| G0-G6 selective portfolio | reported 26-byte saving | Disappeared after corrected warm/side-data accounting |
| LSTM-to-FXCM bridge | isolated 2-6 byte gains depending build/input | Already represented by accepted source lineage where applicable |
| DeltaMemory8 | 1,033-byte loss over 5 MiB FXOT trace | Remove |
| token n-gram bias | about 191-byte in-sample gain; about 3.2-byte chronological holdout gain | Below code-size gate |
| residual ACTW/APM | about 17 bytes | Below code-size gate |
| 11-model mini-cmix | no warm all-in winner found | Discovery-only, not S1 |
| local/shadow LSTM-200 | loss on canonical post-R1 tests | Do not add beside transformer |

## Coordinate rule

The 531 grouped recipient definitions belong specifically to the canonical
587,138,826-byte payload_lex/R1 modeled stream. They remain valid for the
connected `target93` build because it does not enable M3/M5, IDHOIST, SCR2 or
another byte-changing transform.

## Acceptance rule

Only a complete `-e` compression, standalone `archive9` extraction, exact
SHA-256 match, peak RSS under the Hutter limit, and `S1 + S2` accounting can
promote a research result into the production ledger.
