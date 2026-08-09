# Canonical post-R1 region 421 selective test

Historical entropy gate: **197,304 bytes**. The exact archive is 197,309 bytes;
the compact v7 donor plan is 25 bytes; and the accounted all-in incumbent is
**197,334 bytes**. The hard no-regression gate is **197,360 bytes**.

This is the canonical post-R1 result with the proven seven ordered donors. Raw
98--99 KB experiments are not acceptance baselines for this test.
## One-pass page oracle

- Complete post-R1 pages tested: 599
- Compression wall time: 569.94 seconds (9:33.33)
- Peak RSS: 8,447,984 KiB
- Major page faults: 6,146
- Global mini-cmix loss: 4.442767 bytes
- Global donor-profile loss: 1.485402 bytes
- Global combined loss: 6.093859 bytes
- Ideal zero-metadata page oracle gain: 15.767000 bytes
- Pages surviving exact span and shared donor-bank cost: 0

The 202,124-byte discovery archive contains all 599 oracle span records and is
not a compression candidate. No second compression or decompression was run,
because the net selector produced no span and therefore could not improve the accepted donor path.

## Production decision

Keep the seven-donor replay for canonical region 421. Do not assign mini-cmix
or the donor probability specialist to this region. Both remain optional
research modes for other post-R1 pages and are excluded from the default build.

Artifacts:

- `page421.span_oracle.csv`: per-page baseline/mini/donor/combined log loss.
- `page421.oracle.experts.csv`: exact post-R1 page spans used by discovery.
- `page421.oracle.time.txt`: GNU time report.
