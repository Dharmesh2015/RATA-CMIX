# Accepted post-R1 region 421

The accepted seven-donor result is retained as the foundation.

| Item | Bytes |
| --- | ---: |
| Baseline entropy payload | 198,852 |
| Baseline archive (5-byte header) | 198,857 |
| Donor-assisted entropy payload | 197,304 |
| Verified donor-assisted archive | 197,309 |
| Compact F4CP v7 donor plan | 25 |
| Accounted all-in result | **197,334** |
| Hard no-regression gate | **197,360** |

The current source reproduced the exact 197,309-byte archive and its SHA-256.
Compression and decompression both restored the canonical 1 MiB post-R1 region
byte-for-byte. The compact v7 plan separately passed an exact writer/reader
size check: predicted=25 actual=25 reread=25.

region421.f4cp is the readable 121-byte external research plan. It is not the
archive cost. During compression it is rewritten into the 25-byte compact v7
representation stored in the archive.

The seven donor offsets and lengths are in donor_edges.csv. The decoder can
reconstruct them from earlier decoded post-R1 bytes; donor contents are not
embedded.

## Executable accounting

The accepted donor-only build is 386,912 bytes. The probability-specialist
framework adds 41,040 bytes; the selectable 11-predictor mini expert adds a
further 12,288 bytes. The combined research build is 440,240 bytes, or 53,328
bytes above the accepted build. It stays research-only until full-file archive
savings exceed that executable growth.
## Acceptance policy

- Never replace this path with a result above 197,334 bytes after side data.
- Never allow any experiment to exceed the 197,360-byte hard gate.
- Keep gross-positive selective predictor pages in research catalogs even when
  they cannot yet amortize their assignment metadata.
- Exact donor replay changes adaptive state. Interior full-stream use therefore
  requires full-prefix validation including downstream bytes.
- Probability-only donor and mini-cmix experts do not mutate PPMD, LSTM, FXCM,
  ByteMixer, or the main mixer state and are the safe mode for page-local gains.