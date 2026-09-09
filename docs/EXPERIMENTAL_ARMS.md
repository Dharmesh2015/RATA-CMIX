# Experimental probability and preprocessing arms

The accepted baseline on `exp/gch-optimization` includes `Scr2Match` in its
agreement-only form. On the 1 MiB matrix it produced 84,713 bytes versus
84,726 bytes for control, a 13-byte payload win. KMP fallback and bit-position
confidence splitting were measured and removed.

## Morphology and causal donor -- now baseline

Both observed only already-decoded bytes, abstained at probability 0.5, and
never altered PPMd, Transformer, LSTM, or FXCM state. Measured together
through 4,456,448 of the 5,871,388-byte prefix (run interrupted before
FINAL, but the trend was monotone and still growing at that point, not
fading): -43 bytes against the pre-scr2_agree baseline and -31 against the
scr2v1 tie-silencing reference, including through the 4,325,376..5,242,880
expensive band from docs/COST_MAP.md. That is a stronger, more sustained
signal than either arm alone -- morphology by itself faded to zero against
donor by ~900 KB, since it targets first-time-unseen literals, which are
naturally front-loaded; combined with donor's flatter, steadier contribution,
the total kept climbing instead.

Both are now unconditional. FX2_MORPHOLOGY_MATCH, FX2_MORPHOLOGY_MATCH_MIXER,
FX2_CAUSAL_DONOR and FX2_CAUSAL_DONOR_MIXER are gone along with BENCH_MORPH
and BENCH_DONOR; there is nothing left to flag off.

## ALTXS M3/M5 -- removed

Explored on this branch, then removed entirely: M5 (WRT-block reorder) and
M3 (PHDA9 densify), imported from upstream ALTXS. A small-input round-trip
probe found the production `-e` path needs genuine full-scale enwik9 --
article reordering and PHDA9 preprocessing degenerate on an arbitrary byte
prefix, producing both an early error and a multi-day runtime projection on
a 2 MB test input -- so no correctness or compression result was ever
obtained for either piece. All source (`src/altxs_transform.{h,cpp}`,
`src/third_party/altxs/`), the `ALTXS`/`ALTXS_M3` make variables, and the
`tools/test_altxs_small.{sh,cmd}` probe are gone; nothing here references
FX2_ALTXS_M5, FX2_ALTXS_M3, or altxs:: any more.
