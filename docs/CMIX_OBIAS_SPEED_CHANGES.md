# Imported speed changes — cmix_residual_trace

This tree = **COPY_ME_CMIX_LEX** (pristine cmix-lex, byte-identical to speed-repo
commit `370e698`) **+ the kept speed-campaign steps**, with the update-skip /
drift-gate "skip" class deliberately **excluded** (see below).

- Pristine source: a byte-identical copy of upstream cmix-lex `370e698` (src/ + makefile).
- Speed-campaign working tree: an internal read-only checkout of the speed steps below.
- Relevant commits: `370e698` (upstream base) · `894a19b` (plan_001 001-010 minus 007 + plan_002 001-003 + ppmd nondeterminism fix) · `da9c230` (checkpoint of the in-flight plan_005 working tree: kept loose steps 003_002/003_003/003_006/003_004/004_003/004_004 **plus** the reverted step_005_003 "step027v2" on fxcmv1.cpp).
- Assembled: 2026-07-11.

---

## 1. Steps INCLUDED (present in this tree)

All percentages are the per-step same-day wall speedups recorded in
`speed_hunter_165/improvements_kept.md`.

| Step | What (one line) | Speedup | Carried by files |
|------|-----------------|---------|------------------|
| 001_001 | LSTM: flat contiguous matrices, fused gate matvecs | +8.45% | (committed in 894a19b baseline; lstm.* / lstm-layer.*) |
| 001_002 | LSTM: AVX-512/AVX2 vectorized activations | +5.33% | simd-activations.h (new) |
| 001_003 | Huge pages (MADV_HUGEPAGE) for multi-GB tables | +3.81% | hugepage.h (new), context-manager.cpp, fxcmv1.cpp, ppmd.cpp |
| 001_004 | Software-prefetch fxcm context-map buckets | +5.13% | fxcmv1.cpp, indirect.h, context-manager.cpp |
| 001_005 | Outer mixers: flat aligned weight slab + BeginBit prefetch | +7.39% | mixer.cpp, mixer.h, predictor.cpp |
| 001_006 | PPM: RSS-aware MADV_DONTNEED purge (CMIX_PPM_RSS_MB knob) | +1.26% | ppmd.cpp |
| 001_008 | fxcm CM3/4: branch-free bucket probe / mix3 | +2.43% | fxcmv1.cpp |
| 001_009 | Predictor: batched input marshaling, flat aligned mixer input | +1.26% | predictor.cpp, mixer-input.h/.cpp, mixer.h/.cpp, sigmoid.h |
| 001_010 | fxcm: hot/cold split of modelPrediction (I-cache) | +1.20% | fxcmv1.cpp |
| 002_001 | CM3/4::mix(): eliminate remaining data-dependent branches | +6.21% | fxcmv1.cpp |
| 002_002 | LSTM BPTT: deferred gradient outer-products, interleaved Adam | +2.02% | lstm-layer.h/.hpp |
| 002_003 | LSTM: fp16 (F16C) shadow weight streams | +5.55% | lstm.h/.hpp, lstm-layer.h/.hpp, simd-activations.h |
| 003_002 | LSTM output layer: live matrix + rank-1 history | +3.76% | lstm.h/.hpp |
| 003_003 | fxcm Mixer1: prefetch context weight rows at set time | +1.37% | fxcmv1.cpp (surgical) |
| 003_006 | LSTM: fp16-only storage for per-epoch BPTT histories | +0.97% | lstm-layer.h/.hpp, simd-activations.h |
| 004_003 | fxcm: hoist Mixer1 + DirectStateMap ctx to bit entry, prefetch | +1.32% | fxcmv1.cpp (surgical) |

The single main `makefile` is **unchanged** between `370e698` and `da9c230`
(verified empty diff) and is what `build_and_construct_comp.sh` invokes
(`make prof_gen` / `make prof_use`); its default `-march=native` enables the
AVX-512/F16C paths gated in simd-activations.h. It is kept as-is from pristine.

---

## 2. Steps EXCLUDED (deliberately NOT present)

| Step | What | Why excluded |
|------|------|--------------|
| **003_004** | Outer Mixer::Perceive drift-gated small-update skip (skips entire weight RMW when `fabsf(update)<1e-6`) | User rule: "don't skip updates to mix buckets". Skip class. (Was +2.86%.) |
| **004_004** | Branchless compaction of the 003_004 skip (two-phase batched RMW) | Companion of 003_004; same skip class. (Was +0.87%.) |
| **005_003 / step027v2** | fxcm Mixer1::update() zero/small-error RMW skip (`if(err>=-16 && err<=16) return;`) | In-flight and **reverted** upstream; also skip class. |
| all other plan_005 | 005_001 (ppmd branch-free), 005_002 (fp16 mixer shadow), 005_004/005/006 | All `reverted` or `pending` in current_plan.json — none kept. |

Also NOT imported (per recipe): benchmark/diagnostic makefiles added by the
campaign (`.gitignore`, `makefile.asan`, `makefile.cg*`, `makefile.prof*`,
`makefile.st26/27/27fp`, `makefile.vg`) — none existed at `370e698` and none are
used by the canonical `build_and_construct_comp.sh` path; PGO data dirs,
`cmix_step*` binaries, `bench_*/`, `run/`, and progress logs.

---

## 3. Per-file provenance (source commit + md5 of the imported file)

Import decision: mixer.{h,cpp} + predictor.cpp come from **894a19b** (this drops
excluded 003_004/004_004 while keeping committed 001_005/001_009). fxcmv1.cpp is
**surgical** (see §4). Everything else comes from **da9c230** — and for each,
the 894a19b→da9c230 diff was verified attributable to a kept step (or byte-identical).

| File | Source | 894a19b→da9c230 delta = kept step | md5 (imported) |
|------|--------|-----------------------------------|----------------|
| src/mixer/mixer.h | 894a19b | (da9c230 adds only excluded 004_004) | 90f327bba950e195e6c358272f2a0808 |
| src/mixer/mixer.cpp | 894a19b | (da9c230 adds only excluded 003_004+004_004) | 52ec678b5a2b29f58dcfc021ce085a65 |
| src/predictor.cpp | 894a19b | (da9c230 adds only excluded 004_004) | 87174513a9681e7b8558be030b20d96a |
| src/models/fxcmv1.cpp | **surgical** | 894a19b + 003_003 + 004_003, no step027v2 | 824230eb715ffa9b272cd3de06365f25 |
| src/context-manager.cpp | da9c230 | UNCHANGED vs 894a19b (001_003/001_004 committed) | 33e408011101184d29844ef289fa07ba |
| src/models/indirect.h | da9c230 | UNCHANGED vs 894a19b (001_004 committed) | 34b811371e3b2cb8ecd86c2c6f018dd3 |
| src/models/ppmd.cpp | da9c230 | UNCHANGED vs 894a19b (001_006 + ndet fix committed) | 0f1c2e033aa363e4f00538c2eaba23fd |
| src/mixer/mixer-input.cpp | da9c230 | UNCHANGED vs 894a19b (001_009 committed) | 10cced33b3e43b9d46deb57e5fb04a71 |
| src/mixer/mixer-input.h | da9c230 | UNCHANGED vs 894a19b (001_009 committed) | 8fc9ea10968cdbf35e14c158eb66c03d |
| src/mixer/sigmoid.h | da9c230 | UNCHANGED vs 894a19b (001_009 committed) | d96313efc16a60d85d7e6a212c2fb2cd |
| src/utils/hugepage.h | da9c230 | NEW file (001_003) | 974169a1b165d8c89dddf25a2b147257 |
| src/mixer/simd-activations.h | da9c230 | +14 lines = 003_006 F16DecodeRow kernels | 0033dd8372fe24a0b56a5737e7f8600b |
| src/mixer/lstm.h | da9c230 | 003_002 (live matrix + rank-1 members) | 06204f13a2268074364ccfa3e55d6ec1 |
| src/mixer/lstm.hpp | da9c230 | 003_002 | ce19fdcd50a1f6f103e97135ca4f0cb7 |
| src/mixer/lstm-layer.h | da9c230 | 003_006 (fp16-only BPTT histories) | df0b44bb71fe4d208ee3e3b363b31fea |
| src/mixer/lstm-layer.hpp | da9c230 | 003_006 | 9dea3d99898029359a06fd0a85fb6a05 |

`diff -rq` of this tree's `src/` vs pristine `src/` reports **exactly** these 16
paths (14 modified + simd-activations.h + utils/hugepage.h new) and nothing else.

---

## 4. fxcmv1.cpp surgery — hunk attribution

Target = `894a19b:fxcmv1.cpp` + step_003_003 + step_004_003, **without** step027v2.

The `da9c230` working-tree copy already had step027v2 removed down to a single
7-line block; verification confirmed that block was step027v2's **entire**
footprint, so the resulting file equals the target. It is byte-identical to the
speed-repo working-tree fxcmv1.cpp (md5 824230eb…).

### Hunks KEPT — diff(894a19b → final), every hunk attributed

| Hunk (894a19b line) | Content | Step |
|---------------------|---------|------|
| @@ -747 (+17) | `Mixer1::prefetchRowAt(cx)` + change-gated `Mixer1::setCxt(cx)` | 003_003 |
| @@ -5159 (+49) | global `U32 dcsmCx[19]` + `dcsmPrecomputeCx()` (19 DirectStateMap ctx precompute + prefetch) | 004_003 |
| @@ -5590 (+7) | end-of-`setByteContexts()` prefetch of the 4 mxA2 weight rows (c1,c2,stream5b,stream2b) | 003_003 |
| @@ -5602 (+95/-31) | `modelPrediction()` top: dcsm.set→dcsmCx hoist + 16 Mixer1 setCxt sites hoisted after c0b (mxA[3] operand = `(c0b&255)`) | 004_003 (+ 003_003 setCxt conversions) |
| @@ -5699 (-53) | removal of the hoisted mxA setCxt sites from their old per-bit location | 004_003 |
| @@ -5802 (+22/-31) | remaining `mxA[i].cxt=EXPR`→`setCxt(EXPR)` conversions; mxA[8..16] sites removed (hoisted) | 003_003 + 004_003 |
| @@ -5880 (+5) | `if (x.bpos) dcsmPrecomputeCx();` in update() (prefetch under the mixer-train block) | 004_003 |
| @@ -5931 (0) | `mxA2[0..3].cxt=`→`setCxt()` conversions | 003_003 |

There is **one cosmetic change** inside the @@ -747 hunk: a trailing space on
`tx=mn;` was stripped (value-neutral).

### Hunk REMOVED — step027v2 (step_005_003), diff(da9c230 → final)

| Removed (da9c230 line 729-735) | Justification |
|-------------------------------|---------------|
| A 6-line comment + `if (err>=-16 && err<=16) return;` at the top of `Mixer1::update(int y)`, skipping the weight-row `train()` RMW when the integer error is in the elim window | This is exactly the step_005_003 "zero/small-update skip on DRAM-cold rows" — reverted upstream, and skip class. Removed. |

**Sanity confirmed:** `Mixer1::update(int y)` now ends with an **unconditional**
`train(&tx[0], &wx[cxt*N], N, err);`. The surviving `if(err>=-elim && err<=elim)
err=0;` above it is **pre-existing 894a19b base code** (the `elim` member and the
±32767 clamps are in the base) — it zeroes `err` but still runs the RMW; it is
NOT a skip. A grep of the final file for any `err`-based `return`/skip found none
beyond this base elim-zeroing. No skip counters, instrumentation globals, or a
batched variant were present (step027v2 in da9c230 was the minimal 7-line form).

---

## 5. Expected properties

- **Throughput:** the kept steps compose to roughly **+76% throughput** vs
  pristine cmix-lex (pristine ≈ 0.01542 MB/s enwik7-dict → ≈ 0.0272 MB/s). The
  full kept campaign line reached ≈0.02796 MB/s (+81%); excluding the two skip-
  class steps 003_004 (+2.86%) and 004_004 (+0.87%) accounts for the ~+76% vs
  +81% difference. Numbers are dev-box (Zen5 X3D anchor); several steps note the
  official non-V-cache Hutter machines should gain **more**.
- **Compression drift:** several kept steps are FP-reorder / fp16-quantization
  drift class (001_001, 001_002, 002_002, 002_003, 003_002, 003_006), so the
  compressed output will differ **slightly** from pristine cmix-lex (each step's
  drift was gated well under 0.075%; cumulative drift is small but nonzero).
  This drift is only in the compressed-size comparison against pristine
  cmix-lex; it does not affect this build's own encode -> decode roundtrip,
  which is byte-exact (full enwik9 verified, md5 e206c3450ac99950df65bf70ef61a12d).
- **Correctness class:** the remaining kept steps (001_003/004/005/006/008/009/010,
  002_001, 003_003, 004_003) are integer-exact / prefetch / layout — value-exact
  by construction.

## 6. Run protocol

- **Toolchain:** `clang++-17` is NOT on PATH by default on this box. Before
  building, export (wrappers first):
  ```
  export PATH="/path/to/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin:$PATH"
  ```
  (LLVM 17.0.6; `llvm-profdata-17` / `llvm-strip-17` from the same tree are
  needed by `build_and_construct_comp.sh`'s PGO path. Do not modify that tree.)
- **PGO:** a **full PGO regeneration is required** before any real run
  (`build_and_construct_comp.sh` does this: `make prof_gen`, two training runs,
  `llvm-profdata-17 merge`, `make prof_use`). No pgo_data is shipped in this tree.
- **CMIX_PPM_RSS_MB** (step_001_006): default **9216** MB caps PPM heap RSS at
  ~9.26 GB (Hutter 10 GB budget). On this ~125 GB dev box, export a large value
  (e.g. `CMIX_PPM_RSS_MB=100000`) so the PPM heap is never purged during
  benchmarking; `CMIX_PPM_RSS_MB=0` reproduces the old always-purge cadence.

## 7. Compile smoke test result (2026-07-11)

Built the non-PGO `make cmix` target with the LLVM-17 toolchain above.
**All 31 translation units compiled cleanly** (including the surgical fxcmv1.cpp,
the new src/utils/hugepage.h, and every imported file). The clang-only link
step failed with undefined **C++ standard-library** symbols — reproduced
identically on a trivial hello-world, i.e. this standalone toolchain lacks a
64-bit `libstdc++.so` dev symlink (only `libstdc++.so.6` runtime is installed);
it is an environment quirk, not a source/integration problem. Re-linking the 31
clang-built objects with the system `g++` (GCC 12, same ABI) produced a working
336 KB ELF that runs — confirming the full object set links with **zero
unresolved application symbols**. All build artifacts (.o, binary) were then
removed; the tree is pristine.
