# S1 (`cmix`) build provenance and independent reproduction

**Artifact:** `submission/cmix` — Hutter S1 (the compressor)

| | |
|---|---|
| Size | 3,369,432 bytes |
| SHA-256 | `9b644dcaf37e03bc7803c07f0c36f7258d69a6e88043b6e61382712a4b88dc8d` |
| Source | `main` @ `3d5a9c0` |
| Procedure | `bash build_and_construct_comp.sh` (see `docs/PGO_LTO.md`) |
| PGO | committed `pgo/default.profdata` — NOT regenerated |

## Components
| Component | Bytes |
|---|---:|
| Packed core (clang-17, thin LTO, `llvm-strip`, `objcopy`, UPX `--ultra-brute`) | 166,076 |
| Embedded dictionary (self-compressed, round-trip verified) | 100,054 |
| Embedded article order (self-compressed, round-trip verified) | 200,834 |
| Transformer weights (`models/6m-q4-fp32.tfwc5`) | 2,902,452 |
| **Total S1** | **3,369,432** |

## Independent reproduction (two builders, two machines)

| Build | Machine / image | Result |
|---|---|---|
| A — Naveen Bijalwan | per `docs/PGO_LTO.md` | 3,369,432 · `9b644dca…dc8d` |
| B — Dharmesh Patel | `n4d-highmem-2`, `ubuntu-2004-focal-v20240731`, us-east1-b, clang 17.0.6, UPX 5.1.1 | 3,369,432 · `9b644dca…dc8d` |

`cmp` of A and B: **byte-identical.** Build B was produced from a wiped VM and a
fresh `git archive` of committed `main`, with no artifacts carried over.

## Environment sensitivity (why the image is pinned)

The same source and profile built on a different environment does **not** reproduce
these bytes — measured:

| Variation | S1 | Δ |
|---|---:|---:|
| Ubuntu 20.04 focal, committed profile (canonical) | 3,369,432 | — |
| Ubuntu 22.04, committed profile | 3,369,832 | +400 |
| Ubuntu 22.04, regenerated PGO profile | 3,375,920 | +6,488 |

`docs/PGO_LTO.md` pins `ubuntu-2004-focal-v20240731` for this reason; it is a
build-reproducibility requirement, not a preference.

## Verify
```sh
sha256sum -c submission/cmix.sha256   # run from the directory holding cmix
```
