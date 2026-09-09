# Production Architecture

This describes the checked-in configuration of this branch, which builds on
fx2-cmix-transformer (Vladimer Ivanov, Kaido Orav, Byron Knoll).

## S1 Layout

The cmix compressor is a UPX-packed executable with an appended overlay:

    packed cmix core
    compressed english.dic
    compressed article-order file
    frozen transformer weights
    16-byte HeaderInfo trailer

build_and_construct_comp.sh creates this file. The dictionary and article
order are compressed and verified during the build (compress, decompress,
`cmp` against the original). The transformer blob is losslessly stored
because it is already a compact 2.93 MB model.

The packed core itself is built by `make target93`: clang++-17,
`-march=x86-64-v3 -mtune=generic` (a fixed microarchitecture baseline, not
the build host's exact CPU, since the judge's build host is not known in
advance), `-flto=thin`, and profile-guided (`-fprofile-use=pgo/default.profdata`
when that file is present) -- then stripped and UPX 5.1.1-packed
(`--ultra-brute`). See [README.md](../README.md#how-s1-is-built) for the full
build and PGO-regeneration commands.

## Compression

cmix -e enwik9 archive9 performs:

1. Extract and verify the S1 assets.
2. Decode the embedded dictionary and article order with the classical helper
   predictor.
3. Split enwik9 and apply the embedded article order.
4. Apply PHDA9 and WRT.
5. Require the 586,459,321-byte, 205-symbol post-WRT transformer stream.
   There is no R1 tail reorder.
7. Encode each bit using the connected production predictor.
8. Construct executable archive9.

The connected predictor contains:

- FXCM contexts and final aggregate probability, plus a stationary state
  bank.
- Full FXCM internal LSTM bridge plus the accepted half-strength middle input.
- PPMd order 25 with a 14,000 MiB logical heap.
- Frozen 12-layer, width-192, approximately 6M-parameter CPU transformer.
- Direct, indirect, bracket, word, byte and match models.
- GrammarMatch, exported both as a mixer channel and as its own layer-0
  mixer context keyed on which grammar family fired, ECHO arming on.
- A 2x200 online LSTM expert with BPTT horizon 128, contributing its own
  probability, an expected-byte hint, a disagreement channel against the
  transformer, and a gate on transformer uncertainty. Additive: it does
  not rewrite the mixed prediction.
- SSE and arithmetic coding.

The transformer replaces the online byte LSTM only on the canonical main
stream. Small embedded helper streams use the optimized online 200-cell LSTM
because their vocabularies are incompatible with the frozen 205-symbol model.

## PPM Storage

The PPM allocator uses a stable 14,000 MiB ppm.temp mapping:

- ftruncate creates the logical heap once.
- MAP_SHARED keeps the heap pointer-stable. PPMD stores raw pointers into
  its own heap, so the mapping's address must never move.
- O_NOATIME avoids access-time writes.
- MADV_RANDOM suppresses unhelpful sequential readahead.
- MADV_DONTNEED runs on a fixed byte cadence, on a mapping whose address
  never moves. PPMD stores raw pointers inside its own heap, so eviction
  must not remap; dropping page-table entries on a MAP_SHARED mapping is
  output-neutral.

Dropping resident pages does not alter PPM probabilities or the file-backed
state. The actual judge report must still confirm peak process-tree RSS below
10 GiB.

## S2 Layout

archive9 is another UPX overlay:

    packed decoder-capable core
    compressed english.dic
    frozen transformer weights
    cmix entropy payload
    16-byte HeaderInfo trailer

The article-order file is required only during compression. Information needed
for exact inverse ordering is represented by the transformed stream.

Running archive9 with no arguments creates enwik9_uncompressed.

## Source Package Contents

`tools/create_judging_entry.sh` builds the submitted source tarball from an
explicit allowlist: the production C++ codec, `dictionary/`, the frozen
transformer weights, `pgo/default.profdata` and the inputs it was generated
from, licenses, and this documentation. It never walks the working tree, so
nothing outside that list can reach a submission regardless of what else
exists in a developer checkout.
