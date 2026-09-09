#ifndef PPMD_H
#define PPMD_H

#include "byte-model.h"

// ALREADY SOLVED ELSEWHERE IN THIS REPOSITORY. main, trace/aws,
// release/google-cloud-hutter, release/hutter-s1 and cmix-lex-main all
// carry O_NOATIME, ftruncate, MADV_RANDOM and MADV_DONTNEED, and none of
// them has the address-moving remap. Their note gives the same diagnosis
// this branch arrived at independently: PPMD stores raw pointers inside
// HeapStart, the munmap+mmap cycle destroyed the VMA and relied on the
// kernel handing back the same address, and readahead is wasted on a
// pointer-chased tree. Theirs is the tested version -- it is credited with
// making a completed WSL2/SATA run practical -- so the defaults below
// match it rather than my own first guesses.
//
// The v521 lineage did not inherit it: fx2-cmix, fx3-cmix,
// fx2-cmix-transformer, fxcmv26 and this branch all still had the broken
// remap. That is the whole difference between the two families here.
//
// Back PPM's heap with a file instead of anonymous RAM. Default 0, which
// is the behaviour every measurement on this branch was taken with. main
// defaults its equivalent, FX4_PPMD_MMAP_TO_DISK, to 1: that is the
// release configuration, and this stays off only so the branch's existing
// measurements keep describing the default build.
//
// The point is to fit the Hutter Prize's 10 GB decompressor limit while
// letting the allocator stay large: pages the model is not touching live
// on disk instead of in RSS. That is the trade the reference entry made,
// and it is also why it took 8 days on a memory-constrained Intel box
// against 36.5 hours on AMD -- PPM access is irregular, so once the
// working set exceeds RAM every lookup risks a fault.
//
// Measure FX2_PPMD_MEMORY_MB=9500 in plain RAM before reaching for this.
// If a smaller in-RAM model costs little compression it fits the limit
// with none of this risk.
// PPM's heap is DISK-BACKED. The 14 GB sub-allocator cannot live in
// anonymous RAM inside the judging memory limit, so it is mapped to
// ppm.temp and residency is held down by dropping resident pages on a
// fixed byte cadence. That is output-neutral: clearing present
// page-table entries on a MAP_SHARED mapping changes nothing the model
// reads back, only where the bytes are held.
//
// The mapping's ADDRESS must never move. PPMD stores raw pointers into
// its own heap -- pText, UnitsStart, LoUnit, HiUnit, the free lists and
// every tree node -- so eviction is MADV_DONTNEED, which keeps the
// address, and never munmap()+mmap(), which does not. See ppmd.cpp.
//
// Bytes between residency drops.
#define FX2_PPMD_MMAP_DROP_BYTES 5000

#include <memory>

namespace PPMD {

struct ppmd_Model;

class PPMD : public ByteModel {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  void ByteUpdate();
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
};

} // namespace PPMD

#endif

