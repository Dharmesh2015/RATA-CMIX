// Transparent-huge-page (THP) hint for large, hot, randomly accessed
// allocations.  The benchmark kernels run THP in "madvise" mode, so the
// multi-GB model tables sit on 4 KB pages unless a region is explicitly
// flagged with madvise(MADV_HUGEPAGE).  Flagging them lets the kernel back
// the tables with 2 MB pages at first-touch fault time, which shrinks the
// dTLB working set from ~800k pages to ~1.5k and removes page-walk latency
// from the random-access load chain.
//
// This is a pure address-translation hint: it never changes allocation
// order, contents, zeroing, or the rand() sequence, so compressed output is
// bit-identical.  Call it right after allocating and BEFORE the first touch
// of the memory - khugepaged's background collapse is far too slow to help
// within a single run.
//
// No-op on non-Linux platforms (e.g. the Windows Hutter test machine) and on
// madvise failure; correctness never depends on it.
#ifndef UTILS_HUGEPAGE_H_
#define UTILS_HUGEPAGE_H_

#include <stddef.h>
#include <stdint.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

inline void AdviseHugePages(const void* ptr, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  // Below a couple of 2 MB pages there is nothing to gain.
  if (ptr == NULL || bytes < ((size_t)4 << 20)) return;
  const uintptr_t kPageMask = 4095;  // madvise needs page-aligned start
  uintptr_t begin = ((uintptr_t)ptr + kPageMask) & ~kPageMask;
  uintptr_t end = ((uintptr_t)ptr + bytes) & ~kPageMask;
  if (end > begin) {
    // Best effort; the kernel promotes only 2 MB-aligned blocks fully inside
    // the range, edges stay 4 KB.
    (void)madvise((void*)begin, end - begin, MADV_HUGEPAGE);
  }
#else
  (void)ptr;
  (void)bytes;
#endif
}

#endif  // UTILS_HUGEPAGE_H_
