#include "morphology-match.h"

#include <algorithm>

namespace {
int BitPos(unsigned int c0) {
  int n = 0;
  while (c0 > 1u) { c0 >>= 1; ++n; }
  return n > 7 ? 7 : n;
}
}

MorphologyMatch::MorphologyMatch(const unsigned int& bit_context)
    : Model(1), bit_context_(bit_context) {}

bool MorphologyMatch::IsLiteral(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

unsigned char MorphologyMatch::Fold(unsigned char c) {
  return c >= 'A' && c <= 'Z' ? static_cast<unsigned char>(c + 32) : c;
}

std::uint64_t MorphologyMatch::Hash(std::uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  return x ^ (x >> 33);
}

std::uint64_t MorphologyMatch::ContextKey(int n) const {
  std::uint64_t x = 0x9e3779b97f4a7c15ULL ^ static_cast<unsigned int>(n);
  const int begin = suffix_size_ - n;
  for (int i = begin; i < suffix_size_; ++i) x = x * 257 + suffix_[i];
  // Preserve only a coarse word-position class so endings generalize across
  // previously unseen words instead of becoming exact-word contexts.
  x ^= static_cast<std::uint64_t>(std::min(suffix_size_, 7)) << 56;
  return Hash(x) | 1ULL;
}

bool MorphologyMatch::StillLive(int bpos) const {
  if (bpos == 0) return true;
  const unsigned int got = bit_context_ & ((1u << bpos) - 1u);
  return got == (static_cast<unsigned int>(expected_) >> (8 - bpos));
}

const std::valarray<float>& MorphologyMatch::Predict() {
  if (!active_) { outputs_[0] = 0.5f; return outputs_; }
  const int bpos = BitPos(bit_context_);
  if (!StillLive(bpos)) { outputs_[0] = 0.5f; return outputs_; }
  const bool one = ((expected_ >> (7 - bpos)) & 1) != 0;
  outputs_[0] = one ? confidence_ : 1.0f - confidence_;
  return outputs_;
}

void MorphologyMatch::Perceive(int) {}

void MorphologyMatch::ByteUpdate() {
  const unsigned char byte = static_cast<unsigned char>(bit_context_);
  for (int i = 0; i < pending_size_; ++i) {
    Entry& e = table_[pending_[i]];
    if (e.next == byte) {
      if (e.strength < 255) ++e.strength;
    } else if (e.strength > 0) {
      --e.strength;
    } else {
      e.next = byte;
      e.strength = 1;
    }
    if (e.seen != 0xffffffffu) ++e.seen;
  }

  if (!IsLiteral(byte)) {
    suffix_size_ = 0; pending_size_ = 0; active_ = false; return;
  }
  if (suffix_size_ == static_cast<int>(suffix_.size())) {
    for (int i = 1; i < suffix_size_; ++i) suffix_[i - 1] = suffix_[i];
    --suffix_size_;
  }
  suffix_[suffix_size_++] = Fold(byte);

  pending_size_ = 0;
  Entry* best = nullptr;
  int best_n = 0;
  for (int n = 2; n <= 5 && n <= suffix_size_; ++n) {
    const std::uint64_t key = ContextKey(n);
    const unsigned int slot = static_cast<unsigned int>(key) & kTableMask;
    Entry& e = table_[slot];
    if (e.key != key) {
      e = Entry{}; e.key = key;
    }
    pending_[pending_size_++] = slot;
    if (e.seen >= 3 && e.strength >= 3 && (!best || n > best_n)) {
      best = &e; best_n = n;
    }
  }
  active_ = best != nullptr;
  if (!active_) return;
  expected_ = best->next;
  const float learned = 0.5f + std::min(best->strength, std::uint8_t(96)) / 256.0f;
  confidence_ = std::min(0.875f, learned);
  const unsigned int sb = std::min<unsigned int>(best->strength >> 3, 7);
  context_ = static_cast<unsigned int>((best_n - 2) * 8) + sb;
}

unsigned long long MorphologyMatch::MixerContext() const {
  return active_ ? 1ULL + context_ : 0ULL;
}
