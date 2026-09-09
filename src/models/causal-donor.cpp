#include "causal-donor.h"

#include <algorithm>

namespace {
int BitPos(unsigned int c0) {
  int n = 0;
  while (c0 > 1u) { c0 >>= 1; ++n; }
  return n > 7 ? 7 : n;
}
}

CausalDonor::CausalDonor(const unsigned int& bit_context)
    : Model(1), bit_context_(bit_context), history_(kHistorySize),
      buckets_(kBucketCount) {
  confidence_.fill(0.5f);
}

std::uint64_t CausalDonor::ContextKey() const {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (int i = 7; i >= 0; --i) {
    h ^= history_[(position_ - i) & kHistoryMask];
    h *= 0x100000001b3ULL;
  }
  h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
  return (h ^ (h >> 33)) | 1ULL;
}

int CausalDonor::BackwardLength(std::uint32_t candidate) const {
  int len = 8;
  while (len < 64 && candidate >= static_cast<unsigned int>(len) &&
         position_ >= static_cast<unsigned int>(len) &&
         history_[(candidate - len) & kHistoryMask] ==
             history_[(position_ - len) & kHistoryMask]) {
    ++len;
  }
  return len;
}

bool CausalDonor::StillLive(int bpos) const {
  if (bpos == 0) return true;
  const unsigned int got = bit_context_ & ((1u << bpos) - 1u);
  return got == (static_cast<unsigned int>(expected_) >> (8 - bpos));
}

const std::valarray<float>& CausalDonor::Predict() {
  if (!active_) { outputs_[0] = 0.5f; return outputs_; }
  const int bpos = BitPos(bit_context_);
  if (!StillLive(bpos)) { outputs_[0] = 0.5f; return outputs_; }
  const bool one = ((expected_ >> (7 - bpos)) & 1) != 0;
  const float c = confidence_[state_];
  outputs_[0] = one ? c : 1.0f - c;
  return outputs_;
}

void CausalDonor::Perceive(int bit) {
  if (!active_) return;
  const int bpos = BitPos(bit_context_);
  if (!StillLive(bpos)) return;
  const int want = (expected_ >> (7 - bpos)) & 1;
  confidence_[state_] += 0.01f * ((bit == want ? 1.0f : 0.0f) -
                                  confidence_[state_]);
  confidence_[state_] = std::max(0.01f, std::min(0.99f, confidence_[state_]));
}

void CausalDonor::ByteUpdate() {
  const unsigned char byte = static_cast<unsigned char>(bit_context_);
  history_[position_ & kHistoryMask] = byte;
  active_ = false;

  if (position_ >= 7) {
    const std::uint64_t key = ContextKey();
    Bucket& bucket = buckets_[static_cast<unsigned int>(key) & kBucketMask];
    if (bucket.key == key) {
      std::array<unsigned char, kWays> value{};
      std::array<int, kWays> score{};
      std::array<int, kWays> votes{};
      int used = 0;
      for (int i = 0; i < kWays; ++i) {
        const std::uint32_t candidate = bucket.pos[i];
        if (candidate == 0xffffffffu || candidate + 1 >= position_ ||
            position_ - candidate >= kHistorySize) continue;
        bool equal = true;
        for (int j = 0; j < 8; ++j) {
          if (history_[(candidate - j) & kHistoryMask] !=
              history_[(position_ - j) & kHistoryMask]) { equal = false; break; }
        }
        if (!equal) continue;
        const unsigned char next = history_[(candidate + 1) & kHistoryMask];
        const int length = BackwardLength(candidate);
        int v = 0;
        while (v < used && value[v] != next) ++v;
        if (v == used) { value[used] = next; ++used; }
        score[v] += length;
        ++votes[v];
      }
      int best = -1;
      bool tie = false;
      for (int i = 0; i < used; ++i) {
        if (best < 0 || score[i] > score[best]) { best = i; tie = false; }
        else if (score[i] == score[best]) tie = true;
      }
      if (best >= 0 && !tie) {
        expected_ = value[best];
        const int lb = std::min(score[best] >> 4, 7);
        const int vb = std::min(votes[best] - 1, 3);
        state_ = lb * 4 + vb;
        active_ = true;
      }
    } else {
      bucket = Bucket{};
      bucket.key = key;
    }
    for (int i = kWays - 1; i > 0; --i) bucket.pos[i] = bucket.pos[i - 1];
    bucket.pos[0] = static_cast<std::uint32_t>(position_);
  }
  ++position_;
}

unsigned long long CausalDonor::MixerContext() const {
  return active_ ? 1ULL + static_cast<unsigned int>(state_) : 0ULL;
}
