#ifndef CAUSAL_DONOR_H
#define CAUSAL_DONOR_H

#include "model.h"

#include <array>
#include <cstdint>
#include <vector>



// Multi-reference episodic match expert. Donors are always earlier decoded
// bytes; the expert emits a probability and never replays donor bytes into the
// primary predictor, so a rejected prediction cannot disturb later state.
class CausalDonor : public Model {
 public:
  explicit CausalDonor(const unsigned int& bit_context);
  const std::valarray<float>& Predict();
  void Perceive(int bit);
  void ByteUpdate();
  unsigned long long MixerContext() const;

 private:
  static const unsigned int kHistoryBits = 24;
  static const unsigned int kHistorySize = 1u << kHistoryBits;
  static const unsigned int kHistoryMask = kHistorySize - 1;
  static const unsigned int kBucketBits = 18;
  static const unsigned int kBucketCount = 1u << kBucketBits;
  static const unsigned int kBucketMask = kBucketCount - 1;
  static const int kWays = 4;
  static const int kStates = 32;

  struct Bucket {
    std::uint64_t key = 0;
    std::array<std::uint32_t, kWays> pos{{0xffffffffu, 0xffffffffu,
                                         0xffffffffu, 0xffffffffu}};
  };

  std::uint64_t ContextKey() const;
  int BackwardLength(std::uint32_t candidate) const;
  bool StillLive(int bpos) const;

  const unsigned int& bit_context_;
  std::vector<unsigned char> history_;
  std::vector<Bucket> buckets_;
  std::array<float, kStates> confidence_{};
  std::uint64_t position_ = 0;
  bool active_ = false;
  unsigned char expected_ = 0;
  int state_ = 0;
};

#endif
