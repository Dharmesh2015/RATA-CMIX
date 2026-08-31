#include "token-ngram-bias.h"

#include <algorithm>
#include <cmath>

namespace {

float Logistic(float value) {
  if (value >= 12.0f) return 0.99999386f;
  if (value <= -12.0f) return 0.00000614f;
  return 1.0f / (1.0f + std::exp(-value));
}

}  // namespace

TokenNgramBias::TokenNgramBias() {
  for (auto& table : tables_) table.reset(new Entry[kTableSize]());
  for (unsigned int i = 0; i < log_count_.size(); ++i) {
    log_count_[i] = std::log(static_cast<float>(i) + 0.5f);
  }
}

std::uint64_t TokenNgramBias::Mix64(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31;
  return value;
}

std::uint64_t TokenNgramBias::MakeKey(unsigned int order) const {
  if (history_size_ < order) return 0;
  std::uint64_t value = 0x9e3779b97f4a7c15ull ^
      (static_cast<std::uint64_t>(bit_position_) << 56) ^
      (static_cast<std::uint64_t>(byte_prefix_) << 40) ^ order;
  for (unsigned int i = 0; i < order; ++i) {
    value = Mix64(value ^ (static_cast<std::uint64_t>(history_[i]) <<
        ((i * 11u) & 31u)) ^ (0x100000001b3ull * (i + 1u)));
  }
  return Mix64(value) | 1ull;
}

float TokenNgramBias::Predict(float base_probability, float base_logit) {
  base_probability = std::max(1.0e-5f,
      std::min(1.0f - 1.0e-5f, base_probability));
  const float confidence = std::fabs(base_logit);
  const unsigned int confidence_bucket =
      (confidence >= 1.0f) + (confidence >= 2.0f) +
      (confidence >= 4.0f);
  context_ = ((bit_position_ & 7u) << 4) |
      (confidence_bucket << 2) | (byte_prefix_ & 3u);

  float correction = 0.0f;
  for (unsigned int order = 0; order < kOrders; ++order) {
    const std::uint64_t key = MakeKey(order + 1u);
    active_keys_[order] = key;
    active_indices_[order] = static_cast<unsigned int>(key) & kTableMask;
    inputs_[order] = 0.0f;
    if (key == 0) continue;
    const Entry& entry = tables_[order][active_indices_[order]];
    const unsigned int total = entry.zero + entry.one;
    if (entry.key != key || total < 8u) continue;
    const float expert_logit =
        log_count_[entry.one] - log_count_[entry.zero];
    inputs_[order] = std::max(-4.0f,
        std::min(4.0f, expert_logit - base_logit));
    correction += weights_[context_][order] * inputs_[order];
  }
  correction = std::max(-2.0f, std::min(2.0f, correction));
  probability_ = Logistic(base_logit + correction);
  if (!(probability_ > 0.0f && probability_ < 1.0f)) {
    probability_ = base_probability;
  }
  return probability_;
}

void TokenNgramBias::Perceive(int bit) {
  const float error = static_cast<float>(bit) - probability_;
  const float rate = 0.00035f + 0.00065f * recent_error_[context_];
  for (unsigned int order = 0; order < kOrders; ++order) {
    float& weight = weights_[context_][order];
    weight = std::max(-0.5f, std::min(0.5f,
        weight + rate * error * inputs_[order]));

    if (active_keys_[order] == 0) continue;
    Entry& entry = tables_[order][active_indices_[order]];
    if (entry.key != active_keys_[order]) {
      entry.key = active_keys_[order];
      entry.zero = 0;
      entry.one = 0;
    }
    if (static_cast<unsigned int>(entry.zero) + entry.one >= 65500u) {
      entry.zero = static_cast<std::uint16_t>((entry.zero + 1u) >> 1);
      entry.one = static_cast<std::uint16_t>((entry.one + 1u) >> 1);
    }
    std::uint16_t& count = bit ? entry.one : entry.zero;
    if (count != 65535u) ++count;
  }
  recent_error_[context_] = 0.995f * recent_error_[context_] +
      0.005f * std::fabs(error);

  byte_prefix_ = (byte_prefix_ << 1) | static_cast<unsigned int>(bit != 0);
  if (++bit_position_ == 8u) {
    for (unsigned int i = kOrders - 1; i > 0; --i) {
      history_[i] = history_[i - 1];
    }
    history_[0] = static_cast<std::uint8_t>(byte_prefix_);
    history_size_ = std::min(history_size_ + 1u, kOrders);
    bit_position_ = 0;
    byte_prefix_ = 0;
  }
}
