#ifndef FX4_TOKEN_NGRAM_BIAS_H
#define FX4_TOKEN_NGRAM_BIAS_H

#include <array>
#include <cstdint>
#include <memory>

// A bounded causal correction expert inspired by the useful part of
// Nacrith's adaptive n-gram bias. It has no model file, GPU path, or side
// data: encoder and decoder learn the same four byte-context tables online.
class TokenNgramBias {
 public:
  TokenNgramBias();

  float Predict(float base_probability, float base_logit);
  void Perceive(int bit);

 private:
  static constexpr unsigned int kOrders = 4;
  static constexpr unsigned int kTableBits = 15;
  static constexpr unsigned int kTableSize = 1u << kTableBits;
  static constexpr unsigned int kTableMask = kTableSize - 1u;
  static constexpr unsigned int kContexts = 128;

  struct Entry {
    std::uint64_t key = 0;
    std::uint16_t zero = 0;
    std::uint16_t one = 0;
  };

  std::uint64_t MakeKey(unsigned int order) const;
  static std::uint64_t Mix64(std::uint64_t value);

  std::array<std::unique_ptr<Entry[]>, kOrders> tables_;
  std::array<std::uint64_t, kOrders> active_keys_{};
  std::array<unsigned int, kOrders> active_indices_{};
  std::array<float, kOrders> inputs_{};
  std::array<std::array<float, kOrders>, kContexts> weights_{};
  std::array<float, kContexts> recent_error_{};
  std::array<std::uint8_t, kOrders> history_{};
  std::array<float, 65536> log_count_{};

  unsigned int history_size_ = 0;
  unsigned int bit_position_ = 0;
  unsigned int byte_prefix_ = 0;
  unsigned int context_ = 0;
  float probability_ = 0.5f;
};

#endif
