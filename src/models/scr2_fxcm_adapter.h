#ifndef SCR2_FXCM_ADAPTER_H
#define SCR2_FXCM_ADAPTER_H

#include "scr2_tokens.h"

#include <array>
#include <cstdint>
#include <cstdlib>

namespace scr2 {

class FxcmAdapter {
 public:
  static constexpr std::uint32_t kTableBits = 15;
  static constexpr std::uint32_t kTableSize = 1u << kTableBits;
  static constexpr std::uint32_t kTableMask = kTableSize - 1;
  static constexpr unsigned int kOutputs = 3;

  struct Counter {
    std::uint16_t n0;
    std::uint16_t n1;
  };

  FxcmAdapter() = default;
  FxcmAdapter(const FxcmAdapter&) = delete;
  FxcmAdapter& operator=(const FxcmAdapter&) = delete;
  ~FxcmAdapter() { Free(); }

  void Init() {
    if (initialized_) return;
    for (unsigned int i = 0; i < kOutputs; ++i) {
      tables_[i] = static_cast<Counter*>(
          std::calloc(kTableSize, sizeof(Counter)));
      if (!tables_[i]) std::abort();
    }
    initialized_ = true;
    Reset();
  }

  void Free() {
    for (unsigned int i = 0; i < kOutputs; ++i) {
      std::free(tables_[i]);
      tables_[i] = nullptr;
    }
    initialized_ = false;
  }

  void Reset() {
    pending_marker_ = false;
    last_token_ = 0;
    previous_token_ = 0;
    token_age_ = 255;
    token_class_ = 0;
    last_raw_ = 0;
    line_hash_ = 2166136261u;
    virtual_hash_ = 0;
    virtual_word_hash_ = 0;
    virtual_suffix_hash_ = 0;
    predicted_active_.fill(false);
    predicted_index_.fill(0);
  }

  void OnByte(std::uint8_t byte) {
    line_hash_ = Mix(line_hash_, byte);
    if (byte == '\n') line_hash_ = 2166136261u;

    if (pending_marker_) {
      pending_marker_ = false;
      if (byte == 0) {
        AgeToken();
      } else if (byte <= kTokenCount) {
        previous_token_ = last_token_;
        last_token_ = byte;
        token_age_ = 0;
        const TokenInfo& info = kTokens[byte];
        token_class_ = info.cls;
        virtual_hash_ = info.hash ^ RotateLeft(info.prefix_hash, 7) ^
            RotateLeft(info.suffix_hash, 17) ^ info.length;
        virtual_word_hash_ = info.word_hash;
        virtual_suffix_hash_ = info.suffix_hash;
      } else {
        AgeToken();
      }
    } else if (byte == kMarker) {
      pending_marker_ = true;
    } else {
      AgeToken();
    }
    last_raw_ = byte;
  }

  std::array<float, kOutputs> Predict(std::uint32_t partial_byte,
      std::uint32_t bit_position, std::uint32_t stream2,
      std::uint32_t stream3, std::uint32_t structural_context) {
    std::array<float, kOutputs> result{{0.5f, 0.5f, 0.5f}};
    if (!initialized_) return result;

    const std::uint32_t phase = pending_marker_ ? 1u : 0u;
    const std::uint32_t key0 = Hash(0x13579BDFu, partial_byte,
        bit_position, phase | (std::uint32_t(last_token_) << 8),
        structural_context ^ line_hash_ ^ last_raw_);
    result[0] = Lookup(0, key0, true);

    const bool recent = pending_marker_ || token_age_ < 16;
    const std::uint32_t key1 = Hash(0x2468ACE1u, partial_byte,
        bit_position, virtual_hash_ ^ RotateLeft(virtual_word_hash_, 11),
        (std::uint32_t(token_class_) << 24) |
            (std::uint32_t(token_age_) << 16) | (stream2 & 0xffffu));
    result[1] = Lookup(1, key1, recent);

    const std::uint32_t key2 = Hash(0xB7E15163u, partial_byte,
        bit_position,
        (std::uint32_t(previous_token_) << 24) |
            (std::uint32_t(last_token_) << 16) |
            (std::uint32_t(token_class_) << 8) | token_age_,
        virtual_suffix_hash_ ^ stream3 ^ RotateLeft(stream2, 9));
    result[2] = Lookup(2, key2, recent);
    return result;
  }

  void Update(int bit) {
    if (!initialized_) return;
    for (unsigned int i = 0; i < kOutputs; ++i) {
      if (!predicted_active_[i]) continue;
      Counter& counter = tables_[i][predicted_index_[i]];
      if (bit) {
        if (counter.n1 != 65535u) ++counter.n1;
      } else {
        if (counter.n0 != 65535u) ++counter.n0;
      }
      const std::uint32_t total =
          std::uint32_t(counter.n0) + counter.n1;
      if (total > 4096u) {
        counter.n0 = std::uint16_t((counter.n0 + 1u) >> 1);
        counter.n1 = std::uint16_t((counter.n1 + 1u) >> 1);
      }
    }
  }

 private:
  static std::uint32_t RotateLeft(std::uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
  }

  static std::uint32_t Mix(std::uint32_t hash, std::uint32_t value) {
    hash ^= value + 0x9E3779B9u + (hash << 6) + (hash >> 2);
    hash *= 0x85EBCA6Bu;
    return hash ^ (hash >> 13);
  }

  static std::uint32_t Hash(std::uint32_t seed, std::uint32_t a,
      std::uint32_t b, std::uint32_t c, std::uint32_t d) {
    std::uint32_t hash = Mix(seed, a);
    hash = Mix(hash, b);
    hash = Mix(hash, c);
    return Mix(hash, d);
  }

  float Lookup(unsigned int table, std::uint32_t key, bool active) {
    predicted_active_[table] = active;
    if (!active) return 0.5f;
    const std::uint32_t index = key & kTableMask;
    predicted_index_[table] = index;
    const Counter& counter = tables_[table][index];
    const std::uint32_t probability =
        ((std::uint32_t(counter.n1) + 1u) * 4096u) /
        (std::uint32_t(counter.n0) + counter.n1 + 2u);
    const std::uint32_t bounded =
        probability < 1u ? 1u : (probability > 4095u ? 4095u : probability);
    return static_cast<float>(bounded) / 4096.0f;
  }

  void AgeToken() {
    if (token_age_ < 255u) ++token_age_;
    if (token_age_ >= 32u) {
      token_class_ = 0;
      virtual_hash_ = 0;
      virtual_word_hash_ = 0;
      virtual_suffix_hash_ = 0;
    }
  }

  Counter* tables_[kOutputs]{};
  std::array<std::uint32_t, kOutputs> predicted_index_{{0, 0, 0}};
  std::array<bool, kOutputs> predicted_active_{{false, false, false}};
  bool initialized_ = false;
  bool pending_marker_ = false;
  std::uint8_t last_token_ = 0;
  std::uint8_t previous_token_ = 0;
  std::uint8_t token_age_ = 255;
  std::uint8_t token_class_ = 0;
  std::uint8_t last_raw_ = 0;
  std::uint32_t line_hash_ = 2166136261u;
  std::uint32_t virtual_hash_ = 0;
  std::uint32_t virtual_word_hash_ = 0;
  std::uint32_t virtual_suffix_hash_ = 0;
};

}  // namespace scr2

#endif
