#ifndef MORPHOLOGY_MATCH_H
#define MORPHOLOGY_MATCH_H

#include "model.h"

#include <array>
#include <cstdint>



// Learns the next literal byte from lower-cased suffixes of the current word.
// It never changes the stream and carries no side data.
class MorphologyMatch : public Model {
 public:
  explicit MorphologyMatch(const unsigned int& bit_context);
  const std::valarray<float>& Predict();
  void Perceive(int bit);
  void ByteUpdate();
  unsigned long long MixerContext() const;

 private:
  struct Entry {
    std::uint64_t key = 0;
    std::uint32_t seen = 0;
    std::uint8_t next = 0;
    std::uint8_t strength = 0;
  };
  static const unsigned int kTableBits = 17;
  static const unsigned int kTableSize = 1u << kTableBits;
  static const unsigned int kTableMask = kTableSize - 1;

  static bool IsLiteral(unsigned char c);
  static unsigned char Fold(unsigned char c);
  static std::uint64_t Hash(std::uint64_t x);
  std::uint64_t ContextKey(int suffix) const;
  bool StillLive(int bpos) const;

  const unsigned int& bit_context_;
  std::array<Entry, kTableSize> table_{};
  std::array<unsigned char, 8> suffix_{};
  std::array<std::uint32_t, 4> pending_{};
  int suffix_size_ = 0;
  int pending_size_ = 0;
  bool active_ = false;
  unsigned char expected_ = 0;
  unsigned int context_ = 0;
  float confidence_ = 0.5f;
};

#endif
