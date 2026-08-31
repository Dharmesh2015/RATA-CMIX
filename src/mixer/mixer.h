#ifndef MIXER_H
#define MIXER_H

#include <vector>
#include "../ds/emhash_map.hpp"
#include "mixer-input.h"
#include <memory>
#include <cstdint>
#include <cstdlib>

class Mixer {
 public:
  Mixer(const MixerInput& layer, const unsigned long long& context,
      float learning_rate, unsigned int extra_input_size);
  // Phase-5 two-pass protocol: the mixer's context is fixed for the whole
  // bit by the time Predictor::Predict() reaches the mixer block, so the
  // hash lookup runs once per bit in BeginBit() (instead of once in Mix and
  // again in Perceive) and the cold weight block is prefetched there.
  void BeginBit();
  float Mix();
  void Perceive(int bit);

 private:
  float* FindSlot();

  // Flat 64-byte-aligned input row, captured once (the layer's row is
  // allocated before any mixer is constructed and never reallocated).
  const float* inputs_;
  // The extra-input row is sized after the mixers are built, so it is
  // fetched from the layer per use instead of captured here.
  const MixerInput& layer_;
  uint16_t inputs_size_;
  uint16_t extra_inputs_size_;
  float p_, learning_rate_;
  const unsigned long long& context_;
  unsigned long long steps_;
  // Per-context weights live in one flat 64-byte-aligned slab: slot_floats_
  // per context, the main weights immediately followed by the extra weights.
  // Slot 0 is the shared fallback used once the map hits its size limit (the
  // old context_base_). The map stores float offsets into the slab, so the
  // value stays tiny (denser probing than the old two-valarray ContextData)
  // and the dot product reads one contiguous aligned block with no pointer
  // chase. Slab capacity covers the map limit, so it never reallocates and
  // cached pointers stay valid for the whole bit.
  struct FreeDeleter { void operator()(void* p) const { std::free(p); } };
  std::unique_ptr<float[], FreeDeleter> slab_;
  uint32_t slot_floats_;
  uint32_t slots_used_;
  float* weights_;        // cached by BeginBit() for the current bit
  float* extra_weights_;
  emhash6::HashMap<unsigned int, uint32_t> context_map_;
};

#endif
