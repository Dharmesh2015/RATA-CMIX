#include "mixer.h"

#include "sigmoid.h"
#include "../utils/hugepage.h"

#include <numeric>
#include <utility>
#include <cstring>
#include <math.h>
#include <immintrin.h>
#include <sys/resource.h>

namespace {
// Same cap as the old GetContextData(): at most this many distinct contexts
// get their own weight slot; everything past that shares slot 0.
constexpr uint32_t kContextLimit = 10000;

#ifdef FX4_MIXER_WINIT
// Ported from fx-deepmix (Halvor Yttredal): stock cmix zero-initializes
// every fresh context row, so a brand-new context starts at p=0.5 ignoring
// every input and must relearn the roster average from scratch. Seeding
// each weight to FX4_MIXER_WINIT (e.g. ~1/inputs_size_) instead starts a
// fresh context near the roster average. Online-adaptive only: this changes
// the initial value a never-before-seen context's weights start at, not
// whether/how they keep adapting; encoder and decoder compute the identical
// seed since it's a compile-time constant, not a learned/offline value.
constexpr float kMixerWinit = (float)(FX4_MIXER_WINIT);
inline void InitSlot(float* slot, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) slot[i] = kMixerWinit;
}
#else
inline void InitSlot(float* slot, uint32_t n) {
  std::memset(slot, 0, n * sizeof(float));
}
#endif
}

Mixer::Mixer(const MixerInput& layer, const unsigned long long& context,
    float learning_rate, unsigned int extra_input_size) :
    inputs_(layer.Inputs()), layer_(layer), inputs_size_(layer.NumInputs()),
    extra_inputs_size_(extra_input_size), p_(0.5),
    learning_rate_(learning_rate), context_(context), steps_(0) {
  // Whole cache lines per slot so every context's block starts 64B-aligned.
  slot_floats_ = (inputs_size_ + extra_inputs_size_ + 15u) & ~15u;
  const size_t slab_bytes =
      (size_t)(kContextLimit + 1) * slot_floats_ * sizeof(float);
  slab_.reset(static_cast<float*>(std::aligned_alloc(64, slab_bytes)));
  AdviseHugePages(slab_.get(), slab_bytes);
  // Only allocated slots are ever memset/touched, so physical memory use
  // matches the old on-demand valarray allocations.
  InitSlot(slab_.get(), slot_floats_);  // shared slot 0
  slots_used_ = 1;
  weights_ = slab_.get();
  extra_weights_ = slab_.get() + inputs_size_;
  context_map_.reserve(kContextLimit);
}

float* Mixer::FindSlot() {
  auto it = context_map_.find(context_);
  if (it != context_map_.end()) return slab_.get() + it->second;
  if (context_map_.size() >= kContextLimit) return slab_.get();
  const uint32_t offset = slots_used_ * slot_floats_;
  ++slots_used_;
  float* slot = slab_.get() + offset;
  InitSlot(slot, slot_floats_);  // zero-init (or FX4_MIXER_WINIT-seeded)
  context_map_.insert({static_cast<unsigned int>(context_), offset});
  return slot;
}

void Mixer::BeginBit() {
  float* slot = FindSlot();
  weights_ = slot;
  extra_weights_ = slot + inputs_size_;
  // Kick off the cold block's first lines; with every mixer doing this ahead
  // of the Mix pass the independent DRAM miss chains overlap instead of
  // serializing inside each dot product. The hardware stream prefetcher
  // follows the rest of the sequential block.
  const char* base = reinterpret_cast<const char*>(slot);
  for (int i = 0; i < 8; ++i) {
    _mm_prefetch(base + 64 * i, _MM_HINT_T0);
  }
  if (extra_inputs_size_ != 0) {
    _mm_prefetch(reinterpret_cast<const char*>(slot + inputs_size_),
                 _MM_HINT_T0);
  }
}

float Mixer::Mix() {
  // Same accumulation order as always; weights_ was resolved by BeginBit(),
  // so the per-bit hash lookup and the valarray pointer chase are gone.
  const float* const inputs = inputs_;
  const float* const weights = weights_;
  float p = 0;
  for (unsigned int i = 0; i < inputs_size_; ++i) {
    p += inputs[i] * weights[i];
  }
  p_ = p;
  if (extra_inputs_size_ != 0) {
    const float* const extra_inputs = layer_.ExtraInputs();
    const float* const extra_weights = extra_weights_;
    float e = 0;
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      e += extra_inputs[i] * extra_weights[i];
    }
    p_ += e;
  }
  return p_;
}

void Mixer::Perceive(int bit) {
  float decay=0.2f;
  if ( steps_ < 25000000) {
      decay = 0.3f;
      if ( steps_ < 5000000) {
          decay = 0.7f;
          if ( steps_ < 1000000)
              decay = 1.0f;
      }
  }
  ++steps_;

  float update =   learning_rate_ * (Sigmoid::Logistic(p_) - bit);
  if(fabsf(update)<0.000000000005f && extra_inputs_size_>0) {
      return;
  }
  update = decay * update;
  // weights_/extra_weights_ still point at this bit's slot (cached by
  // BeginBit()), so the old second hash lookup per bit is gone.
  float* const weights = weights_;
  const float* const inputs = inputs_;
  for (unsigned int i = 0; i < inputs_size_; ++i) {
    weights[i] -= update * inputs[i];
  }
  if (extra_inputs_size_ != 0) {
    float* const extra_weights = extra_weights_;
    const float* const extra_inputs = layer_.ExtraInputs();
    for (unsigned int i = 0; i < extra_inputs_size_; ++i) {
      extra_weights[i] -= update * extra_inputs[i];
    }
  }
}
