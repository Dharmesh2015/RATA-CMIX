#include "mixer-input.h"

namespace {
float* AlignedFloatRow(size_t count, float fill) {
  size_t bytes = (count * sizeof(float) + 63u) & ~size_t(63);
  if (bytes == 0) bytes = 64;
  float* row = static_cast<float*>(std::aligned_alloc(64, bytes));
  for (size_t i = 0; i < count; ++i) row[i] = fill;
  return row;
}
}  // namespace

MixerInput::MixerInput(const Sigmoid& sigmoid, float eps) :
    logit_table_(sigmoid.Table()), logit_size_(sigmoid.TableSize()),
    min_(eps), max_(1 - eps),
    stretched_min_(sigmoid.Logit(0)), stretched_max_(sigmoid.Logit(1)) {}

void MixerInput::SetNumModels(int num_models) {
  // Same 0.5 fill the old valarray resize used.
  inputs_.reset(AlignedFloatRow(num_models, 0.5f));
  num_models_ = num_models;
}

void MixerInput::SetExtraInputSize(size_t size) {
  extra_inputs_.reset(AlignedFloatRow(size, 0.0f));
}
