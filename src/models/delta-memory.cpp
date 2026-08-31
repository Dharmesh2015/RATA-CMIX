#include "delta-memory.h"

#include <algorithm>
#include <cmath>

namespace {

float Logistic(float value) {
  if (value >= 12.0f) return 0.99999386f;
  if (value <= -12.0f) return 0.00000614f;
  return 1.0f / (1.0f + std::exp(-value));
}

float Clamp(float value, float lo, float hi) {
  return std::max(lo, std::min(hi, value));
}

}  // namespace

DeltaMemory8::DeltaMemory8() {}

std::array<float, DeltaMemory8::kDim> DeltaMemory8::BuildKey(
    float base_logit, float ppmd_logit, float lstm_logit,
    float fxcm_logit) const {
  std::array<float, kDim> key{};
  // Residual signals (expert vs. current mix), same clamp convention as
  // TokenNgramBias's "expert_logit - base_logit" inputs.
  key[0] = Clamp(ppmd_logit - base_logit, -4.0f, 4.0f);
  key[1] = Clamp(lstm_logit - base_logit, -4.0f, 4.0f);
  key[2] = Clamp(fxcm_logit - base_logit, -4.0f, 4.0f);
  // Pairwise disagreement between the two richest experts.
  key[3] = Clamp(ppmd_logit - lstm_logit, -4.0f, 4.0f);
  key[4] = Clamp(ppmd_logit - fxcm_logit, -4.0f, 4.0f);
  // Position within the byte, centered to roughly [-1, 1].
  key[5] = (static_cast<float>(bit_position_) / 3.5f) - 1.0f;
  // This corrector's own recent miscalibration, already in [0, 1].
  key[6] = recent_error_;
  // Bias term.
  key[7] = 1.0f;
  return key;
}

float DeltaMemory8::Predict(float base_probability, float base_logit,
    float ppmd_logit, float lstm_logit, float fxcm_logit) {
  base_probability =
      Clamp(base_probability, 1.0e-5f, 1.0f - 1.0e-5f);
  const auto key = BuildKey(base_logit, ppmd_logit, lstm_logit, fxcm_logit);
  for (unsigned int i = 0; i < kDim; ++i) key_[i] = key[i];

  float raw = 0.0f;
  for (unsigned int i = 0; i < kDim; ++i) raw += weight_[i] * key_[i];
  const float correction = Clamp(raw, -2.0f, 2.0f);

  probability_ = Logistic(base_logit + correction);
  if (!(probability_ > 0.0f && probability_ < 1.0f)) {
    probability_ = base_probability;
  }
  return probability_;
}

void DeltaMemory8::Perceive(int bit) {
  const float error = static_cast<float>(bit) - probability_;
  // beta scales with recent miscalibration, same spirit as
  // TokenNgramBias's adaptive rate; lambda is a gentle leak so the state
  // cannot grow without bound between infrequent, large corrections.
  const float beta = 0.005f + 0.02f * recent_error_;
  constexpr float kLambda = 0.999f;
  for (unsigned int i = 0; i < kDim; ++i) {
    weight_[i] = Clamp(kLambda * weight_[i] + beta * error * key_[i],
        -2.0f, 2.0f);
  }
  recent_error_ = 0.995f * recent_error_ + 0.005f * std::fabs(error);

  if (++bit_position_ == 8u) bit_position_ = 0;
}
