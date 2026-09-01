#include "esn-nlms.h"

#include <cmath>

float EsnNlmsExpert::Clamp(float value, float lo, float hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

float EsnNlmsExpert::FastTanh(float value) {
  // Deterministic bounded reservoir activation; unlike libm tanh this is a
  // handful of scalar operations and does not alter any primary model.
  value = Clamp(value, -8.0f, 8.0f);
  return value / (1.0f + std::fabs(value));
}

void EsnNlmsExpert::UpdateReservoir(
    const std::array<float, kUnits>& features) {
  const auto previous = state_;
  for (unsigned int i = 0; i < kUnits; ++i) {
    // Fixed sparse recurrent graph plus a deterministic signed projection of
    // the current model-disagreement features. The coefficients are small
    // enough to keep the reservoir in its responsive, non-saturated range.
    float sum = 0.46f * previous[i]
        + 0.23f * previous[(i + 3u) & 7u]
        - 0.17f * previous[(i + 5u) & 7u];
    for (unsigned int j = 0; j < kUnits; ++j) {
      const unsigned int code = (i * 13u + j * 7u + 3u) & 7u;
      const float coefficient =
          code < 3u ? -0.125f : (code < 6u ? 0.125f : 0.25f);
      sum += coefficient * features[j];
    }
    state_[i] = FastTanh(sum);
  }
}

float EsnNlmsExpert::Predict(float base_probability, float base_logit,
    float ppmd_logit, float byte_model_logit, float fxcm_logit,
    unsigned int bit_position, unsigned int stream_class,
    unsigned int ppmd_order, unsigned int match_length) {
  base_probability_ = Clamp(base_probability, 1.0e-5f, 1.0f - 1.0e-5f);

  std::array<float, kUnits> features{{
      Clamp((ppmd_logit - base_logit) * 0.25f, -1.0f, 1.0f),
      Clamp((byte_model_logit - base_logit) * 0.25f, -1.0f, 1.0f),
      Clamp((fxcm_logit - base_logit) * 0.25f, -1.0f, 1.0f),
      Clamp((ppmd_logit - byte_model_logit) * 0.25f, -1.0f, 1.0f),
      static_cast<float>(bit_position & 7u) / 3.5f - 1.0f,
      static_cast<float>(stream_class & 7u) / 3.5f - 1.0f,
      Clamp(static_cast<float>(ppmd_order) / 12.0f - 1.0f,
          -1.0f, 1.0f),
      Clamp(static_cast<float>(match_length) / 32.0f - 1.0f
          + (recent_error_ - 0.5f), -1.0f, 1.0f)
  }};
  UpdateReservoir(features);

  float correction = 0.0f;
  for (unsigned int i = 0; i < kUnits; ++i) {
    correction += readout_[i] * state_[i];
  }
  correction = Clamp(correction, -2.0f, 2.0f);
  // First-order logit correction. This keeps the ESN in residual space
  // without an exp() in the per-bit path; the Bayesian outer mixture below
  // supplies the strict baseline loss bound.
  expert_probability_ = Clamp(base_probability_ +
      correction * base_probability_ * (1.0f - base_probability_),
      1.0e-5f, 1.0f - 1.0e-5f);

  return Clamp(base_probability_ + static_cast<float>(expert_weight_) *
      (expert_probability_ - base_probability_),
      1.0e-5f, 1.0f - 1.0e-5f);
}

void EsnNlmsExpert::Perceive(int bit) {
  const float target = bit ? 1.0f : 0.0f;
  const float error = target - expert_probability_;
  float norm = 0.25f;
  for (float value : state_) norm += value * value;
  const float step = 0.035f * error / norm;
  for (unsigned int i = 0; i < kUnits; ++i) {
    readout_[i] = Clamp(readout_[i] + step * state_[i], -2.0f, 2.0f);
  }

  const float base_likelihood =
      bit ? base_probability_ : 1.0f - base_probability_;
  const float expert_likelihood =
      bit ? expert_probability_ : 1.0f - expert_probability_;
  const double mixed_likelihood =
      (1.0 - expert_weight_) * base_likelihood +
      expert_weight_ * expert_likelihood;
  if (mixed_likelihood > 0.0) {
    expert_weight_ =
        expert_weight_ * expert_likelihood / mixed_likelihood;
  }
  if (--bits_until_gate_reset_ == 0) {
    expert_weight_ = 0.01;
    bits_until_gate_reset_ = kGateResetBits;
  }
  recent_error_ =
      0.995f * recent_error_ + 0.005f * std::fabs(target - base_probability_);
}
