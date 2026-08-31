#ifndef FX4_DELTA_MEMORY_H
#define FX4_DELTA_MEMORY_H

#include <array>

// A tiny online associative correction expert, in the spirit of the delta
// rule used by fast-weight / linear-attention memories (e.g. arXiv 2605.12357
// section on delta-rule state updates: S <- lambda*S + beta*(v - S.k) k^T).
// That paper's S is a matrix mapping keys to *vector* values (sized for
// retrieving rich content); this corrector only ever needs a *scalar*
// output (one logit correction per bit), which is the d_value = 1
// specialization of the same rule: S is an 8-dim weight vector, not a
// matrix, and the update reduces to a small leaky least-mean-squares
// adaptive filter. No offline training, no model file: encoder and decoder
// learn the same 8 floats online, purely from bits already coded.
//
// This is deliberately complementary to TokenNgramBias, not a duplicate:
// TokenNgramBias corrects from discrete exact-match n-gram hash lookups;
// this corrects from continuous, dense features (the existing predictors'
// logits and their disagreement), so it can generalize between contexts
// that never hash-match but are numerically similar.
class DeltaMemory8 {
 public:
  DeltaMemory8();

  // ppmd_logit/lstm_logit/fxcm_logit are the same per-bit logits already
  // computed at the call site in Predictor::Predict() (ppmd_model_index,
  // byte_mixer_index, aggregate_fxcm_logit).
  float Predict(float base_probability, float base_logit, float ppmd_logit,
      float lstm_logit, float fxcm_logit);
  void Perceive(int bit);

 private:
  static constexpr unsigned int kDim = 8;

  std::array<float, kDim> BuildKey(float base_logit, float ppmd_logit,
      float lstm_logit, float fxcm_logit) const;

  float weight_[kDim] = {};
  float key_[kDim] = {};
  float probability_ = 0.5f;
  float recent_error_ = 0.0f;
  unsigned int bit_position_ = 0;
};

#endif
