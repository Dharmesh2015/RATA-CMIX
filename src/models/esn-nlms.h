#ifndef FX4_ESN_NLMS_H
#define FX4_ESN_NLMS_H

#include <array>

// Small causal echo-state residual expert with a normalized LMS readout.
//
// The reservoir is fixed; only its eight output weights learn online. A
// Bayesian two-expert mixture blends the corrected probability with the
// incoming baseline. The gate alone restarts with a 1% expert prior at each
// deterministic 1 MiB logical boundary; reservoir and NLMS training remain
// continuous. Each interval is bounded by -log2(0.99) relative to baseline,
// or about one byte total over the canonical 587 MB stream.
// No trained asset, plan, or side data is required: compression and
// decompression recreate all state from the same already-coded bits.
class EsnNlmsExpert {
 public:
  float Predict(float base_probability, float base_logit, float ppmd_logit,
      float byte_model_logit, float fxcm_logit, unsigned int bit_position,
      unsigned int stream_class, unsigned int ppmd_order,
      unsigned int match_length);
  void Perceive(int bit);

 private:
  static constexpr unsigned int kUnits = 8;
  static constexpr unsigned int kGateResetBits = 8u * 1048576u;

  static float Clamp(float value, float lo, float hi);
  static float FastTanh(float value);
  void UpdateReservoir(const std::array<float, kUnits>& features);

  std::array<float, kUnits> state_{};
  std::array<float, kUnits> readout_{};
  float base_probability_ = 0.5f;
  float expert_probability_ = 0.5f;
  float recent_error_ = 0.5f;
  // Exact Bayesian posterior weight, initialized with a 1% expert prior.
  double expert_weight_ = 0.01;
  unsigned int bits_until_gate_reset_ = kGateResetBits;
};

#endif
