#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "fx4_config.h"
#include "mixer/sigmoid.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/byte-mixer.h"
#include "mixer/sse.h"
#include "models/model.h"
#include "models/byte-model.h"
#include "context-manager.h"
#include "models/direct.h"
#include "models/direct-hash.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#ifdef KH_OBIAS
#include "models/obias-prior.h"
#endif
#if FX4_SELECTIVE_POSTR1
#include "models/postr1_experts.h"
#endif
#include "mixer/lstm.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/sparse.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/bit-context.h"
#include "contexts/combined-context.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include <vector>
#include <set>
#include <memory>
#include <optional>
#include <array>
#include <cstdint>

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab, bool scr2_enabled = false);
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);
  void FreeFxcmMemory();
  void EnablePostR1Portfolio(std::uint32_t mask);
  void SetPostR1Span(std::uint64_t logical_offset, std::uint32_t mask,
      std::uint8_t stream_class, std::uint8_t profile_id,
      std::uint16_t mini_model_mask);
  void SetPostR1DonorProfile(const std::vector<std::uint8_t>& bytes,
      const std::vector<std::uint32_t>& segment_lengths);
  bool HasPostR1DonorProfile() const;
  float PpmdByteProbability(std::uint8_t byte) const;
  unsigned int PpmdEffectiveOrder() const;
  unsigned int PpmdEscapeDepth() const;
  float PpmdEscapeRate() const;
#if FX4_DONOR_FORK_DISCOVERY && FX4_SELECTIVE_POSTR1
  void SetPostR1BranchSignals(PostR1Experts* target) const;
#endif

#if defined(KH_TRACE) || defined(KH_BITLSTM32) || \
    FX4_RESIDUAL_ORACLE_TRACE || FX4_RESIDUAL_LSTM96
  // Terminal-head features staged by Predict() and consumed immediately by
  // Encoder::Encode/Decoder::Decode. Raw/known-bit paths never consume them.
  const float* kh_stage1_in_ = nullptr;
  int kh_stage1_n_ = 0;
  float kh_m1raw_ = 0.0f;
  int kh_override_ = 0;
#endif
#if FX4_RESIDUAL_ORACLE_TRACE || FX4_RESIDUAL_LSTM96
  float trace_ppmd_probability_ = 0.5f;
  float trace_lstm_probability_ = 0.5f;
  float trace_fxcm_probability_ = 0.5f;
  std::uint8_t trace_bit_position_ = 0;
  std::uint8_t trace_ppmd_order_ = 0;
  std::uint8_t trace_escape_depth_ = 0;
  std::uint8_t trace_match_length_ = 0;
  std::uint8_t trace_stream_class_ = 0;
#endif
 private:
  unsigned long long GetNumModels();
  void AddMixer(int layer, const unsigned long long& context,
      float learning_rate);
  void AddAuxiliary();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddMatch();
  void AddDoubleIndirect();
#if FX4_MINI_CMIX
  void AddMiniCmix();
  float PredictMiniCmix(std::uint16_t model_mask);
  void PerceiveMiniCmix(int bit, std::uint16_t model_mask);
  void ByteUpdateMiniCmix(std::uint16_t model_mask);
#endif
  void AddMixers();
#if FX4_SPECIALIST_CORRECTOR
  unsigned int SpecialistStreamClass() const;
  float PredictSpecialist(float base_probability,
      float ppmd_logit, float lstm_logit, float fxcm_logit);
  void PerceiveSpecialist(int bit);
#endif
#if FX4_SELECTIVE_POSTR1
  void UpdatePostR1ResidualDistribution();
  float PostR1ResidualProbability() const;
#endif
  llvm::SmallVector<Indirect<Nonstationary>, 32> indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_; // run map
  llvm::SmallVector<Direct, 4> direct_models_;
  llvm::SmallVector<Match, 10> match_models_;
#if FX4_MINI_CMIX
  llvm::SmallVector<Direct, 3> mini_direct_models_;
  llvm::SmallVector<DirectHash, 2> mini_direct_hash_models_;
  llvm::SmallVector<Indirect<Nonstationary>, 4> mini_indirect_models_;
  llvm::SmallVector<Match, 2> mini_match_models_;
  std::vector<unsigned char> mini_shared_map_;
  unsigned long long mini_longest_match_ = 0;

  static constexpr unsigned int kMiniCmixModelCount = 11;
  static constexpr unsigned int kMiniCmixFeatureCount =
      kMiniCmixModelCount + 1;
  static constexpr unsigned int kMiniCmixContextCount = 128;
  std::array<std::array<float, kMiniCmixFeatureCount>,
      kMiniCmixContextCount> mini_cmix_weights_{};
  std::array<float, kMiniCmixFeatureCount> mini_cmix_inputs_{};
  std::array<float, kMiniCmixModelCount> mini_cmix_model_probabilities_{};
  std::array<float, kMiniCmixContextCount> mini_cmix_recent_error_{};
  unsigned int mini_cmix_context_ = 0;
  float mini_cmix_probability_ = 0.5f;
  bool mini_cmix_used_ = false;
  bool mini_cmix_tracking_ = false;
  std::uint16_t mini_cmix_model_mask_ = 0;
#endif

  std::optional<Bracket> bracket_model_;
  size_t auxiliary_size_ = 3; // FXCM, LSTM, direct PPMd
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  llvm::SmallVector<Mixer, 24> mixer_0_;
  llvm::SmallVector<Mixer, 1> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  unsigned long long final_mixer_context_ = 0;
  Sigmoid sigmoid_;
  std::array<float, 4096> fxcm_stretched_inputs_;
  float fxcm_neutral_input_ = 0.0f;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<ByteMixer> byte_mixer_;
#ifdef KH_OBIAS
  std::unique_ptr<KhObiasPrior> obias_;
  bool obias_active_ = false;
  bool obias_keep_aux_ = false;
#endif
#if FX4_SHADOW_LSTM200
  std::optional<ByteMixer> shadow_lstm200_;
  float shadow_lstm200_output_ = 0.5f;
#endif
  std::vector<bool> vocab_;
  FXCM fxcm_model_;
#if FX4_SELECTIVE_POSTR1
  std::unique_ptr<PostR1Experts> postr1_experts_;
  std::uint32_t postr1_portfolio_mask_ = 0;
  bool postr1_prediction_used_ = false;
  std::array<float, 512> postr1_mass_{};
  std::array<float, 256> postr1_residual_one_{};
  unsigned int postr1_residual_gain_ = 0;
#if FX4_DONOR_FORK_DISCOVERY
  float donor_branch_ppmd_probability_ = 0.5f;
  float donor_branch_lstm_probability_ = 0.5f;
#if FX4_SHADOW_LSTM200
  float donor_branch_shadow_lstm_probability_ = 0.5f;
#endif
  float donor_branch_fxcm_probability_ = 0.5f;
  std::array<float, 4> donor_branch_ppmd_order_bands_{{
      0.5f, 0.5f, 0.5f, 0.5f}};
  unsigned int donor_branch_ppmd_order_ = 0;
  unsigned int donor_branch_escape_depth_ = 0;
  float donor_branch_escape_rate_ = 0.0f;
  float donor_branch_residual_probability_ = 0.5f;
  unsigned int donor_branch_match_length_ = 0;
#endif
#endif
#if FX4_SPECIALIST_CORRECTOR
  static constexpr unsigned int kSpecialistCoarseContexts = 64;
  static constexpr unsigned int kSpecialistContexts = 1024;
  static constexpr unsigned int kSpecialistFeatures = 5;
  std::array<std::array<float, kSpecialistFeatures>,
      kSpecialistContexts> specialist_weights_{};
  std::array<std::array<float, kSpecialistFeatures>,
      kSpecialistCoarseContexts> specialist_coarse_weights_{};
  std::array<float, kSpecialistContexts> specialist_error_{};
  std::array<float, kSpecialistFeatures> specialist_inputs_{};
  unsigned int specialist_coarse_context_ = 0;
  unsigned int specialist_context_ = 0;
  float specialist_probability_ = 0.5f;
#endif
};
#endif
