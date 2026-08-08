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
#include "models/indirect.h"
#include "models/match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
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
      std::uint8_t stream_class, std::uint8_t profile_id);
  void SetPostR1DonorProfile(const std::vector<std::uint8_t>& bytes,
      const std::vector<std::uint32_t>& segment_lengths);

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
  std::vector<bool> vocab_;
  FXCM fxcm_model_;
#if FX4_SELECTIVE_POSTR1
  std::unique_ptr<PostR1Experts> postr1_experts_;
  bool postr1_prediction_used_ = false;
  std::array<float, 512> postr1_mass_{};
  std::array<float, 256> postr1_residual_one_{};
  unsigned int postr1_residual_gain_ = 0;
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
