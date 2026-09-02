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
#include "models/grammar-match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#include "models/esn-nlms.h"
#include "third_party/fx2_transformer/opt/model_opt.h"
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

#include <array>
#include <memory>
#include <optional>
#include <vector>

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab,
      bool enable_transformer6m = false);
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);
  void FreeFxcmMemory();

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
  void InitializeTransformer6m(bool required);
  void Transformer6mByteUpdate();

  unsigned int SpecialistStreamClass() const;
  float PredictSpecialist(float base_probability,
      float ppmd_logit, float transformer_logit, float fxcm_logit);
  void PerceiveSpecialist(int bit);

  llvm::SmallVector<Indirect<Nonstationary>, 32> indirect_ns_models_;
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_;
  llvm::SmallVector<Direct, 4> direct_models_;
  llvm::SmallVector<Match, 10> match_models_;
  std::optional<GrammarMatch> grammar_model_;

  std::optional<Bracket> bracket_model_;
  size_t auxiliary_size_ = 3;  // FXCM, transformer/LSTM, direct PPMd.
  SSE sse_;
  llvm::SmallVector<MixerInput, 2> layers_;
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
  std::optional<EsnNlmsExpert> esn_nlms_;

  std::unique_ptr<fx2::opt::TransformerOpt> transformer6m_;
  std::array<bool, 256> transformer6m_model_vocab_{};
  std::array<int, 256> transformer6m_byte_to_index_{};
  std::vector<unsigned char> transformer6m_vocab_bytes_;
  std::vector<std::uint16_t> transformer6m_half_scratch_;
  std::vector<float> transformer6m_probabilities_;
  std::vector<float> transformer6m_actual_probabilities_;
  std::array<unsigned char, 15> transformer6m_separator_window_{};
  unsigned long long transformer6m_article_tokens_ = 0;
  bool transformer6m_prediction_active_ = false;

  std::vector<bool> vocab_;
  FXCM fxcm_model_;

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
};

#endif
