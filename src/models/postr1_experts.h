#ifndef FX4_POSTR1_EXPERTS_H
#define FX4_POSTR1_EXPERTS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

// Selective predictor portfolio for the post-R1 entropy stream. Every expert
// is baseline anchored: a zero mask returns the accepted predictor probability
// exactly. Masks are supplied by a decoder-visible span plan or by the causal
// prefix router, never by looking at undecoded bytes.
class PostR1Experts {
 public:
  enum Mask : std::uint32_t {
    kStructural       = 1u << 0,
    kPpmdEscapeOrder  = 1u << 1,
    kSparseVirtualPpm = 1u << 2,
    kWordXmlPpm       = 1u << 3,
    kResidualLstm     = 1u << 4,
    kMicroDiffusion   = 1u << 5,
    kRareResidual     = 1u << 6,
    kCtsSkipCts       = 1u << 7,
    kDmc              = 1u << 8,
    kTokenMatch       = 1u << 9,
    kEpisodicCache    = 1u << 10,
    kTinySsm          = 1u << 11,
    kContextMixer     = 1u << 12,
    kConfidenceBptt   = 1u << 13,
    kOracle           = 1u << 14,
    kDonorProfile     = 1u << 15,
    kAllPredictors    = (1u << 16) - 1u,
  };

  enum class StreamClass : std::uint8_t {
    kProse = 0,
    kXml = 1,
    kNumber = 2,
    kDate = 3,
    kTable = 4,
    kReference = 5,
    kUrl = 6,
    kIdentifier = 7,
    kTemplate = 8,
    kList = 9,
    kMixed = 10,
  };

  PostR1Experts();
  ~PostR1Experts();

  void EnablePortfolio(std::uint32_t mask);
  void SetSpan(std::uint64_t logical_offset, std::uint32_t mask,
      StreamClass stream_class, std::uint8_t profile_id);
  void SetModelSignals(float ppmd_probability, float lstm_probability,
      float fxcm_probability, const std::array<float, 4>& ppmd_order_bands,
      unsigned int ppmd_order, unsigned int escape_depth,
      float escape_rate, float residual_byte_probability,
      unsigned int match_length);

  float Predict(float baseline_probability, unsigned int bit_context,
      unsigned int bit_position);
  void Perceive(int bit);
  void ByteUpdate(std::uint8_t byte);
  void SetDonorProfile(const std::vector<std::uint8_t>& bytes,
      const std::vector<std::uint32_t>& segment_lengths);

  std::uint32_t training_mask() const {
    const std::uint32_t other =
        observed_mask_ & ~(kDonorProfile | kContextMixer);
    return other == 0 ? active_mask_ : observed_mask_;
  }

 private:
  float donor_confidence() const;
  float profile_donor_confidence() const;
  bool WriteOracle(const char* path) const;
  static constexpr unsigned int kExpertCount = 12;
  static constexpr unsigned int kMixerFeatures = kExpertCount + 5;
  static constexpr unsigned int kMixerContexts = 2048;
  static constexpr unsigned int kCountTableSize = 1u << 16;
  static constexpr unsigned int kMicroTableSize = 1u << 13;
  static constexpr unsigned int kDmcNodes = 1u << 15;
  static constexpr unsigned int kPhraseCandidates = 8;
  static constexpr unsigned int kPhraseHashes = 3;
  static constexpr unsigned int kPhraseVotes =
      kPhraseCandidates * kPhraseHashes + 1;
  static constexpr unsigned int kDonorProfileContexts = 4;
  static constexpr unsigned int kDonorProfileSlots = 1u << 15;

  struct Counts {
    std::uint16_t zero = 1;
    std::uint16_t one = 1;
  };

  struct DmcNode {
    std::uint16_t next[2] = {0, 0};
    std::uint16_t count[2] = {1, 1};
  };

  struct OracleStat {
    double loss_bits = 0.0;
    std::uint64_t bits = 0;
  };
  struct DonorSlot {
    std::uint64_t key = 0;
    std::uint16_t confidence = 0;
    std::uint8_t prediction = 0;
  };


  float CountProbability(Counts* table, std::uint32_t index) const;
  void UpdateCount(Counts* table, std::uint32_t index, int bit);
  float ResidualToBit(float residual_probability, float baseline) const;
  float StructuralPrediction(unsigned int bit_position);
  float SparsePrediction(unsigned int bit_position);
  float WordXmlPrediction(unsigned int bit_position);
  float CtsPrediction(unsigned int bit_position);
  float DmcPrediction();
  float MatchPrediction(unsigned int bit_position);
  std::uint32_t MixerContext(unsigned int bit_position) const;
  float DonorProfilePrediction(unsigned int bit_position) const;
  void UpdateDonorProfilePrediction();
  static std::uint32_t ExpertMask(unsigned int expert);
  bool ExpertEnabled(unsigned int expert) const;
  bool ExpertOutputEnabled(unsigned int expert) const;
  // Extracted from Predict()'s prior local lambda, unchanged math: combines
  // mixer_input_ deltas for experts whose ExpertMask() bit is set in `mask`.
  float CorrectionFor(std::uint32_t mask) const;
  void UpdateExpertGains(int bit);
  void UpdateHashes(std::uint8_t byte);
  void UpdateStreamClass(std::uint8_t byte);
  void EnsureEpisodic();
  static float ClampProbability(float probability);
  static float Logit(float probability);
  static float Logistic(float logit);
  static std::uint32_t MixHash(std::uint64_t value);

  std::uint32_t active_mask_ = 0;
  std::uint32_t observed_mask_ = 0;
  StreamClass stream_class_ = StreamClass::kMixed;
  StreamClass plan_stream_class_ = StreamClass::kMixed;
  std::array<std::uint16_t, 11> stream_class_score_{};
  std::uint8_t profile_id_ = 0;
  std::uint64_t logical_offset_ = 0;
  std::uint64_t bytes_seen_ = 0;

  float baseline_probability_ = 0.5f;
  float ppmd_probability_ = 0.5f;
  float lstm_probability_ = 0.5f;
  float fxcm_probability_ = 0.5f;
  float residual_byte_probability_ = 0.5f;
  std::array<float, 4> ppmd_order_bands_{{0.5f, 0.5f, 0.5f, 0.5f}};
  unsigned int ppmd_order_ = 0;
  unsigned int escape_depth_ = 0;
  float escape_rate_ = 0.0f;
  unsigned int match_length_ = 0;

  std::array<float, kExpertCount> expert_probability_{};
  std::array<float, kExpertCount> expert_delta_{};
  std::array<float, kMixerFeatures> mixer_input_{};
  std::array<std::array<std::int16_t, kMixerFeatures>,
      kMixerContexts> mixer_weight_{};
  std::array<std::uint16_t, kMixerContexts> mixer_error_{};
  std::array<std::array<std::int64_t, kExpertCount>,
      kMixerContexts> expert_gain_total_{};
  std::array<std::array<std::uint64_t, kExpertCount>,
      kMixerContexts> expert_gain_square_{};
  std::array<std::uint32_t, kMixerContexts> expert_observations_{};
  std::array<bool, kExpertCount> expert_enabled_{};
  std::uint32_t mixer_context_ = 0;
  float final_probability_ = 0.5f;

  std::array<Counts, 4096> structural_counts_{};
  std::array<std::array<Counts, kCountTableSize>, 3> sparse_counts_{};
  std::array<Counts, kCountTableSize> word_xml_counts_{};
  std::array<Counts, kMicroTableSize> micro_counts_{};
  std::array<Counts, 4096> residual_counts_{};
  std::array<std::array<Counts, kCountTableSize>, 4> cts_counts_{};
  std::array<DmcNode, kDmcNodes> dmc_{};
  std::uint16_t dmc_state_ = 1;
  std::uint32_t dmc_next_free_ = 2;

  std::array<std::uint8_t, 64> recent_bytes_{};
  std::uint32_t recent_pos_ = 0;
  std::uint64_t hash16_ = 0;
  std::uint64_t hash32_ = 0;
  std::uint64_t hash64_ = 0;
  std::uint64_t word_hash_ = 0;
  std::uint32_t residual_history_ = 0;
  std::uint32_t error_run_ = 0;
  float recent_error_ = 0.25f;
  std::uint8_t last_hard_prediction_ = 0;
  std::uint8_t current_prefix_ = 1;
  std::uint8_t last_bit_position_ = 0;

  std::vector<std::uint8_t> phrase_ring_;
  std::vector<std::array<std::uint32_t, kPhraseCandidates>> phrase_position_;
  std::uint32_t phrase_mask_ = 0;
  std::uint32_t phrase_write_ = 0;
  std::uint8_t phrase_prediction_ = 0;
  std::uint16_t phrase_confidence_ = 0;
  std::uint8_t phrase_agreement_ = 0;
  std::uint32_t phrase_distance_ = 0;
  std::uint16_t continuation_length_ = 0;

  std::array<OracleStat, kExpertCount + 2> oracle_{};
  std::array<std::vector<DonorSlot>, kDonorProfileContexts> donor_profile_;
  std::uint8_t donor_profile_prediction_ = 0;
  std::uint16_t donor_profile_confidence_ = 0;
  std::uint8_t donor_profile_agreement_ = 0;

};

#endif
