#ifndef FX4_POSTR1_EXPERTS_H
#define FX4_POSTR1_EXPERTS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "../fx4_config.h"

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
    // Reuses the retired TinySSM plan bit, so F4CP mask width does not grow.
    kCausalCnn        = 1u << 11,
    kContextMixer     = 1u << 12,
    kConfidenceBptt   = 1u << 13,
    kOracle           = 1u << 14,
    kDonorProfile     = 1u << 15,
    kMiniCmix         = 1u << 16,
    kLegacyDonorReplay = 1u << 17,
    kUrlStructure     = 1u << 18,
    // Train/observe the listed experts while returning the accepted baseline.
    // This preserves causal state for a later selectively active span.
    kShadowOnly       = 1u << 19,
    kShadowLstm200    = 1u << 20,
    kTopologyRecurrence = 1u << 21,
    kAllPredictors    = ((1u << 22) - 1u) & ~kShadowOnly,
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
      StreamClass stream_class, std::uint8_t profile_id,
      std::uint16_t mini_model_mask);
  void SetModelSignals(float ppmd_probability, float lstm_probability,
      float shadow_lstm200_probability, float fxcm_probability,
      float mini_cmix_probability,
      const std::array<float, 11>& mini_model_probabilities,
      const std::array<float, 4>& ppmd_order_bands, unsigned int ppmd_order,
      unsigned int escape_depth, float escape_rate,
      float residual_byte_probability, unsigned int match_length);

  float Predict(float baseline_probability, unsigned int bit_context,
      unsigned int bit_position);
  void Perceive(int bit);
  void ByteUpdate(std::uint8_t byte);
  void SetDonorProfile(const std::vector<std::uint8_t>& bytes,
      const std::vector<std::uint32_t>& segment_lengths);
  bool HasDonorProfile() const { return !donor_profile_[0].empty(); }
  bool MiniCmixNeeded() const {
    return (evaluation_mask_ & kMiniCmix) != 0 && mini_model_mask_ != 0;
  }
  std::uint16_t MiniCmixModelMask() const { return mini_model_mask_; }
  bool MiniCmixTrackingNeeded() const {
    return (observed_mask_ & kMiniCmix) != 0;
  }

  std::uint32_t training_mask() const { return evaluation_mask_; }

 private:
  float donor_confidence() const;
  float profile_donor_confidence() const;
  bool WriteOracle(const char* path) const;
  bool WriteSpanOracle(const char* path) const;
#if FX4_CAUSAL_CNN
  static constexpr unsigned int kExpertCount = 16;
#else
  static constexpr unsigned int kExpertCount = 15;
#endif
  static constexpr unsigned int kMixerFeatures = kExpertCount + 5;
  static constexpr unsigned int kMixerContexts = 2048;
  static constexpr unsigned int kCountTableSize = 1u << 16;
  static constexpr unsigned int kMicroTableSize = 1u << 13;
  static constexpr unsigned int kPpmdCalibrationSize = 1u << 14;
  static constexpr unsigned int kDmcNodes = 1u << 15;
  static constexpr unsigned int kPhraseCandidates = 8;
  static constexpr unsigned int kPhraseHashes = 3;
  static constexpr unsigned int kPhraseVotes =
      kPhraseCandidates * kPhraseHashes + 1;
  static constexpr unsigned int kDonorProfileContexts = 4;
  static constexpr unsigned int kDonorProfileSlots = 1u << 15;
  static constexpr unsigned int kDonorGateContexts = 32;
  static constexpr unsigned int kUrlPhaseTableSize = 1u << 12;
  static constexpr unsigned int kUrlShapeTableSize = 1u << 13;
  static constexpr unsigned int kUrlContinuationTableSize = 1u << 12;
#if FX4_TOPOLOGY_RECURRENCE
  static constexpr unsigned int kTopologyContexts = 3;
  static constexpr unsigned int kTopologyCountTableSize = 1u << 16;
  static constexpr unsigned int kTopologyTokenSlots = 1u << 15;

  static constexpr unsigned int kTopologyCopySlots = 1u << 15;
#endif
#if FX4_CAUSAL_CNN
  static constexpr unsigned int kCnnChannels = 8;
  static constexpr unsigned int kCnnLayers = 6;
  static constexpr unsigned int kCnnHistory = 128;
  static constexpr unsigned int kCnnFeatures = 12;
  static constexpr unsigned int kCnnUpdateBytes = 64;
#endif
  enum class UrlPhase : std::uint8_t {
    kOutside = 0,
    kHost = 1,
    kPort = 2,
    kPath = 3,
    kQueryKey = 4,
    kQueryValue = 5,
    kFragment = 6,
    kPercentFirst = 7,
    kPercentSecond = 8,
  };

  struct Counts {
    std::uint16_t zero = 1;
    std::uint16_t one = 1;
  };

  struct DmcNode {
    std::uint16_t next[2] = {0, 0};
    std::uint16_t count[2] = {1, 1};
  };

  using FullCountTable = std::array<Counts, kCountTableSize>;

  struct AdaptiveState {
    std::array<std::array<std::int16_t, kMixerFeatures>,
        kMixerContexts> mixer_weight{};
    std::array<std::uint16_t, kMixerContexts> mixer_error{};
    std::array<std::array<std::int64_t, kExpertCount>,
        kMixerContexts> expert_gain_total{};
    std::array<std::array<std::uint64_t, kExpertCount>,
        kMixerContexts> expert_gain_square{};
    std::array<std::uint32_t, kMixerContexts> expert_observations{};

    AdaptiveState() { mixer_error.fill(8192); }
  };

  struct OracleStat {
    double loss_bits = 0.0;
    std::uint64_t bits = 0;
  };
  struct SpanOracleStat {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t bits = 0;
    std::uint32_t mask = 0;
    std::uint8_t stream_class = 0;
    std::uint8_t profile_id = 0;
    std::array<double, 9> loss_bits{};
    std::array<double, kExpertCount> singleton_loss_bits{};
    std::array<std::array<double, 4>, 6> gain_sweep_loss_bits{};
  };
  struct DonorSlot {
    std::uint64_t key = 0;
    std::uint16_t confidence = 0;
    std::uint8_t prediction = 0;
  };

#if FX4_TOPOLOGY_RECURRENCE
  struct TopologyTokenSlot {
    std::uint32_t key = 0;
    std::uint32_t last = UINT32_MAX;
    std::uint32_t previous = UINT32_MAX;
  };

  struct TopologyCopySlot {
    std::uint32_t key = 0;
    std::uint8_t prediction = 0;
    std::uint8_t confidence = 0;
  };

  struct TopologyState {
    std::array<std::array<Counts, kTopologyCountTableSize>,
        kTopologyContexts> counts{};
    std::array<TopologyTokenSlot, kTopologyTokenSlots> tokens{};
    std::array<std::uint32_t, kTopologyContexts> bit_context{};
    std::array<TopologyCopySlot, kTopologyCopySlots> copy{};
    std::array<std::uint32_t, 16> canonical_keys{};
    std::uint64_t token_hash = 0;
    std::uint64_t canonical_hash = 0x6a09e667f3bcc909ULL;
    std::uint32_t distance_history = 0;
    std::uint16_t relation_history = 0;
    std::uint32_t token_ordinal = 0;
    std::uint32_t event_count = 0;
    std::uint32_t repeat_count = 0;
    std::uint32_t last_chord_start = UINT32_MAX;
    std::uint32_t last_chord_end = UINT32_MAX;
    std::uint16_t token_length = 0;
    std::uint8_t canonical_pos = 0;
    std::uint16_t bytes_since_event = 0;
    std::uint32_t copy_context_key = 0;
    std::uint8_t last_token_class = 0;
    std::uint8_t token_class_flags = 0;
    std::uint8_t copy_prediction = 0;
    std::uint8_t copy_confidence = 0;
    bool in_token = false;
  };
#endif
#if FX4_CAUSAL_CNN
  struct CausalCnnState {
    using ChannelHistory = std::array<
        std::array<std::int16_t, kCnnHistory>, kCnnChannels>;
    std::array<ChannelHistory, kCnnLayers + 1> activation{};
    std::array<std::int16_t, kCnnFeatures> features{};
    std::array<std::array<std::int16_t, kCnnFeatures>, 8> head_weight{};
    std::array<std::int16_t, 8> head_bias{};
    std::array<std::array<std::int32_t, kCnnFeatures>, 8> gradient{};
    std::array<std::int32_t, 8> bias_gradient{};
    std::uint32_t write = 0;
    std::uint16_t bytes_until_update = kCnnUpdateBytes;
    std::int32_t error_sum = 0;
    std::uint32_t disagreement_sum = 0;
    std::uint32_t entropy_sum = 0;
    std::uint8_t residual_byte = 0;
    float last_probability = 0.5f;
  };
#endif


  float CountProbability(const Counts* table, std::uint32_t index) const;
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
  float UrlPrediction(unsigned int bit_position) const;
#if FX4_TOPOLOGY_RECURRENCE
  float TopologyPrediction(unsigned int bit_position);
  float TopologyConfidence() const;
#endif
#if FX4_CAUSAL_CNN
  void EnsureCausalCnn();
  float CausalCnnPrediction(unsigned int bit_position);
  void TrainCausalCnn(int bit);
  void UpdateCausalCnnByte(std::uint8_t byte);
#endif
  void UrlContexts(unsigned int bit_position, std::uint32_t* phase_context,
      std::uint32_t* shape_context,
      std::uint32_t* continuation_context) const;
  void UpdateUrlState(std::uint8_t byte);
  static std::uint8_t UrlByteClass(std::uint8_t byte);
  void UpdateDonorProfilePrediction();
#if FX4_TOPOLOGY_RECURRENCE
  void EnsureTopology();
  void UpdateTopologyByte(std::uint8_t byte);
  void FinishTopologyToken(std::uint64_t token_hash,
      std::uint16_t token_length, std::uint8_t token_class);
#endif
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
  void EnsureTables(std::uint32_t mask);
  static float ClampProbability(float probability);
  static float Logit(float probability);
  static float Logistic(float logit);
  static std::uint32_t MixHash(std::uint64_t value);

  std::uint32_t active_mask_ = 0;
  std::uint32_t evaluation_mask_ = 0;
  std::uint32_t observed_mask_ = 0;
  StreamClass stream_class_ = StreamClass::kMixed;
  StreamClass plan_stream_class_ = StreamClass::kMixed;
  std::array<std::uint16_t, 11> stream_class_score_{};
  std::uint8_t profile_id_ = 0;
  std::uint16_t mini_model_mask_ = 0;
  std::uint64_t logical_offset_ = 0;
  std::uint64_t bytes_seen_ = 0;

  float baseline_probability_ = 0.5f;
  float ppmd_probability_ = 0.5f;
  float lstm_probability_ = 0.5f;
  float shadow_lstm200_probability_ = 0.5f;
  float fxcm_probability_ = 0.5f;
  float mini_cmix_probability_ = 0.5f;
  std::array<float, 11> mini_model_probabilities_{};
  float residual_byte_probability_ = 0.5f;
  std::array<float, 4> ppmd_order_bands_{{0.5f, 0.5f, 0.5f, 0.5f}};
  unsigned int ppmd_order_ = 0;
  unsigned int escape_depth_ = 0;
  float escape_rate_ = 0.0f;
  unsigned int match_length_ = 0;

  std::array<float, kExpertCount> expert_probability_{};
  std::array<float, kExpertCount> expert_delta_{};
  std::array<float, kMixerFeatures> mixer_input_{};
  std::unique_ptr<AdaptiveState> adaptive_state_;
  std::array<bool, kExpertCount> expert_enabled_{};
  std::uint32_t mixer_context_ = 0;
  float final_probability_ = 0.5f;

  std::unique_ptr<std::array<Counts, 4096>> structural_counts_;
  std::unique_ptr<std::array<Counts, kPpmdCalibrationSize>>
      ppmd_calibration_counts_;
  std::uint32_t ppmd_calibration_context_ = 0;
  std::unique_ptr<std::array<FullCountTable, 3>> sparse_counts_;
  std::unique_ptr<FullCountTable> word_xml_counts_;
  std::unique_ptr<std::array<Counts, kMicroTableSize>> micro_counts_;
  std::unique_ptr<std::array<Counts, 4096>> residual_counts_;
  std::unique_ptr<std::array<FullCountTable, 4>> cts_counts_;
  std::unique_ptr<std::array<Counts, kUrlPhaseTableSize>>
      url_phase_counts_;
  std::unique_ptr<std::array<Counts, kUrlShapeTableSize>>
      url_shape_counts_;
  std::unique_ptr<std::array<Counts, kUrlContinuationTableSize>>
      url_continuation_counts_;
  std::unique_ptr<std::array<DmcNode, kDmcNodes>> dmc_;
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
  UrlPhase url_phase_ = UrlPhase::kOutside;
  UrlPhase url_percent_return_phase_ = UrlPhase::kOutside;
  std::uint8_t url_marker_state_ = 0;
  std::uint8_t url_previous_class_ = 0;
  std::uint8_t url_alphabet_mask_ = 0;
  std::uint8_t url_segment_length_ = 0;
  std::uint8_t url_label_index_ = 0;
  std::uint8_t url_parameter_index_ = 0;
  std::uint8_t url_previous_separator_ = 0;
  std::uint64_t url_segment_hash_ = 0;

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
  std::vector<SpanOracleStat> span_oracle_;
  bool oracle_only_ = false;
  bool oracle_summary_enabled_ = false;
  FILE* mini_subset_trace_ = nullptr;
  std::array<std::vector<DonorSlot>, kDonorProfileContexts> donor_profile_;
  std::array<std::uint8_t, 32> donor_recent_bytes_{};
  std::uint32_t donor_recent_pos_ = 0;
  std::uint64_t donor_bytes_seen_ = 0;
  std::uint8_t donor_profile_prediction_ = 0;
  std::uint16_t donor_profile_confidence_ = 0;
  std::uint8_t donor_profile_agreement_ = 0;
  std::array<std::int32_t, kDonorGateContexts> donor_gate_score_{};
  std::array<std::uint16_t, kDonorGateContexts> donor_gate_hits_{};
  std::uint8_t donor_gate_context_ = 0;
#if FX4_TOPOLOGY_RECURRENCE
  std::unique_ptr<TopologyState> topology_;
#endif
#if FX4_CAUSAL_CNN
  std::unique_ptr<CausalCnnState> causal_cnn_;
#endif

};

#endif
