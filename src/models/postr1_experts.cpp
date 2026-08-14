#include "postr1_experts.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace {

constexpr float kMinimumProbability = 1.0e-5f;
constexpr std::uint32_t kPhraseBytes = 16u << 20;
constexpr std::uint32_t kPhraseSlots = 1u << 18;
constexpr std::uint64_t kRollingBase = 0x9e3779b185ebca87ULL;

inline std::uint32_t Rotl32(std::uint32_t value, unsigned int shift) {
  return (value << shift) | (value >> (32 - shift));
}

std::uint64_t RollingPower(unsigned int length) {
  std::uint64_t power = 1;
  while (length--) power *= kRollingBase;
  return power;
}

std::uint64_t DonorContextHash(
    const std::uint8_t* bytes, unsigned int length) {
  std::uint64_t hash = 0x6a09e667f3bcc909ULL ^
      (static_cast<std::uint64_t>(length) << 56);
  for (unsigned int i = 0; i < length; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]) +
        0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  }
  return hash;
}

}  // namespace

PostR1Experts::PostR1Experts() {
  expert_probability_.fill(0.5f);
  mixer_error_.fill(8192);
  const char* oracle_path = std::getenv("FX4_POSTR1_ORACLE");
  oracle_summary_enabled_ = oracle_path && *oracle_path;
  const char* oracle_only = std::getenv("FX4_POSTR1_ORACLE_ONLY");
  oracle_only_ = oracle_only && std::strcmp(oracle_only, "0") != 0;
  const char* trace_path = std::getenv("FX4_MINI_SUBSET_TRACE");
  if (trace_path && *trace_path) {
    mini_subset_trace_ = std::fopen(trace_path, "wb");
    if (mini_subset_trace_) {
      const std::uint8_t header[8] = {
          'F', '4', 'M', 'T', 1, 0, 11, 0};
      std::fwrite(header, 1, sizeof(header), mini_subset_trace_);
      std::setvbuf(mini_subset_trace_, nullptr, _IOFBF, 1u << 20);
    }
  }
}

PostR1Experts::~PostR1Experts() {
  const char* path = std::getenv("FX4_POSTR1_ORACLE");
  if (path && *path) WriteOracle(path);
  path = std::getenv("FX4_POSTR1_SPAN_ORACLE");
  if (path && *path) WriteSpanOracle(path);
  if (mini_subset_trace_) std::fclose(mini_subset_trace_);
}

float PostR1Experts::ClampProbability(float probability) {
  return std::max(kMinimumProbability,
      std::min(1.0f - kMinimumProbability, probability));
}

float PostR1Experts::Logit(float probability) {
  probability = ClampProbability(probability);
  return std::log(probability / (1.0f - probability));
}

float PostR1Experts::Logistic(float logit) {
  if (logit >= 12.0f) return 1.0f - kMinimumProbability;
  if (logit <= -12.0f) return kMinimumProbability;
  return 1.0f / (1.0f + std::exp(-logit));
}

std::uint32_t PostR1Experts::MixHash(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<std::uint32_t>(value ^ (value >> 32));
}

void PostR1Experts::EnablePortfolio(std::uint32_t mask) {
  mask &= ~(kTinySsm | kConfidenceBptt | kLegacyDonorReplay);
  if (mask & kEpisodicCache) {
    mask |= kTokenMatch;
    mask &= ~kEpisodicCache;
  }
  observed_mask_ |= mask;
  if (observed_mask_ & kTokenMatch) EnsureEpisodic();
}

void PostR1Experts::SetSpan(std::uint64_t logical_offset,
    std::uint32_t mask, StreamClass stream_class, std::uint8_t profile_id,
    std::uint16_t mini_model_mask) {
  logical_offset_ = logical_offset;
  // TinySSM has no trained update and confidence-gated BPTT would mutate the
  // main LSTM. Keep both archive bits reserved but neutral in this separate
  // post-R1 specialist.
  evaluation_mask_ =
      mask & ~(kTinySsm | kConfidenceBptt | kLegacyDonorReplay);
  if (evaluation_mask_ & kEpisodicCache) {
    evaluation_mask_ |= kTokenMatch;
    evaluation_mask_ &= ~kEpisodicCache;
  }
  observed_mask_ |= evaluation_mask_;
  active_mask_ = evaluation_mask_ & ~kOracle;
  if (oracle_only_ && (evaluation_mask_ & kOracle)) active_mask_ = 0;
  plan_stream_class_ = stream_class;
  stream_class_ = stream_class == StreamClass::kMixed
      ? StreamClass::kMixed : stream_class;
  profile_id_ = profile_id;
  mini_model_mask_ = (evaluation_mask_ & kMiniCmix)
      ? static_cast<std::uint16_t>(mini_model_mask & 0x07ffu) : 0u;
  if (evaluation_mask_ & kTokenMatch) EnsureEpisodic();
  if (evaluation_mask_ & kOracle) {
    SpanOracleStat span;
    span.offset = logical_offset;
    span.mask = evaluation_mask_;
    span.stream_class = static_cast<std::uint8_t>(stream_class);
    span.profile_id = profile_id;
    span_oracle_.push_back(span);
  }
}

void PostR1Experts::SetModelSignals(float ppmd_probability,
    float lstm_probability, float fxcm_probability,
    float mini_cmix_probability,
    const std::array<float, 11>& mini_model_probabilities,
    const std::array<float, 4>& ppmd_order_bands, unsigned int ppmd_order,
    unsigned int escape_depth, float escape_rate,
    float residual_byte_probability, unsigned int match_length) {
  ppmd_probability_ = ClampProbability(ppmd_probability);
  lstm_probability_ = ClampProbability(lstm_probability);
  fxcm_probability_ = ClampProbability(fxcm_probability);
  mini_cmix_probability_ = ClampProbability(mini_cmix_probability);
  mini_model_probabilities_ = mini_model_probabilities;
  ppmd_order_bands_ = ppmd_order_bands;
  ppmd_order_ = ppmd_order;
  escape_depth_ = escape_depth;
  escape_rate_ = std::max(0.0f, std::min(1.0f, escape_rate));
  residual_byte_probability_ = ClampProbability(residual_byte_probability);
  match_length_ = match_length;
}

float PostR1Experts::CountProbability(Counts* table,
    std::uint32_t index) const {
  const Counts& counts = table[index];
  return static_cast<float>(counts.one) /
      static_cast<float>(counts.zero + counts.one);
}

void PostR1Experts::UpdateCount(Counts* table, std::uint32_t index, int bit) {
  Counts& counts = table[index];
  if (bit) {
    if (counts.one != 0xffff) ++counts.one;
  } else if (counts.zero != 0xffff) {
    ++counts.zero;
  }
  if (static_cast<unsigned int>(counts.zero) + counts.one > 49152u) {
    counts.zero = static_cast<std::uint16_t>((counts.zero + 1) >> 1);
    counts.one = static_cast<std::uint16_t>((counts.one + 1) >> 1);
  }
}

float PostR1Experts::ResidualToBit(float residual_probability,
    float baseline) const {
  return baseline >= 0.5f ? 1.0f - residual_probability
                          : residual_probability;
}

float PostR1Experts::StructuralPrediction(unsigned int bit_position) {
  const std::uint8_t previous = recent_bytes_[(recent_pos_ - 1) & 63u];
  const std::uint8_t previous2 = recent_bytes_[(recent_pos_ - 2) & 63u];
  const unsigned int byte_class =
      (previous >= '0' && previous <= '9') ? 1u :
      (previous >= 'A' && previous <= 'Z') ? 2u :
      (previous >= 'a' && previous <= 'z') ? 3u :
      (previous == '<' || previous == '>' || previous == '[' ||
       previous == ']' || previous == '{' || previous == '}') ? 4u :
      (previous == '/' || previous == ':' || previous == '?' ||
       previous == '&' || previous == '=') ? 5u :
      (previous == '\n' || previous == '\r' || previous == ' ') ? 6u : 7u;
  const std::uint32_t context =
      ((static_cast<unsigned int>(stream_class_) * 8u + bit_position) * 8u +
       byte_class) * 4u + ((previous2 >> 5) & 3u);
  return CountProbability(structural_counts_.data(), context & 4095u);
}

float PostR1Experts::SparsePrediction(unsigned int bit_position) {
  const std::uint64_t hashes[3] = {hash16_, hash32_, hash64_};
  float weighted = 0.0f;
  float weight = 0.0f;
  for (unsigned int i = 0; i < 3; ++i) {
    const std::uint32_t context = MixHash(hashes[i] ^
        (static_cast<std::uint64_t>(current_prefix_) << 32) ^ bit_position) &
        (kCountTableSize - 1u);
    const Counts& counts = sparse_counts_[i][context];
    const float evidence = std::min<float>(32.0f,
        counts.zero + counts.one - 2u);
    weighted += CountProbability(sparse_counts_[i].data(), context) *
        (1.0f + evidence);
    weight += 1.0f + evidence;
  }
  return weighted / weight;
}

float PostR1Experts::WordXmlPrediction(unsigned int bit_position) {
  const std::uint32_t context = MixHash(word_hash_ ^
      (static_cast<std::uint64_t>(stream_class_) << 48) ^
      (static_cast<std::uint64_t>(current_prefix_) << 16) ^ bit_position) &
      (kCountTableSize - 1u);
  return CountProbability(word_xml_counts_.data(), context);
}

float PostR1Experts::CtsPrediction(unsigned int bit_position) {
  const unsigned int depths[4] = {6, 8, 10, 12};
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (unsigned int i = 0; i < 4; ++i) {
    const std::uint32_t mask = (1u << depths[i]) - 1u;
    const std::uint32_t same_bit_skip =
        ((residual_history_ >> bit_position) ^
         (residual_history_ >> (bit_position + 8))) & mask;
    const std::uint32_t context = MixHash(
        (residual_history_ & mask) ^ Rotl32(same_bit_skip, 13) ^
        (static_cast<std::uint32_t>(profile_id_) << 24) ^ bit_position) &
        (kCountTableSize - 1u);
    const Counts& counts = cts_counts_[i][context];
    const float evidence = 1.0f + std::min<float>(64.0f,
        counts.zero + counts.one - 2u);
    numerator += CountProbability(cts_counts_[i].data(), context) * evidence;
    denominator += evidence;
  }
  return ResidualToBit(numerator / denominator, baseline_probability_);
}

float PostR1Experts::DmcPrediction() {
  const DmcNode& node = dmc_[dmc_state_ & (kDmcNodes - 1u)];
  const float residual = static_cast<float>(node.count[1]) /
      static_cast<float>(node.count[0] + node.count[1]);
  return ResidualToBit(residual, baseline_probability_);
}

float PostR1Experts::donor_confidence() const {
  if (phrase_agreement_ < 2) return 0.0f;
  const float length = std::min(1.0f, phrase_confidence_ * (1.0f / 64.0f));
  const float agreement = std::min(1.0f, phrase_agreement_ * (1.0f / 3.0f));
  return length * (0.5f + 0.5f * agreement);
}

float PostR1Experts::profile_donor_confidence() const {
  const float evidence = std::min(1.0f,
      static_cast<float>(donor_profile_confidence_) * (1.0f / 12.0f));
  const float agreement = std::min(1.0f,
      static_cast<float>(donor_profile_agreement_) * 0.25f);
  return evidence * (0.5f + 0.5f * agreement);
}

void PostR1Experts::SetDonorProfile(
    const std::vector<std::uint8_t>& bytes,
    const std::vector<std::uint32_t>& segment_lengths) {
  for (auto& table : donor_profile_) {
    table.assign(kDonorProfileSlots, DonorSlot{});
  }
  donor_profile_prediction_ = 0;
  donor_profile_confidence_ = 0;
  donor_profile_agreement_ = 0;
  donor_gate_score_.fill(0);
  donor_gate_hits_.fill(0);
  donor_gate_context_ = 0;
  donor_recent_bytes_.fill(0);
  donor_recent_pos_ = 0;
  donor_bytes_seen_ = 0;
  // Donor winners are measured from an identical warm primary predictor.
  // Reset only the small specialist so independently selected recipients
  // compose without carrying an earlier donor's correction state.
  for (auto& weights : mixer_weight_) weights.fill(0);
  mixer_error_.fill(8192);
  for (auto& gains : expert_gain_total_) gains.fill(0);
  for (auto& squares : expert_gain_square_) squares.fill(0);
  expert_observations_.fill(0);
  expert_enabled_.fill(false);
  expert_probability_.fill(0.5f);
  expert_delta_.fill(0.0f);
  mixer_input_.fill(0.0f);
  mixer_context_ = 0;
  final_probability_ = 0.5f;
  residual_history_ = 0;
  error_run_ = 0;
  recent_error_ = 0.25f;
  last_hard_prediction_ = 0;
  current_prefix_ = 1;
  last_bit_position_ = 0;

  std::vector<std::uint32_t> single_segment;
  const std::vector<std::uint32_t>* segments = &segment_lengths;
  if (segments->empty()) {
    if (bytes.size() > std::numeric_limits<std::uint32_t>::max()) return;
    single_segment.push_back(static_cast<std::uint32_t>(bytes.size()));
    segments = &single_segment;
  }
  std::uint64_t total = 0;
  for (const std::uint32_t length : *segments) total += length;
  if (total != bytes.size()) return;

  static constexpr unsigned int context_lengths[kDonorProfileContexts] = {
      4, 8, 16, 32};
  std::size_t start = 0;
  for (const std::uint32_t length : *segments) {
    const std::size_t end = start + length;
    for (unsigned int context_index = 0;
         context_index < kDonorProfileContexts; ++context_index) {
      const unsigned int context_length = context_lengths[context_index];
      for (std::size_t i = start + context_length; i < end; ++i) {
        const std::uint64_t key = DonorContextHash(
            bytes.data() + i - context_length, context_length);
        DonorSlot& slot = donor_profile_[context_index][
            MixHash(key) & (kDonorProfileSlots - 1u)];
        const std::uint8_t prediction = bytes[i];
        if (slot.confidence == 0) {
          slot.key = key;
          slot.prediction = prediction;
          slot.confidence = 1;
        } else if (slot.key == key && slot.prediction == prediction) {
          if (slot.confidence != 0xffff) ++slot.confidence;
        } else if (slot.key == key) {
          if (slot.confidence > 1) --slot.confidence;
          else {
            slot.prediction = prediction;
            slot.confidence = 1;
          }
        } else if (slot.confidence > 1) {
          --slot.confidence;
        } else {
          slot.key = key;
          slot.prediction = prediction;
          slot.confidence = 1;
        }
      }
    }
    start = end;
  }
  UpdateDonorProfilePrediction();
}

float PostR1Experts::DonorProfilePrediction(
    unsigned int bit_position) const {
  if (donor_profile_confidence_ == 0) return 0.5f;
  const int predicted_bit =
      (donor_profile_prediction_ >> (7 - bit_position)) & 1;
  const float strength = std::min(0.44f,
      0.04f + 0.36f * profile_donor_confidence());
  return predicted_bit ? 0.5f + strength : 0.5f - strength;
}

void PostR1Experts::UpdateDonorProfilePrediction() {
  donor_profile_confidence_ = 0;
  donor_profile_agreement_ = 0;
  if (donor_profile_[0].empty() || donor_bytes_seen_ < 4) return;

  static constexpr unsigned int context_lengths[kDonorProfileContexts] = {
      4, 8, 16, 32};
  std::array<std::uint16_t, 256> vote{};
  std::array<std::uint8_t, kDonorProfileContexts> predictions{};
  std::array<bool, kDonorProfileContexts> matched{};
  for (unsigned int context_index = 0;
       context_index < kDonorProfileContexts; ++context_index) {
    const unsigned int context_length = context_lengths[context_index];
    if (donor_bytes_seen_ < context_length) continue;
    std::array<std::uint8_t, 32> context{};
    for (unsigned int i = 0; i < context_length; ++i) {
      context[i] = donor_recent_bytes_[
          (donor_recent_pos_ - context_length + i) & 31u];
    }
    const std::uint64_t key =
        DonorContextHash(context.data(), context_length);
    const DonorSlot& slot = donor_profile_[context_index][
        MixHash(key) & (kDonorProfileSlots - 1u)];
    if (slot.confidence == 0 || slot.key != key) continue;
    const unsigned int weight =
        (context_index + 1u) * std::min<unsigned int>(slot.confidence, 24u);
    vote[slot.prediction] = static_cast<std::uint16_t>(
        std::min<unsigned int>(0xffffu, vote[slot.prediction] + weight));
    predictions[context_index] = slot.prediction;
    matched[context_index] = true;
  }

  const auto best = std::max_element(vote.begin(), vote.end());
  if (best == vote.end() || *best == 0) return;
  donor_profile_prediction_ =
      static_cast<std::uint8_t>(best - vote.begin());
  donor_profile_confidence_ = *best;
  for (unsigned int i = 0; i < kDonorProfileContexts; ++i) {
    if (matched[i] && predictions[i] == donor_profile_prediction_) {
      ++donor_profile_agreement_;
    }
  }
}

float PostR1Experts::MatchPrediction(unsigned int bit_position) {
  if (phrase_confidence_ == 0 || phrase_agreement_ < 2) return 0.5f;
  const int predicted_bit = (phrase_prediction_ >> (7 - bit_position)) & 1;
  const float strength = std::min(0.48f,
      0.04f + 0.42f * donor_confidence());
  return predicted_bit ? 0.5f + strength : 0.5f - strength;
}

std::uint32_t PostR1Experts::ExpertMask(unsigned int expert) {
  static constexpr std::uint32_t kMasks[kExpertCount] = {
      kStructural, kPpmdEscapeOrder, kSparseVirtualPpm, kWordXmlPpm,
      kResidualLstm, kMicroDiffusion, kRareResidual, kCtsSkipCts,
      kDmc, kTokenMatch, kDonorProfile, kMiniCmix};
  return expert < kExpertCount ? kMasks[expert] : 0u;
}

bool PostR1Experts::ExpertEnabled(unsigned int expert) const {
  return (evaluation_mask_ & ExpertMask(expert)) != 0;
}

bool PostR1Experts::ExpertOutputEnabled(unsigned int expert) const {
  return (active_mask_ & ExpertMask(expert)) != 0;
}

void PostR1Experts::UpdateExpertGains(int bit) {
  const float base = bit ? baseline_probability_ : 1.0f - baseline_probability_;
  const float base_loss = -std::log2(ClampProbability(base));
  auto& totals = expert_gain_total_[mixer_context_];
  auto& squares = expert_gain_square_[mixer_context_];
  for (unsigned int i = 0; i < kExpertCount; ++i) {
    if (!expert_enabled_[i]) continue;
    const float candidate = Logistic(
        Logit(baseline_probability_) + 0.0625f * expert_delta_[i]);
    const float probability = bit ? candidate : 1.0f - candidate;
    const float gain = base_loss + std::log2(ClampProbability(probability));
    const int scaled = std::max(-2048, std::min(2048,
        static_cast<int>(gain * 256.0f)));
    totals[i] += scaled;
    squares[i] += static_cast<std::uint64_t>(
        static_cast<std::int64_t>(scaled) * scaled);
  }
  if (expert_observations_[mixer_context_] != 0xffffffffu) {
    ++expert_observations_[mixer_context_];
  }
}

std::uint32_t PostR1Experts::MixerContext(unsigned int bit_position) const {
  const unsigned int order_band =
      (ppmd_order_ >= 4) + (ppmd_order_ >= 9) + (ppmd_order_ >= 17);
  const unsigned int disagreement =
      (std::fabs(Logit(ppmd_probability_) - Logit(lstm_probability_)) > 1.0f);
  const std::uint32_t profile_family = profile_id_ & 3u;
  const std::uint32_t packed = bit_position |
      (static_cast<std::uint32_t>(stream_class_) << 3) |
      (order_band << 7) | (disagreement << 9) |
      (profile_family << 10);
  return MixHash(packed) & (kMixerContexts - 1u);
}

float PostR1Experts::Predict(float baseline_probability,
    unsigned int bit_context, unsigned int bit_position) {
  current_prefix_ = static_cast<std::uint8_t>(bit_context & 255u);
  last_bit_position_ = static_cast<std::uint8_t>(bit_position & 7u);
  baseline_probability_ = ClampProbability(baseline_probability);
  final_probability_ = baseline_probability_;
  if (evaluation_mask_ == 0) return baseline_probability_;

  expert_probability_.fill(0.5f);
  if (evaluation_mask_ & kStructural) {
    expert_probability_[0] = StructuralPrediction(bit_position);
  }
  if (evaluation_mask_ & kPpmdEscapeOrder) {
    const unsigned int band =
        ppmd_order_ <= 3 ? 0 : ppmd_order_ <= 8 ? 1 :
        ppmd_order_ <= 16 ? 2 : 3;
    const float decay = 1.0f / (1.0f + escape_depth_ + 2.0f * escape_rate_);
    expert_probability_[1] = ClampProbability(
        decay * ppmd_order_bands_[band] + (1.0f - decay) *
        (0.65f * ppmd_probability_ + 0.35f * ppmd_order_bands_[0]));
  }
  if (evaluation_mask_ & kSparseVirtualPpm) {
    expert_probability_[2] = SparsePrediction(bit_position);
  }
  if (evaluation_mask_ & kWordXmlPpm) {
    expert_probability_[3] = WordXmlPrediction(bit_position);
  }
  if (evaluation_mask_ & kResidualLstm) {
    expert_probability_[4] = residual_byte_probability_;
  }
  const unsigned int probability_bucket =
      std::min(31u, static_cast<unsigned int>(baseline_probability_ * 32.0f));
  donor_gate_context_ = static_cast<std::uint8_t>(
      bit_position | ((probability_bucket >> 3) << 3));
  const std::uint32_t micro_context = MixHash(
      probability_bucket | (bit_position << 5) |
      ((residual_history_ & 255u) << 8) |
      (static_cast<unsigned int>(stream_class_) << 16)) &
      (kMicroTableSize - 1u);
  if (evaluation_mask_ & kMicroDiffusion) {
    const float residual_probability =
        CountProbability(micro_counts_.data(), micro_context);
    expert_probability_[5] =
        ResidualToBit(residual_probability, baseline_probability_);
  }
  const std::uint32_t residual_context =
      ((probability_bucket * 8u + bit_position) * 8u +
       std::min(7u, error_run_)) & 4095u;
  if (evaluation_mask_ & kRareResidual) {
    expert_probability_[6] = ResidualToBit(
        CountProbability(residual_counts_.data(), residual_context),
        baseline_probability_);
  }
  if (evaluation_mask_ & kCtsSkipCts) {
    expert_probability_[7] = CtsPrediction(bit_position);
  }
  if (evaluation_mask_ & kDmc) expert_probability_[8] = DmcPrediction();
  if (evaluation_mask_ & kTokenMatch) {
    expert_probability_[9] = MatchPrediction(bit_position);
  }
  if (evaluation_mask_ & kDonorProfile) {
    expert_probability_[10] = DonorProfilePrediction(bit_position);
  }
  if (evaluation_mask_ & kMiniCmix) {
    expert_probability_[11] = mini_cmix_probability_;
  }

  const float base_logit = Logit(baseline_probability_);
  mixer_input_.fill(0.0f);
  expert_delta_.fill(0.0f);
  expert_enabled_.fill(false);
  mixer_context_ = MixerContext(bit_position);
  const std::uint32_t observations = expert_observations_[mixer_context_];
  for (unsigned int i = 0; i < kExpertCount; ++i) {
    if (!ExpertEnabled(i)) continue;
    expert_enabled_[i] = true;
    float delta = std::max(-4.0f, std::min(4.0f,
        Logit(expert_probability_[i]) - base_logit));
    if (i == 9 || i == 10) {
      const float entropy_scale = 4.0f * baseline_probability_ *
          (1.0f - baseline_probability_);
      const float confidence = i == 9
          ? donor_confidence() : profile_donor_confidence();
      delta *= entropy_scale * confidence;
    }
    expert_delta_[i] = delta;

    float gate = 0.0f;
    if (i == 10) {
      // Learn from earlier exact hits in this recipient before trusting the
      // profile. The 32 direct regimes keep this causal and cheap while
      // avoiding false-positive donor matches in otherwise easy contexts.
      const unsigned int context = donor_gate_context_;
      if (donor_profile_agreement_ >= 2 &&
          profile_donor_confidence() > 0.0f) {
        gate = 1.0f;
      } else if (profile_donor_confidence() > 0.0f &&
          donor_gate_hits_[context] >= 8 &&
          donor_gate_score_[context] > 16) {
        gate = std::min(1.0f,
            static_cast<float>(donor_gate_score_[context]) / 128.0f);
      }
    } else if (i == 11) {
      // The subset and correction strength are explicitly selected and paid
      // for in F4CP v6, so do not add a second hidden online gate.
      gate = 1.0f;
    } else if (observations >= 64) {
      const double total =
          static_cast<double>(expert_gain_total_[mixer_context_][i]);
      const double square =
          static_cast<double>(expert_gain_square_[mixer_context_][i]);
      const double variance_sum = std::max(0.0,
          square - total * total / static_cast<double>(observations));
      const double safe_gain = total - 2.0 * std::sqrt(variance_sum + 1.0);
      gate = static_cast<float>(std::max(0.0,
          std::min(1.0, safe_gain / (256.0 * 8.0))));
    }
    mixer_input_[i] = delta * gate;
  }

  if (evaluation_mask_ & kContextMixer) {
    mixer_input_[kExpertCount] = 1.0f;
    mixer_input_[kExpertCount + 4] =
        std::min(4.0f, static_cast<float>(match_length_) / 16.0f);
  }

  const float training_correction = CorrectionFor(evaluation_mask_);
  final_probability_ = Logistic(base_logit + training_correction);
  last_hard_prediction_ = baseline_probability_ >= 0.5f;
  if (active_mask_ == 0) return baseline_probability_;
  return Logistic(base_logit + CorrectionFor(active_mask_));
}

float PostR1Experts::CorrectionFor(std::uint32_t mask) const {
  float correction = 0.0f;
  if (mask & kContextMixer) {
    const auto& weights = mixer_weight_[mixer_context_];
    std::int64_t dot = 0;
    for (unsigned int i = 0; i < kMixerFeatures; ++i) {
      if (i < kExpertCount && (mask & ExpertMask(i)) == 0) continue;
      const int input = static_cast<int>(mixer_input_[i] * 2048.0f);
      dot += static_cast<std::int64_t>(weights[i]) * input;
    }
    correction = static_cast<float>(dot) * (1.0f / 4194304.0f);
  } else {
    float generic_sum = 0.0f;
    unsigned int generic_count = 0;
    for (unsigned int i = 0; i < kExpertCount; ++i) {
      if ((mask & ExpertMask(i)) == 0 ||
          std::fabs(mixer_input_[i]) <= 1.0e-6f) {
        continue;
      }
      if (i == 10) {
        const float donor_gain =
            0.25f * (1.0f + static_cast<float>((profile_id_ >> 6) & 3u));
        correction += donor_gain * mixer_input_[i];
      } else if (i == 11) {
        const float mini_gain =
            0.0625f * (1.0f + static_cast<float>((profile_id_ >> 6) & 3u));
        correction += mini_gain * mixer_input_[i];
      } else {
        generic_sum += mixer_input_[i];
        ++generic_count;
      }
    }
    if (generic_count) {
      correction += 0.0625f * generic_sum / generic_count;
    }
  }
  return std::max(-1.5f, std::min(1.5f, correction));
}

void PostR1Experts::Perceive(int bit) {
  if (evaluation_mask_ == 0) return;
  if (mini_subset_trace_ && (evaluation_mask_ & kOracle) &&
      (evaluation_mask_ & kMiniCmix)) {
    std::array<std::int16_t, 13> record{};
    auto quantize = [](float value) {
      const int scaled = static_cast<int>(std::lrint(
          std::max(-15.999f, std::min(15.999f, value)) * 2048.0f));
      return static_cast<std::int16_t>(scaled);
    };
    const float base_logit = Logit(baseline_probability_);
    record[0] = quantize(base_logit);
    for (unsigned int model = 0; model < 11; ++model) {
      record[model + 1] = quantize(
          Logit(mini_model_probabilities_[model]) - base_logit);
    }
    record[12] = quantize(mixer_input_[10]);
    std::fputc(bit ? 1 : 0, mini_subset_trace_);
    std::fwrite(record.data(), sizeof(record[0]), record.size(),
        mini_subset_trace_);
  }
  const int residual = bit ^ last_hard_prediction_;
  const unsigned int bit_position = last_bit_position_;
  if ((evaluation_mask_ & kDonorProfile) != 0 &&
      std::fabs(expert_delta_[10]) > 1.0e-6f) {
    const float donor_gain =
        0.25f * (1.0f + static_cast<float>((profile_id_ >> 6) & 3u));
    const float candidate = Logistic(
        Logit(baseline_probability_) + donor_gain * expert_delta_[10]);
    const float base_mass = bit ? baseline_probability_
                                : 1.0f - baseline_probability_;
    const float candidate_mass = bit ? candidate : 1.0f - candidate;
    const float gain = std::log2(ClampProbability(candidate_mass) /
        ClampProbability(base_mass));
    const int scaled = std::max(-2048, std::min(2048,
        static_cast<int>(gain * 256.0f)));
    std::int32_t& score = donor_gate_score_[donor_gate_context_];
    score -= score / 128;
    score = std::max(-32768, std::min(32768, score + scaled));
    std::uint16_t& hits = donor_gate_hits_[donor_gate_context_];
    if (hits != 0xffffu) ++hits;
  }
  UpdateExpertGains(bit);
  const std::uint8_t previous = recent_bytes_[(recent_pos_ - 1) & 63u];
  const std::uint8_t previous2 = recent_bytes_[(recent_pos_ - 2) & 63u];
  const unsigned int byte_class =
      (previous >= '0' && previous <= '9') ? 1u :
      (previous >= 'A' && previous <= 'Z') ? 2u :
      (previous >= 'a' && previous <= 'z') ? 3u :
      (previous == '<' || previous == '>' || previous == '[' ||
       previous == ']' || previous == '{' || previous == '}') ? 4u :
      (previous == '/' || previous == ':' || previous == '?' ||
       previous == '&' || previous == '=') ? 5u :
      (previous == '\n' || previous == '\r' || previous == ' ') ? 6u : 7u;
  if (evaluation_mask_ & kStructural) {
    const std::uint32_t context =
        ((static_cast<unsigned int>(stream_class_) * 8u + bit_position) * 8u +
         byte_class) * 4u + ((previous2 >> 5) & 3u);
    UpdateCount(structural_counts_.data(), context & 4095u, bit);
  }
  if (evaluation_mask_ & kSparseVirtualPpm) {
    const std::uint64_t hashes[3] = {hash16_, hash32_, hash64_};
    for (unsigned int i = 0; i < 3; ++i) {
      const std::uint32_t context = MixHash(hashes[i] ^
          (static_cast<std::uint64_t>(current_prefix_) << 32) ^ bit_position) &
          (kCountTableSize - 1u);
      UpdateCount(sparse_counts_[i].data(), context, bit);
    }
  }
  if (evaluation_mask_ & kWordXmlPpm) {
    const std::uint32_t context = MixHash(word_hash_ ^
        (static_cast<std::uint64_t>(stream_class_) << 48) ^
        (static_cast<std::uint64_t>(current_prefix_) << 16) ^ bit_position) &
        (kCountTableSize - 1u);
    UpdateCount(word_xml_counts_.data(), context, bit);
  }
  const unsigned int probability_bucket =
      std::min(31u, static_cast<unsigned int>(baseline_probability_ * 32.0f));
  if (evaluation_mask_ & kMicroDiffusion) {
    const std::uint32_t context = MixHash(
        probability_bucket | (bit_position << 5) |
        ((residual_history_ & 255u) << 8) |
        (static_cast<unsigned int>(stream_class_) << 16)) &
        (kMicroTableSize - 1u);
    UpdateCount(micro_counts_.data(), context, residual);
  }
  if (evaluation_mask_ & kRareResidual) {
    const std::uint32_t context =
        ((probability_bucket * 8u + bit_position) * 8u +
         std::min(7u, error_run_)) & 4095u;
    UpdateCount(residual_counts_.data(), context, residual);
  }
  if (evaluation_mask_ & kCtsSkipCts) {
    const unsigned int depths[4] = {6, 8, 10, 12};
    for (unsigned int i = 0; i < 4; ++i) {
      const std::uint32_t mask = (1u << depths[i]) - 1u;
      const std::uint32_t same_bit_skip =
          ((residual_history_ >> bit_position) ^
           (residual_history_ >> (bit_position + 8))) & mask;
      const std::uint32_t context = MixHash(
          (residual_history_ & mask) ^ Rotl32(same_bit_skip, 13) ^
          (static_cast<std::uint32_t>(profile_id_) << 24) ^ bit_position) &
          (kCountTableSize - 1u);
      UpdateCount(cts_counts_[i].data(), context, residual);
    }
  }
  if (evaluation_mask_ & kDmc) {
    DmcNode& node = dmc_[dmc_state_ & (kDmcNodes - 1u)];
    if (node.count[residual] != 0xffff) ++node.count[residual];
    std::uint16_t next = node.next[residual];
    if (next == 0) {
      if (dmc_next_free_ < kDmcNodes) {
        next = static_cast<std::uint16_t>(dmc_next_free_++);
        node.next[residual] = next;
        dmc_[next] = DmcNode{};
      } else {
        next = dmc_state_;
      }
    } else if (static_cast<unsigned int>(node.count[0]) + node.count[1] > 96u &&
        dmc_next_free_ < kDmcNodes) {
      const std::uint16_t clone =
          static_cast<std::uint16_t>(dmc_next_free_++);
      dmc_[clone] = dmc_[next];
      node.next[residual] = clone;
      next = clone;
    }
    dmc_state_ = next;
  }

  if (evaluation_mask_ & kContextMixer) {
    const float error = final_probability_ - static_cast<float>(bit);
    const float absolute_error = std::fabs(error);
    std::uint16_t& recent = mixer_error_[mixer_context_];
    recent = static_cast<std::uint16_t>(
        (recent * 255u + static_cast<unsigned int>(absolute_error * 65535.0f)) >> 8);
    if (absolute_error > 0.015625f) {
      auto& weights = mixer_weight_[mixer_context_];
      const int rate = 1 + (recent >> 13);
      for (unsigned int i = 0; i < kMixerFeatures; ++i) {
        const int gradient = static_cast<int>(
            error * mixer_input_[i] * static_cast<float>(rate) * 8.0f);
        const int updated = static_cast<int>(weights[i]) - gradient;
        weights[i] = static_cast<std::int16_t>(
            std::max(-32767, std::min(32767, updated)));
      }
    }
  }

  if (evaluation_mask_ & kOracle) {
    auto bit_loss = [bit](float probability) {
      probability = ClampProbability(probability);
      return -std::log2(bit ? probability : 1.0f - probability);
    };
    auto add_loss = [&bit_loss](OracleStat* stat, float probability) {
      stat->loss_bits += bit_loss(probability);
      ++stat->bits;
    };
    if (oracle_summary_enabled_) {
      add_loss(&oracle_[0], baseline_probability_);
      for (unsigned int i = 0; i < kExpertCount; ++i) {
        add_loss(&oracle_[i + 1], expert_probability_[i]);
      }
      add_loss(&oracle_[kExpertCount + 1], final_probability_);
    }

    if (!span_oracle_.empty()) {
      SpanOracleStat& span = span_oracle_.back();
      const float base_logit = Logit(baseline_probability_);
      const std::uint32_t context = evaluation_mask_ & kContextMixer;
      const std::array<std::uint32_t, 4> alternatives{{
          0u, kMiniCmix | context, kDonorProfile | context,
          kMiniCmix | kDonorProfile | context}};
      std::array<double, 4> losses{};
      losses[0] = bit_loss(baseline_probability_);
      for (unsigned int i = 1; i < alternatives.size(); ++i) {
        const std::uint32_t required = alternatives[i] & ~kContextMixer;
        if ((evaluation_mask_ & required) != required) {
          losses[i] = losses[0];
        } else if (i == 3 && (evaluation_mask_ & kMiniCmix) == 0) {
          losses[i] = losses[2];
        } else {
          losses[i] = bit_loss(Logistic(
              base_logit + CorrectionFor(alternatives[i])));
        }
        span.loss_bits[i] += losses[i];
      }
      span.loss_bits[0] += losses[0];
      unsigned int selected = 0;
      if (evaluation_mask_ & kMiniCmix) selected |= 1u;
      if (evaluation_mask_ & kDonorProfile) selected |= 2u;
      span.loss_bits[4] += losses[selected];
      ++span.bits;
    }
  }

  recent_error_ += 0.00390625f *
      (std::fabs(final_probability_ - bit) - recent_error_);
  if (residual) ++error_run_;
  else error_run_ = 0;
  residual_history_ = (residual_history_ << 1) | residual;
}

void PostR1Experts::EnsureEpisodic() {
  if (!phrase_ring_.empty()) return;
  phrase_ring_.assign(kPhraseBytes, 0);
  phrase_position_.resize(kPhraseSlots);
  for (auto& positions : phrase_position_) {
    positions.fill(0xffffffffu);
  }
  phrase_mask_ = kPhraseBytes - 1u;
}

void PostR1Experts::UpdateHashes(std::uint8_t byte) {
  static const std::uint64_t power16 = RollingPower(16);
  static const std::uint64_t power32 = RollingPower(32);
  static const std::uint64_t power64 = RollingPower(64);
  auto rolling = [this, byte](std::uint64_t hash, unsigned int window,
                     std::uint64_t power) {
    const std::uint64_t incoming = static_cast<std::uint64_t>(byte) + 1u;
    std::uint64_t updated = hash * kRollingBase + incoming;
    if (bytes_seen_ >= window) {
      const std::uint64_t outgoing = static_cast<std::uint64_t>(
          recent_bytes_[(recent_pos_ - window) & 63u]) + 1u;
      updated -= outgoing * power;
    }
    return updated;
  };
  hash16_ = rolling(hash16_, 16, power16);
  hash32_ = rolling(hash32_, 32, power32);
  hash64_ = rolling(hash64_, 64, power64);
  const bool word = (byte >= 'A' && byte <= 'Z') ||
      (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
      byte == '_';
  if (word) word_hash_ = word_hash_ * 257u + byte;
  else word_hash_ = MixHash(word_hash_ ^ (static_cast<std::uint64_t>(byte) << 32));
}

void PostR1Experts::UpdateStreamClass(std::uint8_t byte) {
  for (std::uint16_t& score : stream_class_score_) {
    score = static_cast<std::uint16_t>(score - (score >> 5));
  }
  auto reward = [this](StreamClass stream_class, unsigned int amount) {
    std::uint16_t& score = stream_class_score_[static_cast<unsigned int>(stream_class)];
    score = static_cast<std::uint16_t>(std::min(65535u, score + amount));
  };
  const bool digit = byte >= '0' && byte <= '9';
  const bool letter = (byte >= 'A' && byte <= 'Z') ||
      (byte >= 'a' && byte <= 'z');
  if (byte == '<' || byte == '>') reward(StreamClass::kXml, 48);
  else if (byte == '{' || byte == '}') reward(StreamClass::kTemplate, 48);
  else if (byte == '|' || byte == '!') reward(StreamClass::kTable, 36);
  else if (byte == '/' || byte == '?' || byte == '&') reward(StreamClass::kUrl, 28);
  else if (byte == '*' || byte == '#') reward(StreamClass::kList, 28);
  else if (digit) reward(StreamClass::kNumber, 16);
  else if (letter || byte == ' ') reward(StreamClass::kProse, 5);
  if (digit && (recent_bytes_[(recent_pos_ - 1) & 63u] == '-' ||
      recent_bytes_[(recent_pos_ - 1) & 63u] == ':')) {
    reward(StreamClass::kDate, 24);
  }
  if (plan_stream_class_ != StreamClass::kMixed) {
    stream_class_ = plan_stream_class_;
    return;
  }
  unsigned int best = static_cast<unsigned int>(StreamClass::kMixed);
  for (unsigned int i = 0; i < stream_class_score_.size(); ++i) {
    if (stream_class_score_[i] > stream_class_score_[best]) best = i;
  }
  stream_class_ = static_cast<StreamClass>(best);
}

void PostR1Experts::ByteUpdate(std::uint8_t byte) {
  if (evaluation_mask_ == 0) return;
  if ((evaluation_mask_ & kOracle) && !span_oracle_.empty()) {
    ++span_oracle_.back().bytes;
  }
  UpdateStreamClass(byte);
  const std::uint32_t donor_only = kDonorProfile | kContextMixer;
  if ((evaluation_mask_ & ~donor_only) == 0) {
    recent_bytes_[recent_pos_++ & 63u] = byte;
    ++bytes_seen_;
    if (evaluation_mask_ & kDonorProfile) {
      donor_recent_bytes_[donor_recent_pos_++ & 31u] = byte;
      ++donor_bytes_seen_;
      UpdateDonorProfilePrediction();
    }
    return;
  }
  UpdateHashes(byte);
  if (evaluation_mask_ & kTokenMatch) {
    EnsureEpisodic();
    const std::uint32_t current = phrase_write_;
    const bool continued = phrase_distance_ != 0 &&
        phrase_prediction_ == byte && phrase_distance_ < current;
    const std::uint32_t old_distance = phrase_distance_;
    const std::uint16_t old_continuation = continuation_length_;

    phrase_ring_[current & phrase_mask_] = byte;
    phrase_prediction_ = 0;
    phrase_confidence_ = 0;
    phrase_agreement_ = 0;
    phrase_distance_ = 0;
    continuation_length_ = 0;

    std::array<std::uint8_t, kPhraseVotes> predicted{};
    std::array<std::uint16_t, kPhraseVotes> matched{};
    std::array<std::uint32_t, kPhraseVotes> distance{};
    unsigned int valid = 0;

    if (continued) {
      const std::uint32_t source_next = current - old_distance + 1u;
      if (source_next < current) {
        predicted[valid] = phrase_ring_[source_next & phrase_mask_];
        matched[valid] = static_cast<std::uint16_t>(
            std::min(256u, static_cast<unsigned int>(old_continuation) + 1u));
        distance[valid] = old_distance;
        ++valid;
      }
    }

    if (bytes_seen_ + 1u >= 16u) {
      const std::uint64_t hashes[kPhraseHashes] = {hash16_, hash32_, hash64_};
      const unsigned int minimum[kPhraseHashes] = {16u, 32u, 64u};
      const std::uint64_t salts[kPhraseHashes] = {
          0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
          0xa4093822299f31d0ULL};
      for (unsigned int hash_index = 0;
           hash_index < kPhraseHashes; ++hash_index) {
        const std::uint32_t slot =
            MixHash(hashes[hash_index] ^ salts[hash_index]) &
            (kPhraseSlots - 1u);
        auto& positions = phrase_position_[slot];
        for (unsigned int candidate = 0;
             candidate < kPhraseCandidates && valid < kPhraseVotes;
             ++candidate) {
          const std::uint32_t previous = positions[candidate];
          if (previous == 0xffffffffu || previous + 1u >= current ||
              current - previous >= kPhraseBytes - 256u) {
            continue;
          }
          unsigned int match = 0;
          while (match < 256u && match <= previous && match <= current &&
              phrase_ring_[(current - match) & phrase_mask_] ==
              phrase_ring_[(previous - match) & phrase_mask_]) {
            ++match;
          }
          if (match < minimum[hash_index]) continue;
          predicted[valid] = phrase_ring_[(previous + 1u) & phrase_mask_];
          matched[valid] = static_cast<std::uint16_t>(match);
          distance[valid] = current - previous;
          ++valid;
        }
        for (unsigned int i = kPhraseCandidates - 1u; i > 0; --i) {
          positions[i] = positions[i - 1u];
        }
        positions[0] = current;
      }
    }

    unsigned int best_score = 0;
    for (unsigned int i = 0; i < valid; ++i) {
      unsigned int score = 0;
      unsigned int agreement = 0;
      unsigned int longest = 0;
      std::uint32_t best_distance = distance[i];
      for (unsigned int j = 0; j < valid; ++j) {
        if (predicted[j] != predicted[i]) continue;
        score += matched[j];
        if (matched[j] > longest ||
            (matched[j] == longest && distance[j] < best_distance)) {
          longest = matched[j];
          best_distance = distance[j];
        }
        ++agreement;
      }
      if (score > best_score) {
        best_score = score;
        phrase_prediction_ = predicted[i];
        phrase_agreement_ = static_cast<std::uint8_t>(
            std::min(255u, agreement));
        phrase_confidence_ = static_cast<std::uint16_t>(
            std::min(511u, longest + 8u * (agreement - 1u)));
        phrase_distance_ = best_distance;
        continuation_length_ = static_cast<std::uint16_t>(
            std::min(256u, longest));
      }
    }
    ++phrase_write_;
  }
  recent_bytes_[recent_pos_++ & 63u] = byte;
  ++bytes_seen_;
  if (evaluation_mask_ & kDonorProfile) {
    donor_recent_bytes_[donor_recent_pos_++ & 31u] = byte;
    ++donor_bytes_seen_;
    UpdateDonorProfilePrediction();
  }
}

bool PostR1Experts::WriteOracle(const char* path) const {
  if (!path || !*path) return false;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) return false;
  static const char* names[kExpertCount + 2] = {
      "baseline", "structural", "ppmd_escape_order", "sparse_virtual_ppm",
      "word_xml_ppm", "residual_lstm", "micro_diffusion", "rare_residual",
      "cts_skipcts", "dmc", "token_match", "donor_profile", "mini_cmix",
      "selected_mixer"};
  output << "model,bits,loss_bits,equivalent_bytes,bpb\n";
  for (unsigned int i = 0; i < kExpertCount + 2; ++i) {
    const OracleStat& stat = oracle_[i];
    output << names[i] << ',' << stat.bits << ',' << stat.loss_bits << ','
           << stat.loss_bits / 8.0 << ','
           << (stat.bits ? stat.loss_bits / stat.bits : 0.0) << '\n';
  }
  return output.good();
}
bool PostR1Experts::WriteSpanOracle(const char* path) const {
  if (!path || !*path) return false;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) return false;
  output.precision(12);
  output << "offset,length,bits,mask,stream_class,profile_id,"
            "baseline_bytes,mini_bytes,donor_bytes,combined_bytes,"
            "selected_bytes,mini_gain_bytes,donor_gain_bytes,"
            "combined_gain_bytes,best_mode,best_gain_bytes\n";
  static const char* modes[4] = {"baseline", "mini_cmix", "donor_profile",
      "mini_donor"};
  for (const SpanOracleStat& span : span_oracle_) {
    if (span.bits == 0) continue;
    unsigned int best = 0;
    for (unsigned int i = 1; i < 4; ++i) {
      if (span.loss_bits[i] < span.loss_bits[best]) best = i;
    }
    output << span.offset << ',' << span.bytes << ',' << span.bits << ','
           << span.mask << ',' << static_cast<unsigned int>(span.stream_class)
           << ',' << static_cast<unsigned int>(span.profile_id);
    for (double loss : span.loss_bits) output << ',' << loss / 8.0;
    output << ',' << (span.loss_bits[0] - span.loss_bits[1]) / 8.0
           << ',' << (span.loss_bits[0] - span.loss_bits[2]) / 8.0
           << ',' << (span.loss_bits[0] - span.loss_bits[3]) / 8.0
           << ',' << modes[best]
           << ',' << (span.loss_bits[0] - span.loss_bits[best]) / 8.0
           << '\n';
  }
  return output.good();
}

