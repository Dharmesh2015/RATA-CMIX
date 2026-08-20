#include "predictor.h"
#include "fx4_config.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
Predictor::Predictor(const std::vector<bool>& vocab, bool scr2_enabled)
    : manager_(), sigmoid_(100001), vocab_(vocab) {
  (void)scr2_enabled;
  fxcm_neutral_input_ = sigmoid_.Logit(0.5f);
  for (int raw = -2047; raw <= 2047; ++raw) {
    float p = fxcm_model_.RawPredictionProbability(static_cast<short>(raw));
    if (p < 1.0e-4f) p = 1.0e-4f;
    else if (p > 1.0f - 1.0e-4f) p = 1.0f - 1.0e-4f;
    fxcm_stretched_inputs_[raw + 2047] = sigmoid_.Logit(p);
  }
  fxcm_stretched_inputs_[4095] = fxcm_neutral_input_;
#if FX4_SPECIALIST_CORRECTOR
  specialist_error_.fill(0.25f);
#endif
  AddBracket();
  AddPPMD();
  AddWord();
  AddMatch();
  AddDoubleIndirect();
  // Construct the accepted mixer/LSTM before auxiliary models consume rand().
  // This preserves the baseline LSTM initialization exactly.
  AddMixers();
#if FX4_MINI_CMIX
  mini_shared_map_.assign(256u * 100000u, 0);
  AddMiniCmix();
#endif
  auxiliary_size_ = 3;
#if FX4_MINI_CMIX
  mini_cmix_recent_error_.fill(0.25f);
#endif
}

void Predictor::FreeFxcmMemory() {
  fxcm_model_.FreeMemory();
}

void Predictor::EnablePostR1Portfolio(std::uint32_t mask) {
#if FX4_SELECTIVE_POSTR1
  if (!postr1_experts_) {
    postr1_experts_.reset(new PostR1Experts());
    postr1_residual_one_.fill(0.5f);
  }
  postr1_experts_->EnablePortfolio(mask);
  byte_model_->SetOrderBandsNeeded(
      (postr1_experts_->training_mask() & PostR1Experts::kPpmdEscapeOrder) != 0);
#else
  (void)mask;
#endif
}

void Predictor::SetPostR1Span(std::uint64_t logical_offset,
    std::uint32_t mask, std::uint8_t stream_class,
    std::uint8_t profile_id, std::uint16_t mini_model_mask) {
#if FX4_SELECTIVE_POSTR1
  if (mask != 0 && !postr1_experts_) EnablePostR1Portfolio(mask);
  if (!postr1_experts_) return;
  if (stream_class > static_cast<std::uint8_t>(
          PostR1Experts::StreamClass::kMixed)) {
    stream_class = static_cast<std::uint8_t>(
        PostR1Experts::StreamClass::kMixed);
  }
  postr1_residual_gain_ = (profile_id >> 6) & 3u;
  postr1_experts_->SetSpan(logical_offset, mask,
      static_cast<PostR1Experts::StreamClass>(stream_class), profile_id,
      mini_model_mask);
  byte_model_->SetOrderBandsNeeded(
      (postr1_experts_->training_mask() & PostR1Experts::kPpmdEscapeOrder) != 0);
#else
  (void)logical_offset;
  (void)mask;
  (void)stream_class;
  (void)profile_id;
  (void)mini_model_mask;
#endif
}

void Predictor::SetPostR1DonorProfile(
    const std::vector<std::uint8_t>& bytes,
    const std::vector<std::uint32_t>& segment_lengths) {
#if FX4_SELECTIVE_POSTR1
  if (!postr1_experts_) {
    EnablePostR1Portfolio(PostR1Experts::kDonorProfile);
  }
  postr1_experts_->SetDonorProfile(bytes, segment_lengths);
#else
  (void)bytes;
  (void)segment_lengths;
#endif
}

bool Predictor::HasPostR1DonorProfile() const {
#if FX4_SELECTIVE_POSTR1
  return postr1_experts_ && postr1_experts_->HasDonorProfile();
#else
  return false;
#endif
}
#if FX4_DONOR_FORK_DISCOVERY && FX4_SELECTIVE_POSTR1
void Predictor::SetPostR1BranchSignals(PostR1Experts* target) const {
  std::array<float, 11> mini_model_probabilities{};
  mini_model_probabilities.fill(0.5f);
  target->SetModelSignals(donor_branch_ppmd_probability_,
      donor_branch_lstm_probability_, donor_branch_fxcm_probability_, 0.5f,
      mini_model_probabilities, donor_branch_ppmd_order_bands_,
      donor_branch_ppmd_order_, donor_branch_escape_depth_,
      donor_branch_escape_rate_, donor_branch_residual_probability_,
      donor_branch_match_length_);
}
#endif

#if FX4_SELECTIVE_POSTR1
void Predictor::UpdatePostR1ResidualDistribution() {
  const std::valarray<float>& ppmd = byte_model_->BytePredict();
  const std::valarray<float>& lstm = byte_mixer_->ByteProbabilities();
  postr1_mass_.fill(0.0f);
  for (unsigned int symbol = 0; symbol < 256; ++symbol) {
    if (!vocab_[symbol]) continue;
    const float p = std::max(1.0e-30f, ppmd[symbol]);
    const float q = std::max(1.0e-30f, lstm[symbol]);
    const float middle = std::sqrt(p * q);
    float mass = middle;
    if (postr1_residual_gain_ == 0) {
      mass = std::sqrt(p * middle);       // g = 0.25
    } else if (postr1_residual_gain_ == 2) {
      mass = std::sqrt(q * middle);       // g = 0.75
    } else if (postr1_residual_gain_ == 3) {
      mass = q;                           // g = 1.00
    }
    postr1_mass_[256 + symbol] = mass;
  }
  for (int node = 255; node >= 1; --node) {
    postr1_mass_[node] =
        postr1_mass_[node * 2] + postr1_mass_[node * 2 + 1];
    postr1_residual_one_[node] = postr1_mass_[node] > 0.0f
        ? postr1_mass_[node * 2 + 1] / postr1_mass_[node]
        : 0.5f;
  }
}

float Predictor::PostR1ResidualProbability() const {
  const unsigned int node = manager_.bit_context_ & 255u;
  return node == 0 ? 0.5f : postr1_residual_one_[node];
}
#endif

unsigned long long Predictor::GetNumModels() {
  unsigned long long num = 0;

  // models
  num += bracket_model_->NumOutputs(); // bracket
  num += fxcm_model_.NumOutputs();
  num += direct_models_.size();
  num += match_models_.size();
  num += indirect_ns_models_.size();
  num += indirect_r_models_.size();
  num += byte_model_->NumOutputs();
  num += byte_mixer_->NumOutputs();
  return num;
}

void Predictor::AddMixer(int layer, const unsigned long long& context,
    float learning_rate) {
  if (layer == 0) {
    mixer_0_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_0_.size());
  } else {
    mixer_1_.emplace_back(
        layers_[layer].Inputs(), layers_[layer].ExtraInputs(), context,
      learning_rate, mixer_1_.size());
  }
}

void Predictor::AddBracket() {
  bracket_model_.emplace(manager_.bit_context_, 200, 10, 100000, vocab_);
  const Context& context = manager_.AddBracketContext(manager_.bit_context_, 256, 15);
  direct_models_.emplace_back(context.GetContext(), manager_.bit_context_, 30, 0,
      context.Size());
  indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
      manager_.bit_context_, 300, manager_.shared_map_);
}

void Predictor::AddPPMD() {
  byte_model_.emplace(FX4_PPMD_ORDER, FX4_PPMD_MEMORY_MB,
      manager_.bit_context_, vocab_);
  // No post-R1 portfolio is active yet at construction time, so the
  // order-band trees (only consumed by the kPpmdEscapeOrder expert) are
  // unnecessary overhead until/unless EnablePostR1Portfolio or
  // SetPostR1Span actually requests that expert. See
  // PPMD::SetOrderBandsNeeded for why this cannot affect p_base/archive
  // bytes.
  byte_model_->SetOrderBandsNeeded(false);
}

void Predictor::AddWord() {
  float delta = 200;
  std::vector<std::vector<unsigned int>> model_params = {
  {0},
   {0, 1}, 
      {1}, 
      {1, 2},
      {1, 3}, 
       {2, 3},
       {3, 4},
      {1, 2, 4},
      {2, 3, 4},
       {2}
      };
  for (const auto& params : model_params) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    indirect_ns_models_.emplace_back(manager_.nonstationary_, context.GetContext(),
        manager_.bit_context_, delta, manager_.shared_map_);
  }

  std::vector<std::vector<unsigned int>> model_params2 = {
  {0}, 
  {1}, 
      {1, 3},
       {1, 2, 3}, 
       {7, 2}};
  for (const auto& params : model_params2) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, 2000000, &(manager_.longest_match_));
    if (params[0] == 1 && params.size() == 1) {
      indirect_r_models_.emplace_back(manager_.run_map_, context.GetContext(),
          manager_.bit_context_, delta, manager_.shared_map_);
    }
  }
}

void Predictor::AddMatch() {
  float delta = 0.5;
  int limit = 200;
  unsigned long long max_size = 2000000;
  std::vector<std::vector<int>> model_params = {
  {0, 8}, 
  {1, 8}, 
  {7, 4},
      {11, 3}, 
      {13, 2}, 
  };

  for (const auto& params : model_params) {
    const Context& context = manager_.AddContextHashContext(manager_.bit_context_,params[0], params[1]);
    match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, limit, delta, std::min(max_size, context.Size()),
        &(manager_.longest_match_));
  }
}

void Predictor::AddDoubleIndirect() {
  float delta = 400;
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind1,  manager_.bit_context_, delta, manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind2,  manager_.bit_context_, delta, manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind3,  manager_.bit_context_, delta, manager_.shared_map_);
  indirect_ns_models_.emplace_back(manager_.nonstationary_, manager_.ind5,  manager_.bit_context_, delta, manager_.shared_map_);
}

#if FX4_MINI_CMIX
void Predictor::AddMiniCmix() {
  constexpr int direct_limit = 30;
  constexpr float direct_delta = 0.0f;
  for (unsigned int order = 0; order < 3; ++order) {
    const Context& context =
        manager_.AddContextHashContext(manager_.bit_context_, order, 8);
    mini_direct_models_.emplace_back(context.GetContext(),
        manager_.bit_context_, direct_limit, direct_delta, context.Size());
  }
  {
    const Context& context =
        manager_.AddContextHashContext(manager_.bit_context_, 3, 8);
    mini_direct_hash_models_.emplace_back(context.GetContext(),
        manager_.bit_context_, direct_limit, direct_delta, 100000);
  }

  const std::vector<std::vector<unsigned int>> word_indirect = {
      {1, 2, 3}, {1, 2, 3, 4}, {7}, {7, 2}};
  for (const auto& params : word_indirect) {
    const Context& context = manager_.AddSparseContext(manager_.words_, params);
    mini_indirect_models_.emplace_back(manager_.nonstationary_,
        context.GetContext(), manager_.bit_context_, 200,
        mini_shared_map_);
  }

  {
    const Context& context =
        manager_.AddContextHashContext(manager_.bit_context_, 2, 8);
    mini_match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, std::min<unsigned long long>(
            2000000, context.Size()), &mini_longest_match_);
  }
  {
    const Context& context = manager_.AddSparseContext(
        manager_.words_, std::vector<unsigned int>{7});
    mini_match_models_.emplace_back(manager_.history_, context.GetContext(),
        manager_.bit_context_, 200, 0.5, 2000000,
        &mini_longest_match_);
  }
  {
    const Context& context = manager_.AddSparseContext(
        manager_.words_, std::vector<unsigned int>{1});
    // Upstream cmix uses 500,000 slots here. The current DirectHash layout
    // would consume roughly 640 MiB at that size; 100,000 keeps this
    // complementary expert within the Hutter memory budget.
    mini_direct_hash_models_.emplace_back(context.GetContext(),
        manager_.bit_context_, direct_limit, direct_delta, 100000);
  }
}

float Predictor::PredictMiniCmix(std::uint16_t model_mask) {
  model_mask &= 0x07ffu;
  mini_cmix_inputs_.fill(0.0f);
  mini_cmix_model_probabilities_.fill(0.5f);
  mini_cmix_inputs_[0] = 1.0f;
  auto add_prediction = [this](unsigned int model, float probability) {
    const float bounded = std::max(1.0e-4f,
        std::min(1.0f - 1.0e-4f, probability));
    mini_cmix_model_probabilities_[model] = bounded;
    mini_cmix_inputs_[model + 1] = std::max(-4.0f,
        std::min(4.0f, sigmoid_.Logit(bounded)));
  };
  for (unsigned int i = 0; i < mini_direct_models_.size(); ++i)
    add_prediction(i, mini_direct_models_[i].Predict()[0]);
  add_prediction(3, mini_direct_hash_models_[0].Predict()[0]);
  for (unsigned int i = 0; i < mini_indirect_models_.size(); ++i)
    add_prediction(4 + i, mini_indirect_models_[i].Predict()[0]);
  for (unsigned int i = 0; i < mini_match_models_.size(); ++i)
    add_prediction(8 + i, mini_match_models_[i].Predict()[0]);
  add_prediction(10, mini_direct_hash_models_[1].Predict()[0]);

  mini_cmix_context_ = ((manager_.bpos & 7u) << 4) |
      ((manager_.line_class_ & 7u) << 1) |
      static_cast<unsigned int>(manager_.wrt_state_ != 0);
  float logit = 0.0f;
  unsigned int selected = 0;
  for (unsigned int model = 0; model < kMiniCmixModelCount; ++model) {
    if (model_mask & (1u << model)) {
      logit += mini_cmix_inputs_[model + 1];
      ++selected;
    }
  }
  if (selected) logit /= static_cast<float>(selected);
  logit = std::max(-8.0f, std::min(8.0f, logit));
  mini_cmix_probability_ = Sigmoid::Logistic(logit);
  return mini_cmix_probability_;
}

void Predictor::PerceiveMiniCmix(int bit, std::uint16_t model_mask) {
  (void)model_mask;
  for (auto& model : mini_direct_models_) model.Perceive(bit);
  for (auto& model : mini_direct_hash_models_) model.Perceive(bit);
  for (auto& model : mini_indirect_models_) model.Perceive(bit);
  for (auto& model : mini_match_models_) model.Perceive(bit);
}
void Predictor::ByteUpdateMiniCmix(std::uint16_t model_mask) {
  (void)model_mask;
  for (auto& model : mini_direct_models_) model.ByteUpdate();
  for (auto& model : mini_direct_hash_models_) model.ByteUpdate();
  for (auto& model : mini_indirect_models_) model.ByteUpdate();
  for (auto& model : mini_match_models_) model.ByteUpdate();
}
#endif

unsigned int Discretize(float p) {
  return 1 + 4094 * p;
}
void Predictor::AddMixers() {
  unsigned int vocab_size = 0;
  for (unsigned int i = 0; i < vocab_.size(); ++i) {
    if (vocab_[i]) ++vocab_size;
  }
  byte_mixer_.emplace(1, manager_.bit_context_, vocab_,
      vocab_size, new Lstm(vocab_size, vocab_size, FX4_LSTM_CELLS,
          FX4_LSTM_LAYERS, FX4_LSTM_HORIZON, FX4_LSTM_LEARNING_RATE,
          FX4_LSTM_GRADIENT_CLIP));

  for (int i = 0; i < 2; ++i) {
    layers_.emplace_back(sigmoid_,
        1.0e-4);
  }

  unsigned long long input_size = GetNumModels();
#if FX4_STDERR_PROGRESS
  std::cout << "num models " << input_size << "\n";
#endif
  layers_[0].SetNumModels(input_size);

  AddMixer(0, manager_.mx9, 0.005); 
  AddMixer(0, manager_.mx10, 0.0005); 
  AddMixer(0, manager_.mx11, 0.005); 
  AddMixer(0, manager_.mx12, 0.0005); 
  AddMixer(0, manager_.mx13, 0.005); 
  AddMixer(0, manager_.mxx, 0.001);
  AddMixer(0, manager_.recent_bytes_[2], 0.002);
  AddMixer(0, manager_.line_break_, 0.0007);
  AddMixer(0, manager_.longest_match_, 0.0005);
  AddMixer(0, manager_.mx19cxt, 0.002);
  AddMixer(0, manager_.auxiliary_context_, 0.0005);
  AddMixer(0, manager_.mx18, 0.001);
  AddMixer(0,manager_.mx7, 0.001);
  AddMixer(0, manager_.wordscxt, 0.005);
  AddMixer(0, manager_.b2streamcxt, 0.001);
  AddMixer(0, manager_.mx5, 0.001);
  AddMixer(0, manager_.mx6, 0.005);
  AddMixer(0, manager_.b3streamcxt, 0.001);
  AddMixer(0, manager_.mx8, 0.001);
  AddMixer(0, manager_.mx17, 0.005);
  AddMixer(0, manager_.mx16, 0.005);
  AddMixer(0, manager_.mx14, 0.005);
  AddMixer(0, manager_.mx15, 0.005);

  input_size = mixer_0_.size() + auxiliary_size_;
  layers_[1].SetNumModels(input_size);

  AddMixer(1, final_mixer_context_, 0.0003);

  layers_[0].SetExtraInputSize(mixer_0_.size());

}
int lstmpr=0, lstmex=0;
float byte_mixer_output=0.0f;

float Predictor::Predict() {
  unsigned int input_index = 0;
#if FX4_SELECTIVE_POSTR1
  float mini_cmix_probability = 0.5f;
  std::array<float, 11> mini_model_probabilities{};
  mini_model_probabilities.fill(0.5f);
#if FX4_MINI_CMIX
  mini_cmix_tracking_ = postr1_experts_ &&
      postr1_experts_->MiniCmixTrackingNeeded();
  mini_cmix_used_ = postr1_experts_ && postr1_experts_->MiniCmixNeeded();
  mini_cmix_model_mask_ = mini_cmix_used_
      ? postr1_experts_->MiniCmixModelMask() : 0u;
  if (mini_cmix_tracking_) {
    mini_cmix_probability = PredictMiniCmix(mini_cmix_model_mask_);
    mini_model_probabilities = mini_cmix_model_probabilities_;
  }
#endif
#endif
  auto bracket_model_output = bracket_model_->Predict()[0];
  layers_[0].SetInput(input_index++, bracket_model_output);

  const unsigned int fxcm_model_outputs = fxcm_model_.NumOutputs();
  const short* fxcm_raw_outputs = fxcm_model_.RawPredictions();
  unsigned int fxcm_active_outputs = fxcm_model_.ActivePredictions();
  if (fxcm_active_outputs > fxcm_model_outputs) fxcm_active_outputs = fxcm_model_outputs;
  // fxcmv1 emits outputs densely from zero; carrying an active count avoids
  // a hot per-bit mask scan and the old unused float prediction mirror.
  // The lookup table is already clamped to MixerInput's stretched range, so
  // these 560-ish assignments can skip duplicate bounds checks.
  for (unsigned int j = 0; j < fxcm_active_outputs; ++j) {
    layers_[0].SetStretchedInputUnchecked(
        input_index, fxcm_stretched_inputs_[fxcm_raw_outputs[j] + 2047]);
    ++input_index;
  }
  for (unsigned int j = fxcm_active_outputs; j < fxcm_model_outputs; ++j) {
    layers_[0].SetStretchedInputUnchecked(input_index, fxcm_neutral_input_);
    ++input_index;
  }
  const float aggregate_fxcm_probability = fxcm_model_.FinalProbability();
  const float bounded_fxcm_probability = std::max(1.0e-4f,
      std::min(1.0f - 1.0e-4f, aggregate_fxcm_probability));
  const float aggregate_fxcm_logit =
      sigmoid_.Logit(bounded_fxcm_probability);

  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    const std::valarray<float>& outputs = direct_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }


  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    const std::valarray<float>& outputs = match_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
 
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_ns_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
 
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    const std::valarray<float>& outputs = indirect_r_models_[i].Predict();
    for (unsigned int j = 0; j < outputs.size(); ++j) {
      layers_[0].SetInput(input_index, outputs[j]);
      ++input_index;
    }
  }
  const unsigned int ppmd_model_index = input_index;
  layers_[0].SetInput(input_index++, byte_model_->Predict()[0]);

  float byte_mixer_override = -1;

  if (byte_mixer_output == 0 || byte_mixer_output == 1) byte_mixer_override = byte_mixer_output;
  layers_[0].SetInput(input_index++, byte_mixer_output);
  auto byte_mixer_index = input_index - 1;

  bool postr1_training = false;
#if FX4_SELECTIVE_POSTR1
  postr1_training =
      postr1_experts_ && postr1_experts_->training_mask() != 0;
#endif
  const float ppmd_probability =
      Sigmoid::Logistic(layers_[0].Inputs()[ppmd_model_index]);
  const float lstm_probability =
      Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
  const float auxiliary_average =
      (bounded_fxcm_probability + lstm_probability + ppmd_probability) /
      3.0f;
  manager_.auxiliary_context_ = auxiliary_average * 15;

  for (unsigned int i = 0; i < mixer_0_.size(); ++i) {
    float p = mixer_0_[i].Mix();
    layers_[0].SetExtraInput(i, p);
    layers_[1].SetStretchedInput(i, p);
  }
  layers_[1].SetStretchedInput(mixer_0_.size(), aggregate_fxcm_logit);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[byte_mixer_index]);
  layers_[1].SetStretchedInput(
      mixer_0_.size() + 2, layers_[0].Inputs()[ppmd_model_index]);

  final_mixer_context_ = 0;

  float p = Sigmoid::Logistic(mixer_1_[0].Mix());
  p = sse_.Predict(p);
#if FX4_SPECIALIST_CORRECTOR
  p = PredictSpecialist(p, layers_[0].Inputs()[ppmd_model_index],
      layers_[0].Inputs()[byte_mixer_index],
      aggregate_fxcm_logit);
#endif
#if FX4_DONOR_FORK_DISCOVERY && FX4_SELECTIVE_POSTR1
  donor_branch_ppmd_probability_ =
      Sigmoid::Logistic(layers_[0].Inputs()[ppmd_model_index]);
  donor_branch_lstm_probability_ =
      Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
  donor_branch_fxcm_probability_ = aggregate_fxcm_probability;
  donor_branch_ppmd_order_bands_ = byte_model_->PredictOrderBands();
  donor_branch_ppmd_order_ = byte_model_->EffectiveOrder();
  donor_branch_escape_depth_ = byte_model_->LastEscapeDepth();
  donor_branch_escape_rate_ = byte_model_->RecentEscapeRate();
  donor_branch_residual_probability_ = PostR1ResidualProbability();
  donor_branch_match_length_ =
      static_cast<unsigned int>(manager_.longest_match_);
#endif
#if FX4_SELECTIVE_POSTR1
  postr1_prediction_used_ = false;
  if (byte_mixer_override < 0 && postr1_training) {
    postr1_experts_->SetModelSignals(
        Sigmoid::Logistic(layers_[0].Inputs()[ppmd_model_index]),
        Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]),
        aggregate_fxcm_probability, mini_cmix_probability,
        mini_model_probabilities,
        byte_model_->PredictOrderBands(), byte_model_->EffectiveOrder(),
        byte_model_->LastEscapeDepth(), byte_model_->RecentEscapeRate(),
        PostR1ResidualProbability(),
        static_cast<unsigned int>(manager_.longest_match_));
    p = postr1_experts_->Predict(
        p, manager_.bit_context_, manager_.bpos & 7u);
    postr1_prediction_used_ = true;
  }
#endif
  if (byte_mixer_override >= 0) return byte_mixer_override;
  return p;
}

void Predictor::Perceive(int bit) {
#if FX4_SELECTIVE_POSTR1
  if (postr1_experts_ && postr1_prediction_used_)
    postr1_experts_->Perceive(bit);
#endif
#if FX4_SPECIALIST_CORRECTOR
  PerceiveSpecialist(bit);
#endif
  bracket_model_->Perceive(bit);


  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }
#if FX4_MINI_CMIX
  if (mini_cmix_tracking_) PerceiveMiniCmix(bit, 0x07ffu);
#endif
  byte_model_->Perceive(bit);

  byte_mixer_->Perceive(bit);

  for (auto& mixer: mixer_0_) {
    mixer.Perceive(bit);
  }
  for (auto& mixer: mixer_1_) {
    mixer.Perceive(bit);
  }

  sse_.Perceive(bit);

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;

  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }

    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
#if FX4_MINI_CMIX
    if (mini_cmix_tracking_) ByteUpdateMiniCmix(0x07ffu);
#endif
    byte_model_->ByteUpdate();

    const std::valarray<float>& p = byte_model_->BytePredict();
    for (unsigned int j = 0; j < 256; ++j) {
      byte_mixer_->SetInput(j,p[j]);
    }

    byte_mixer_->ByteUpdate();
#if FX4_SELECTIVE_POSTR1
    if (postr1_experts_) {
      if (postr1_experts_->training_mask() & PostR1Experts::kResidualLstm) {
        UpdatePostR1ResidualDistribution();
      }
      postr1_experts_->ByteUpdate(manager_.recent_bytes_[0]);
    }
#endif
  }
  byte_mixer_output = byte_mixer_->Predict()[0];
  lstmpr=Discretize(byte_mixer_output);
  lstmex=byte_mixer_->ex;
  fxcm_model_.Perceive(bit);
  if (byte_update)manager_.bit_context_ = 1;
}

#if FX4_SPECIALIST_CORRECTOR
unsigned int Predictor::SpecialistStreamClass() const {
  const unsigned int c = static_cast<unsigned int>(manager_.recent_bytes_[0]);
  if (manager_.wrt_state_ != 0 || c >= 0x80 ||
      (c >= 'a' && c <= 'z')) {
    return 0;
  }
  if ((c >= '0' && c <= '9') || manager_.line_class_ == 3) return 1;
  if (c == 'L' || c == 'N' || c == '/' || manager_.line_class_ == 2) {
    return 2;
  }
  if (c == 'P' || c == 'Q' || c == 'R' || c == 'M') return 3;
  if (c == '[' || c == ']' || manager_.line_class_ == 5) return 4;
  if (c == '\n' || c == '*' || manager_.line_class_ == 1 ||
      manager_.line_class_ == 6) {
    return 5;
  }
  if (c == ':' || c == '?' || c == 'O') return 6;
  return 7;
}

float Predictor::PredictSpecialist(float base_probability,
    float ppmd_logit, float lstm_logit, float fxcm_logit) {
  constexpr float kMinimumProbability = 1.0e-5f;
  const float bounded_probability = std::max(kMinimumProbability,
      std::min(1.0f - kMinimumProbability, base_probability));
  const float base_logit = sigmoid_.Logit(bounded_probability);
  const float disagreement = std::fabs(ppmd_logit - lstm_logit);
  const unsigned int disagreement_bucket = disagreement >= 1.0f;
  const float confidence = std::fabs(base_logit);
  const unsigned int confidence_bucket =
      (confidence >= 0.5f) + (confidence >= 1.5f) +
      (confidence >= 3.0f);
  const float order_reliability = std::min(1.0f,
      static_cast<float>(byte_model_->EffectiveOrder() + 1u) / 16.0f);
  const float escape_penalty = std::min(1.0f,
      byte_model_->RecentEscapeRate() +
      static_cast<float>(byte_model_->LastEscapeDepth()) / 8.0f);
  const float ppmd_reliability =
      order_reliability * (1.0f - escape_penalty);
  const unsigned int ppmd_reliability_bucket = ppmd_reliability >= 0.35f;
  specialist_coarse_context_ =
      SpecialistStreamClass() * 8u + (manager_.bpos & 7u);
  // Reuse the old permanently-zero donor bit for a decoder-visible PPMd
  // reliability split. Low-reliability contexts retain their old indexes.
  specialist_context_ =
      ((specialist_coarse_context_ * 2u + disagreement_bucket) * 2u +
       ppmd_reliability_bucket) * 4u + confidence_bucket;

  auto bounded_delta = [base_logit](float model_logit) {
    return std::max(-4.0f, std::min(4.0f, model_logit - base_logit));
  };
  specialist_inputs_[0] = 1.0f;
  specialist_inputs_[1] = bounded_delta(ppmd_logit);
  specialist_inputs_[2] = bounded_delta(lstm_logit);
  specialist_inputs_[3] = bounded_delta(fxcm_logit);
  specialist_inputs_[4] =
      std::max(-4.0f, std::min(4.0f, lstm_logit - ppmd_logit));
  const float match_reliability = std::min(1.0f,
      static_cast<float>(manager_.longest_match_) / 64.0f);
  specialist_inputs_[5] = specialist_inputs_[1] * ppmd_reliability;
  specialist_inputs_[6] = specialist_inputs_[2] *
      (1.0f - ppmd_reliability);
  specialist_inputs_[7] = specialist_inputs_[3] * match_reliability;

  const auto& weights = specialist_weights_[specialist_context_];
  const auto& coarse_weights =
      specialist_coarse_weights_[specialist_coarse_context_];
  float correction = 0.0f;
  for (unsigned int i = 0; i < kSpecialistFeatures; ++i) {
    correction +=
        (weights[i] + coarse_weights[i]) * specialist_inputs_[i];
  }
  correction = std::max(-1.5f, std::min(1.5f, correction));
  specialist_probability_ = Sigmoid::Logistic(base_logit + correction);
  return specialist_probability_;
}

void Predictor::PerceiveSpecialist(int bit) {
  const float error = specialist_probability_ - static_cast<float>(bit);
  float& recent_error = specialist_error_[specialist_context_];
  recent_error += 0.00390625f * (std::fabs(error) - recent_error);
  const float learning_rate =
      FX4_SPECIALIST_LEARNING_RATE * (0.5f + recent_error);
  auto& weights = specialist_weights_[specialist_context_];
  auto& coarse_weights =
      specialist_coarse_weights_[specialist_coarse_context_];
  for (unsigned int i = 0; i < kSpecialistFeatures; ++i) {
    const float update = learning_rate * error * specialist_inputs_[i];
    weights[i] -= 0.75f * update;
    coarse_weights[i] -= 0.25f * update;
  }
}
#endif
void Predictor::Pretrain(int bit) {
  bracket_model_->Predict();
  fxcm_model_.Predict();
    
  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Predict();
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Predict();
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Predict();
  }
#if FX4_MINI_CMIX
  PredictMiniCmix(0x07ffu);
#endif

  bracket_model_->Perceive(bit);
  fxcm_model_.Perceive(bit);
    
  for (unsigned int i = 0; i < direct_models_.size(); ++i) {
    direct_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < match_models_.size(); ++i) {
    match_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
    indirect_ns_models_[i].Perceive(bit);
  }
  for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
    indirect_r_models_[i].Perceive(bit);
  }
#if FX4_MINI_CMIX
  PerceiveMiniCmix(bit, 0x07ffu);
#endif

  bool byte_update = false;
  if (manager_.bit_context_ >= 128) byte_update = true;
  manager_.UpdateContexts(bit);
  if (byte_update) {
    bracket_model_->ByteUpdate();

    for (unsigned int i = 0; i < direct_models_.size(); ++i) {
      direct_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < match_models_.size(); ++i) {
      match_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_ns_models_.size(); ++i) {
      indirect_ns_models_[i].ByteUpdate();
    }
    for (unsigned int i = 0; i < indirect_r_models_.size(); ++i) {
      indirect_r_models_[i].ByteUpdate();
    }
#if FX4_MINI_CMIX
    ByteUpdateMiniCmix(0x07ffu);
#endif
    manager_.bit_context_ = 1;
  }
}
