#include "predictor.h"
#include "fx4_config.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
Predictor::Predictor(const std::vector<bool>& vocab) : manager_(),
    sigmoid_(100001), vocab_(vocab) {
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
  AddMixers();
  auxiliary_size_ = 2;
}

void Predictor::FreeFxcmMemory() {
  fxcm_model_.FreeMemory();
}

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

  AddMixer(1,manager_.zero_context_, 0.0003);

  layers_[0].SetExtraInputSize(mixer_0_.size());

}
int lstmpr=0, lstmex=0;
float byte_mixer_output=0.0f;

float Predictor::Predict() {
  unsigned int input_index = 0;
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
  auto fxcm_model_index = input_index - 1;
  

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

  float auxiliary_average = Sigmoid::Logistic(layers_[0].Inputs()[fxcm_model_index]) + Sigmoid::Logistic(layers_[0].Inputs()[byte_mixer_index]);
  auxiliary_average /= auxiliary_size_;
  manager_.auxiliary_context_ =auxiliary_average * 15;

  for (unsigned int i = 0; i < mixer_0_.size(); ++i) {
    float p = mixer_0_[i].Mix();
    layers_[0].SetExtraInput(i, p);
    layers_[1].SetStretchedInput(i, p);
  }
  layers_[1].SetStretchedInput(mixer_0_.size(), layers_[0].Inputs()[fxcm_model_index]);
  layers_[1].SetStretchedInput(mixer_0_.size() + 1, layers_[0].Inputs()[byte_mixer_index]);

  float p = Sigmoid::Logistic(mixer_1_[0].Mix());
  p = sse_.Predict(p);
#if FX4_SPECIALIST_CORRECTOR
  p = PredictSpecialist(p, layers_[0].Inputs()[ppmd_model_index],
      layers_[0].Inputs()[byte_mixer_index],
      layers_[0].Inputs()[fxcm_model_index]);
#endif
  if (byte_mixer_override >= 0) return byte_mixer_override;
  return p;
}

void Predictor::Perceive(int bit) {
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
    byte_model_->ByteUpdate();

    const std::valarray<float>& p = byte_model_->BytePredict();
    for (unsigned int j = 0; j < 256; ++j) {
      byte_mixer_->SetInput(j,p[j]);
    }

    byte_mixer_->ByteUpdate();

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
  const unsigned int donor_bucket = 0;
  specialist_coarse_context_ =
      SpecialistStreamClass() * 8u + (manager_.bpos & 7u);
  specialist_context_ =
      (((specialist_coarse_context_ * 2u + disagreement_bucket) * 2u +
        donor_bucket) * 4u) + confidence_bucket;

  auto bounded_delta = [base_logit](float model_logit) {
    return std::max(-4.0f, std::min(4.0f, model_logit - base_logit));
  };
  specialist_inputs_[0] = 1.0f;
  specialist_inputs_[1] = bounded_delta(ppmd_logit);
  specialist_inputs_[2] = bounded_delta(lstm_logit);
  specialist_inputs_[3] = bounded_delta(fxcm_logit);
  specialist_inputs_[4] =
      std::max(-4.0f, std::min(4.0f, lstm_logit - ppmd_logit));
  specialist_inputs_[5] = 0.0f;

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
    manager_.bit_context_ = 1;
  }
}
