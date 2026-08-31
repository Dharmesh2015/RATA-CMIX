#ifndef BYTE_MIXER_H
#define BYTE_MIXER_H

#include <vector>
#include <memory>

#include "../models/byte-model.h"
#include "lstm.h"

class ByteMixer : public ByteModel {
 public:
  ByteMixer(unsigned int num_models, const unsigned int& bit_context,
      const std::vector<bool>& vocab, unsigned int vocab_size, Lstm* lstm);
  void SetInput(int index, float val);
  // Replace the online LSTM output with one probability per vocabulary byte.
  // Used only by the feature-gated pretrained transformer path.
  void SetProbs(const float* vocab_probs);
#ifdef KH_OBIAS
  // bias256 is indexed by RAW byte value; folded to vocab positions here
  // (identity when the vocab is all 256 bytes, as on transformed enwik9).
  void SetOutputBias(const float* bias256);
#endif
  void ByteUpdate();
  void SetRecurrentTraining(bool enabled) {
    if (lstm_) lstm_->SetRecurrentTraining(enabled);
  }
  const std::valarray<float>& ByteProbabilities() const { return probs_; }

 private:
  std::unique_ptr<Lstm> lstm_;
  const unsigned int& byte_;
  std::valarray<int> byte_map_;
  std::valarray<float> inputs_;
#ifdef KH_OBIAS
  std::valarray<float> folded_bias_;
#endif
  unsigned int num_models_, vocab_size_, offset_;
};

#endif
