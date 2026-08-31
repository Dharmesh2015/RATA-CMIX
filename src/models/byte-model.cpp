#include "byte-model.h"

#include <iterator>
#include <numeric>

ByteModel::ByteModel(const std::vector<bool>& vocab) : ex(0),top_(255), mid_(0),
    bot_(0),  vocab_(vocab), probs_(1.0 / 256, 256) {}

 std::valarray<float>& ByteModel::Predict()  {
  auto mid = bot_ + ((top_ - bot_) / 2);
  // Use std::begin(probs_) + offset rather than &probs_[offset] for the
  // half-open range endpoints: valarray::operator[] bounds-checks the index
  // even when only forming a one-past-the-end pointer (top_ can be 255,
  // making top_ + 1 == probs_.size()), which is undefined behavior for
  // valarray unlike the common vector idiom -- it asserts under -O0 and
  // segfaults under -O3. Pointer arithmetic on std::begin() computes the
  // same address without indexing through operator[].
  float num = std::accumulate(std::begin(probs_) + (mid + 1),
                               std::begin(probs_) + (top_ + 1), 0.0f);
  float denom = std::accumulate(std::begin(probs_) + bot_,
                                 std::begin(probs_) + (mid + 1), num);
  ex = bot_;
    float max_prob_val = probs_[bot_];
    for (int i = bot_ + 1; i <= top_; i++) {
      if (probs_[i] > max_prob_val) {
        max_prob_val = probs_[i];
        ex = i;
      }
    }
  if (denom == 0) outputs_[0] = 0.5;
  else outputs_[0] = num / denom;
  return outputs_;
}

const std::valarray<float>& ByteModel::BytePredict() {
  return probs_;
}

void ByteModel::Perceive(int bit) {
  mid_ = bot_ + ((top_ - bot_) / 2);
  if (bit) {
    bot_ = mid_ + 1;
  } else {
    top_ = mid_;
  }
}

void ByteModel::ByteUpdate() {
  top_ = 255;
  bot_ = 0;
  for (int i = 0; i < 256; ++i) {
    if (!vocab_[i]) probs_[i] = 0;
  }
}
