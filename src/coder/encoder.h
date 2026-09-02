#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>
#include <vector>

#include "../predictor.h"

class Encoder {
 public:
  Encoder(std::ofstream* output, Predictor* predictor);
  void Encode(int bit);
  void Flush();
  size_t OutputSize() const {
    return flushed_bytes_ + output_buffer_.size();
  }

 private:
  void WriteByte(unsigned int byte);
  void FlushBuffer();
  unsigned int Discretize(float probability);

  std::vector<char> output_buffer_;
  size_t flushed_bytes_ = 0;
  std::ofstream* output_;
  unsigned int low_;
  unsigned int high_;
  Predictor* predictor_;
};

#endif
