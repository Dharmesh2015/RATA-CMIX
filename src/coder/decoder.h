#ifndef DECODER_H
#define DECODER_H

#include <fstream>

#include "../predictor.h"

class Decoder {
 public:
  Decoder(std::ifstream* input, Predictor* predictor);
  int Decode();

 private:
  int ReadByte();
  unsigned int Discretize(float probability);

  std::ifstream* input_;
  unsigned int low_;
  unsigned int high_;
  unsigned int code_;
  Predictor* predictor_;
};

#endif
