#include "decoder.h"

Decoder::Decoder(std::ifstream* input, Predictor* predictor)
    : input_(input), low_(0), high_(0xffffffff), code_(0),
      predictor_(predictor) {
  for (int i = 0; i < 4; ++i) {
    code_ = (code_ << 8) + (ReadByte() & 0xff);
  }
}

int Decoder::ReadByte() {
  const int byte = static_cast<unsigned char>(input_->get());
  return input_->good() ? byte : 0;
}

unsigned int Decoder::Discretize(float probability) {
  return 1 + static_cast<unsigned int>(65534 * probability);
}

int Decoder::Decode() {
  const unsigned int probability = Discretize(predictor_->Predict());
  const unsigned int midpoint =
      low_ + ((high_ - low_) >> 16) * probability +
      (((high_ - low_) & 0xffff) * probability >> 16);
  int bit = 0;
  if (code_ <= midpoint) {
    bit = 1;
    high_ = midpoint;
  } else {
    low_ = midpoint + 1;
  }
  predictor_->Perceive(bit);

  while (((low_ ^ high_) & 0xff000000) == 0) {
    low_ <<= 8;
    high_ = (high_ << 8) + 255;
    code_ = (code_ << 8) + ReadByte();
  }
  return bit;
}
