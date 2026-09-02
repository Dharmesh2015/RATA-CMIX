#include "encoder.h"

Encoder::Encoder(std::ofstream* output, Predictor* predictor)
    : output_(output), low_(0), high_(0xffffffff), predictor_(predictor) {
  output_buffer_.reserve(FX4_IO_BUFFER_BYTES);
}

void Encoder::WriteByte(unsigned int byte) {
  output_buffer_.push_back(static_cast<char>(byte));
  if (output_buffer_.size() == FX4_IO_BUFFER_BYTES) FlushBuffer();
}

void Encoder::FlushBuffer() {
  if (output_buffer_.empty()) return;
  output_->write(output_buffer_.data(),
      static_cast<std::streamsize>(output_buffer_.size()));
  flushed_bytes_ += output_buffer_.size();
  output_buffer_.clear();
}

unsigned int Encoder::Discretize(float probability) {
  return 1 + static_cast<unsigned int>(65534 * probability);
}

void Encoder::Encode(int bit) {
  const unsigned int probability = Discretize(predictor_->Predict());
  const unsigned int midpoint =
      low_ + ((high_ - low_) >> 16) * probability +
      (((high_ - low_) & 0xffff) * probability >> 16);
  if (bit) {
    high_ = midpoint;
  } else {
    low_ = midpoint + 1;
  }
  predictor_->Perceive(bit);

  while (((low_ ^ high_) & 0xff000000) == 0) {
    WriteByte(high_ >> 24);
    low_ <<= 8;
    high_ = (high_ << 8) + 255;
  }
}

void Encoder::Flush() {
  while (((low_ ^ high_) & 0xff000000) == 0) {
    WriteByte(high_ >> 24);
    low_ <<= 8;
    high_ = (high_ << 8) + 255;
  }
  WriteByte(high_ >> 24);
  FlushBuffer();
  output_->flush();
}
