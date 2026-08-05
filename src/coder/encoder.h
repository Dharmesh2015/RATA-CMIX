#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>
#include <cstdint>
#include <vector>

#include "../predictor.h"

class Encoder {
 public:
  Encoder(std::ofstream* os, Predictor* p);
  void Encode(int bit);
  void EncodeRawBit(int bit, unsigned int p = 32768);
  void ObserveKnownBit(int bit);
  void ObserveKnownByte(unsigned int byte);
  void BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
      unsigned int prev4);
  void EndTraceByte();
  bool StartCostTrace(const char* path, std::uint64_t stream_size);
  void Flush();
  void SetCountOnly(bool enabled) { count_only_ = enabled; }
  size_t OutputSize() const { return out_.size() + count_only_bytes_; }
 private:
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::vector<char> out_;
  size_t count_only_bytes_ = 0;
  bool count_only_ = false;
  std::ofstream* os_;
  std::ofstream cost_trace_;
  double trace_cost_bits_ = 0.0;
  bool trace_byte_active_ = false;
  unsigned int x1_, x2_;
  Predictor* p_;
};
#endif
