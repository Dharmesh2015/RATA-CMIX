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

  // Non-mutating projection of what OutputSize() would read immediately
  // after Flush(): replays Flush()'s renormalization loop (same
  // shift/carry-less logic) on local copies of x1_/x2_ so branch
  // comparisons at a region boundary see the true finalized cost, including
  // the pending arithmetic interval, without disturbing the live encoder or
  // requiring an actual Flush()/resume.
  size_t ProjectedFinalOutputSize() const;

  // Independent branch clone for dual-coder region evaluation: copies the
  // arithmetic-coder interval (x1_/x2_), buffered output, and count-only
  // state, sharing the same output stream pointer and Predictor as the
  // original (the predictor is intentionally NOT cloned -- both branches
  // read the single shared p_base/p_selected values computed externally
  // via EncodeRawBit(), never calling Predict()/Perceive() through the
  // clone itself). Cost-trace state is deliberately not copied (unused by
  // clones). Only one of the resulting branches should ever call Flush()
  // or write to the shared output stream.
  Encoder Clone() const;

  static unsigned int DiscretizeProbability(float p);

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
