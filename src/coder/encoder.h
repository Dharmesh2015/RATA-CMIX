#ifndef ENCODER_H
#define ENCODER_H

#include <fstream>
#include <cstdint>
#include <vector>

#include "../predictor.h"

#ifdef KH_TRACE
#include <cstdio>
#include <cstdint>
#endif

#ifdef KH_BITLSTM32
#include <memory>
#include "../models/bitlstm32-head.h"
#endif
#if FX4_RESIDUAL_LSTM96
#ifndef KH_BITLSTM32
#include <memory>
#endif
#include "../models/residual-lstm96-head.h"
#endif

class Encoder {
 public:
  Encoder(std::ofstream* os, Predictor* p);
  void Encode(int bit);
  void EncodeRawBit(int bit, unsigned int p = 32768);
  void ObserveKnownBit(int bit);
  void ObserveKnownByte(unsigned int byte);
  double ObserveKnownBitCost(int bit);
  double ObserveKnownByteCost(unsigned int byte);
  void BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
      unsigned int prev4);
  void EndTraceByte();
  bool StartCostTrace(const char* path, std::uint64_t stream_size);
#if FX4_RESIDUAL_ORACLE_TRACE
  bool StartOracleTrace(std::uint64_t stream_size);
  bool OracleTraceComplete() const {
    return oracle_trace_.is_open() && oracle_record_limit_ != 0 &&
        oracle_record_count_ >= oracle_record_limit_;
  }
#endif
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
  // Discovery branches only need the current interval and emitted-byte count.
  // Avoid copying the complete archive buffer into every count-only trial.
  Encoder CloneCountOnly() const;

  static unsigned int DiscretizeProbability(float p);

 private:
  Encoder(std::ofstream* os, Predictor* p, bool load_terminal_head);
  void WriteByte(unsigned int byte);
  unsigned int Discretize(float p);

  std::vector<char> out_;
  size_t count_only_bytes_ = 0;
  bool count_only_ = false;
  std::ofstream* os_;
  std::ofstream cost_trace_;
  double trace_cost_bits_ = 0.0;
  bool trace_byte_active_ = false;
#if FX4_RESIDUAL_ORACLE_TRACE
  void WriteOracleRecord(int bit, unsigned int base_p, unsigned int final_p);
  std::ofstream oracle_trace_;
  std::uint64_t oracle_record_limit_ = 0;
  std::uint64_t oracle_record_count_ = 0;
#endif
  unsigned int x1_, x2_;
  Predictor* p_;

#ifdef KH_BITLSTM32
  // Per-coded-bit LSTM32 correction head. Constructed only when env
  // KH_BITLSTM32 names a weight blob; with it unset the coder is byte-
  // identical to the golden-256 build. Applied ONLY in Encode() (the coded
  // bits the res_v3 trace recorded), never in EncodeRawBit/ObserveKnownBit.
  std::unique_ptr<KhBitLstm32Head> bitlstm_;
#endif

#if FX4_RESIDUAL_LSTM96
  // Decoder-owned terminal correction. Encoder invokes this exact class and
  // call sequence; there is no separate encoder implementation.
  std::unique_ptr<KhResidualLstm96Head> residual_lstm96_;
#endif

#ifdef KH_TRACE
  // KH_TRACE trace format v3 ("res_v3"). Observation only: the coder state and
  // emitted archive are byte-identical to an untraced run (with KH_TRACE_DIR
  // unset the per-bit overhead is a single bool test). Enabled at runtime when
  // env var KH_TRACE_DIR names a writable directory. Files are keyed by pid AND
  // a process-wide Encoder <seq> counter, so a process that constructs several
  // Encoders never clobbers its own files:
  //   enc.<pid>.<seq>.res   : one fixed 56-byte record per CODED bit, written
  //                           ONLY from Encode() (EncodeRawBit / ObserveKnownBit
  //                           never write). Record k aligns 1:1 with coded bit k.
  //                           { u16 final_p; u8 flags (bit0=coded bit,
  //                             bit1=lstm/byte-mixer override active); u8 rsvd;
  //                             f16[26] } where f16[0..24] = the 25 stage-1
  //                           (layers_[1]) mixer inputs and f16[25] = the raw
  //                           mixer_1_[0].Mix() dot product (pre-Logistic).
  //   enc.<pid>.<seq>.bytes : per input byte { u8 byte; f32 bits_cost }, cost =
  //                           sum of -log2 P(actual bit) from the coder's
  //                           discretized p over the byte's 8 coded bits.
  //   enc.<pid>.<seq>.meta  : text header written at construction; totals and
  //                           truncated flag appended at Flush() (or on failure).
  // See TRACE_FORMAT.md for the exact field/offset/index spec.
  void KhOpen();
  void KhWriteRes(int bit, unsigned int final_p);
  void KhFail(const char* why);
  void KhFinalizeMeta(int truncated);
  FILE* kh_res_f_ = nullptr;
  FILE* kh_bytes_f_ = nullptr;
  bool kh_on_ = false;
  bool kh_opened_ = false;
  bool kh_meta_done_ = false;
  int kh_seq_ = 0;
  unsigned long long kh_bit_index_ = 0;
  unsigned long long kh_byte_index_ = 0;
  double kh_cur_bits_ = 0.0;
  unsigned int kh_cur_byte_ = 0;
#endif
};
#endif
