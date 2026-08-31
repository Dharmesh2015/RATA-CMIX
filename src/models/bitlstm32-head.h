#ifndef BITLSTM32_HEAD_H
#define BITLSTM32_HEAD_H

// KH_BITLSTM32: per-coded-bit LSTM32 correction head ("shuffle_v2_lstm32_seq64
// _all_no_xml"), applied to the terminal probability AFTER Discretize-equivalent
// quantization and BEFORE arithmetic coding, identically on encode and decode.
//
// The head reproduces, live and causally, the exact 92-feature vector the torch
// trainer built from the res_v3 residual trace (see BITLSTM32_SPEC.md), runs a
// 1-layer LSTM (hid=32, previous-bit input, state reset every 64 coded bits to
// match seq64 TBPTT training), and adds its scalar output to the terminal logit.
//
// Determinism contract: everything here is plain scalar/AVX fp32 + double
// accumulators, compiled in ONE translation unit with value-safe fp
// (-ffp-model=precise; see makefile), entry points noinline. Encode and decode
// therefore execute the identical machine code on identical inputs.
//
// Compile-gated by -DKH_BITLSTM32; runtime-gated by env KH_BITLSTM32=<blob>.
// With the env unset (or the define absent) the coder output is byte-identical
// to the golden-256 build.

#ifdef KH_BITLSTM32

#include <cstddef>
#include <cstdint>

class KhBitLstm32Head {
 public:
  // Loads the weight blob (format: tools/export_bitlstm32_blob.py). On any
  // error prints one warning and stays disabled (ok() == false).
  explicit KhBitLstm32Head(const char* blob_path);
  // Same, from an in-memory blob image (KH_BITLSTM32_EMBED: the fp16 blob
  // linked into the decode-only binary). File and memory sources share the
  // identical parse/widen code, so the weight arrays are byte-identical.
  KhBitLstm32Head(const unsigned char* blob_data, size_t blob_size);
  KhBitLstm32Head(const KhBitLstm32Head& other);
  KhBitLstm32Head& operator=(const KhBitLstm32Head&) = delete;
  ~KhBitLstm32Head();

  bool ok() const { return ok_; }

  // Called once per coded bit, immediately after the coder computed
  // base_p = Discretize(Predict()) (1..65535). stage1 = the 25-float stage-1
  // mixer input row (layers_[1].Inputs(); index 23 = dead fxcm, skipped),
  // m1raw = raw mixer_1_[0].Mix(), override_active = byte-mixer 0/1 override.
  // Returns the corrected discretized probability to code with (== base_p when
  // the override is active: those bits were loss-masked during training).
  // Must be followed by exactly one Observe() with the actual coded bit.
  unsigned int Adjust(unsigned int base_p, const float* stage1, int n_stage1,
                      float m1raw, int override_active);

  // Advances all causal state (windows, shock clock, byte history, previous
  // bit, LSTM stream position) with the actual coded bit.
  void Observe(int bit);

 private:
  bool ok_ = false;
  struct Impl;
  Impl* impl_ = nullptr;
};

#endif  // KH_BITLSTM32
#endif  // BITLSTM32_HEAD_H
