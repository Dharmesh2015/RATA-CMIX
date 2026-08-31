#ifndef OBIAS_PRIOR_H
#define OBIAS_PRIOR_H

// KH_OBIAS: output-side residual-on-prior upstream for the byte LSTM.
//
// Replaces the 256-wide PPMd aux INPUT of the byte mixer's LSTM with an
// additive OUTPUT-logit bias computed by a small frozen upstream net
// (trained by tools/train_cmix_ppmd16_stack_torch.py, arm obias_up):
//
//   logits_i += (g0 + w_g . z16) * ln(max(p_i, 1e-6)) + (W_s z16)_i
//
// where p = PPMd order-25 next-byte distribution and z16 =
// tanh(bottleneck(LSTM32(byte lags 1/2/4/8 emb + 9 position features +
// low-rank projection of the clamped centered log dist))). In KH_OBIAS mode
// the byte mixer's SetInput stays all-zero (the aux input-weight block then
// provably never moves from init -- the exact legacy zero-input path), so
// no Lstm rebuild is needed; the bias is added in Lstm::Predict right
// before the exp (see lstm.hpp SetOutputBias).
//
// Determinism contract (house style, as bitlstm32-head): all math in this
// one translation unit, compiled value-safe (-ffp-model=precise), entry
// points noinline; encode and decode execute identical machine code.
//
// Compile-gated by -DKH_OBIAS; runtime-gated by env KH_OBIAS=<blob path>.
// Env unset (or define absent) => output byte-identical to the base build.

#ifdef KH_OBIAS

#include <cstddef>
#include <cstdint>

class KhObiasPrior {
 public:
  // Loads the fp16 weight blob (format: tools/export_obias_blob.py in the
  // pytorch-cmix-lstm worktree). On any error prints one warning and stays
  // disabled (ok() == false).
  //
  // KH_OBIAS_CONST_GATE builds (attempt-4 launch shape): pass nullptr to
  // get the COMPILED constant gate g (no blob, no file, nothing rides
  // archive9): bias_i = g * ln(max(p_i,1e-6)). Executes the exact sched
  // code path with a = KH_OBIAS_CONST_GATE, b = 0, tau = 1 -- bit-identical
  // to the gated A/B arm that used the equivalent 76-byte const blob.
  explicit KhObiasPrior(const char* blob_path);
  ~KhObiasPrior();

  bool ok() const { return ok_; }

  // Called once per completed byte, at the byte boundary, with the byte
  // that just completed and PPMd's 256-way distribution over the NEXT
  // byte (probabilities, sum ~1: exactly byte_model_->BytePredict()).
  // Advances the upstream recurrent state and recomputes the bias row.
  void Advance(unsigned char byte, const float* p256);

  // The current 256-float logit bias row (valid until the next Advance).
  const float* Bias() const;

  // Test hooks (parity harness).
  uint64_t BytesCompleted() const;
  // Current z16 bottleneck activations / gate scalar (valid until the next
  // Advance). Parity-only: compared against the torch golden vectors
  // (obias_handoff/parity_reference: z16_f32, gate_f32, bias_f32).
  const float* Z16() const;
  float Gate() const;

 private:
  bool ok_ = false;
  struct Impl;
  Impl* impl_ = nullptr;
};

#endif  // KH_OBIAS
#endif  // OBIAS_PRIOR_H
