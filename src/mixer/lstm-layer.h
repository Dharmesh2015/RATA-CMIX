#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include <valarray>
#include <vector>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "simd-activations.h"
#include "../utils/hugepage.h"

// Flat 64-byte-aligned zero-initialized float buffers replace the previous
// valarray<valarray<float>> storage. This is a layout-only rework: every
// numerical loop keeps the exact per-element evaluation order of the old
// code, so the compressed output stays bit-identical.
struct LstmBufFree { void operator()(void* p) const { free(p); } };
using LstmFloatBuf = std::unique_ptr<float[], LstmBufFree>;
using LstmHalfBuf = std::unique_ptr<uint16_t[], LstmBufFree>;

inline size_t LstmPad16(size_t n) { return (n + 15) & ~size_t(15); }

inline LstmFloatBuf LstmAllocFloats(size_t n) {
  size_t bytes = (n * sizeof(float) + 63) & ~size_t(63);
  if (bytes == 0) bytes = 64;
  float* p = static_cast<float*>(aligned_alloc(64, bytes));
  AdviseHugePages(p, bytes);  // >=4MB only; must precede first touch (memset)
  memset(p, 0, bytes);
  return LstmFloatBuf(p);
}

// fp16 shadow buffers (step_002_003); see simd-activations.h F16 helpers.
inline LstmHalfBuf LstmAllocHalf(size_t n) {
  size_t bytes = (n * sizeof(uint16_t) + 63) & ~size_t(63);
  if (bytes == 0) bytes = 64;
  uint16_t* p = static_cast<uint16_t*>(aligned_alloc(64, bytes));
  AdviseHugePages(p, bytes);
  memset(p, 0, bytes);
  return LstmHalfBuf(p);
}

// step_003_006: the per-epoch histories that ForwardPass writes once per byte
// and BackwardPass re-reads once per burst (gate activations, layer-norm
// rows, cell-state snapshots, tanh(state), input-gate values, layer-input
// rows) are stored fp16-ONLY on F16-capable targets - BackwardPass is their
// sole reader, so there is no fp32 master to keep. The current epoch's
// values, which the forward math itself consumes, live in fp32 scratch rows
// and are quantized into the history as they are finalized. Gradient
// arithmetic, Adam state and weight masters stay fp32.
#if SIMD_ACT_F16
using LstmHistBuf = LstmHalfBuf;
using LstmHistElem = uint16_t;
inline LstmHistBuf LstmAllocHist(size_t n) { return LstmAllocHalf(n); }
#else
using LstmHistBuf = LstmFloatBuf;
using LstmHistElem = float;
inline LstmHistBuf LstmAllocHist(size_t n) { return LstmAllocFloats(n); }
#endif

struct NeuronLayer {
  // input_size: full logical weight-row length. The row splits into a leading
  // one-hot symbol block of sym_width columns and a dense block holding the
  // per-timestep input vector weights (aux inputs + hidden state + bias).
  //
  // Storage layout:
  //  - Symbol-block arrays live TRANSPOSED in one slab, one row group per
  //    symbol: [w | u | m | v], each row num_cells wide (scell_ stride).
  //    The forward matvec and the per-epoch gradient update touch exactly one
  //    symbol column, which is now one contiguous row instead of a
  //    stride-ssym_ walk over num_cells cache lines.
  //  - Dense-block arrays live in one slab, one row group per cell:
  //    [w | u | m | v], each row sdense_ wide. The epoch-0 Adam pass is one
  //    contiguous forward stream, and the forward matvec reads w rows at a
  //    constant stride of 4 row-widths.
  //  - err_ retains each BPTT epoch's fully post-processed error vector so
  //    the dense gradient outer-product can be materialized once per burst
  //    (see LstmLayer::BackwardPass) instead of re-streaming the u slab every
  //    epoch.
  //  - state_/norm_ are per-epoch histories (LstmHistBuf: fp16 on F16
  //    targets, see above); BackwardPass is their only reader.
  NeuronLayer(unsigned int input_size, unsigned int sym_width,
      unsigned int num_cells, int horizon, int offset)
      : ivar_(horizon), gamma_(1.0, num_cells),
      gamma_u_(num_cells), gamma_m_(num_cells), gamma_v_(num_cells),
      beta_(num_cells), beta_u_(num_cells), beta_m_(num_cells),
      beta_v_(num_cells),
      sym_width_(sym_width), dense_width_(input_size - sym_width),
      trows_(input_size - offset),
      sdense_(LstmPad16(dense_width_)), scell_(LstmPad16(num_cells)),
      symt_(LstmAllocFloats((size_t)sym_width_ * 4 * scell_)),
      dense_(LstmAllocFloats((size_t)num_cells * 4 * sdense_)),
      err_(LstmAllocFloats((size_t)horizon * scell_)),
      state_(LstmAllocHist((size_t)horizon * scell_)),
      norm_(LstmAllocHist((size_t)horizon * scell_)),
      transpose_(LstmAllocFloats(trows_ * scell_)) {}

  float* wsymt(size_t s) { return symt_.get() + (s * 4 + 0) * scell_; }
  float* usymt(size_t s) { return symt_.get() + (s * 4 + 1) * scell_; }
  float* msymt(size_t s) { return symt_.get() + (s * 4 + 2) * scell_; }
  float* vsymt(size_t s) { return symt_.get() + (s * 4 + 3) * scell_; }
  // fp16 shadow of this gate's dense w rows. The slab is owned by LstmLayer
  // and fuses the three gates per cell as [wf | wi | wo] (stride
  // whd_stride_ = 3*sdense_), so the fused forward matvec streams one
  // contiguous half-width block per cell. Rows are re-encoded from the fp32
  // master right after Adam updates them (once per horizon) while they are
  // still cache-hot.
  uint16_t* whdense(size_t i) { return whd_ + i * whd_stride_; }
  float* wdense(size_t i) { return dense_.get() + (i * 4 + 0) * sdense_; }
  float* udense(size_t i) { return dense_.get() + (i * 4 + 1) * sdense_; }
  float* mdense(size_t i) { return dense_.get() + (i * 4 + 2) * sdense_; }
  float* vdense(size_t i) { return dense_.get() + (i * 4 + 3) * sdense_; }
  LstmHistElem* state(size_t e) { return state_.get() + e * scell_; }
  LstmHistElem* norm(size_t e) { return norm_.get() + e * scell_; }
  float* err(size_t e) { return err_.get() + e * scell_; }
  float* transpose(size_t r) { return transpose_.get() + r * scell_; }

  std::valarray<float> ivar_, gamma_, gamma_u_, gamma_m_, gamma_v_,
      beta_, beta_u_, beta_m_, beta_v_;
  size_t sym_width_, dense_width_, trows_, sdense_, scell_;
  LstmFloatBuf symt_, dense_, err_;
  LstmHistBuf state_, norm_;
  LstmFloatBuf transpose_;
  uint16_t* whd_ = nullptr;
  size_t whd_stride_ = 0;
};

class LstmLayer {
 public:
  LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
      unsigned int output_size, unsigned int num_cells, int horizon,
      float gradient_clip, float learning_rate);
  void ForwardPass(const float* input, int input_symbol, float* hidden,
      int hidden_start);
  void BackwardPass(const float* input, int epoch, int layer,
      int input_symbol, float* hidden_error);
  static inline float Rand() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

 private:
  // state_ is the running cell state, padded to scell_ (padding lanes stay
  // zero) so its per-epoch fp16 snapshot can use the full-width row encoder.
  LstmFloatBuf state_;
  std::valarray<float> state_error_, stored_error_;
  // Per-epoch input-row pointers recorded while the BPTT burst walks the
  // epochs top-down (non-F16 targets only; F16 targets store an fp16 copy of
  // each epoch's input row in inp_hist_ instead, halving the bytes the
  // deferred gradient outer-product re-streams at epoch 0).
  std::vector<const float*> input_ptrs_;
  float gradient_clip_, learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned int dense_width_;
  size_t scell_, sdense_;
  LstmHistBuf tanh_state_, input_gate_state_, last_state_;
  LstmHistBuf inp_hist_;   // horizon x sdense_ fp16 input-row history (F16)
  LstmFloatBuf scratch_;   // 8 scell_-wide fp32 rows (F16 targets only)
  LstmHalfBuf wdense3h_;  // fused per-cell [wf|wi|wo] fp16 shadow slab
  unsigned long long update_steps_ = 0;
  NeuronLayer forget_gate_, input_node_, output_gate_;

  void ClipGradients(std::valarray<float>* arr);
  void ClipGradients(float* p, size_t n);
  void LayerNorm(NeuronLayer& neurons, float* nrm, float* st);
  void BackwardPass(NeuronLayer& neurons, const float* input, int epoch,
      int layer, int input_symbol, float* hidden_error);
};
#include "lstm-layer.hpp"

#endif
