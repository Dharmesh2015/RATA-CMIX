#include "lstm.h"


#include <numeric>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>

inline Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    learning_rate_(learning_rate), num_cells_(num_cells),
    num_layers_(num_layers), epoch_(0), horizon_(horizon),
    input_size_(input_size), output_size_(output_size) {
  layer_input_len_.resize(num_layers);
  unsigned int max_len = 0;
  for (unsigned int i = 0; i < num_layers; ++i) {
    layer_input_len_[i] = (i == 0) ? (1 + num_cells + input_size)
                                   : (input_size + 1 + num_cells * 2);
    if (layer_input_len_[i] > max_len) max_len = layer_input_len_[i];
  }
  sli_ = LstmPad16(max_len);
  soh_ = LstmPad16(num_cells * num_layers + 1);
  so_ = LstmPad16(output_size);
  she_ = LstmPad16(num_cells);
  hidden_size_ = num_cells * num_layers + 1;
  hidden_ = LstmAllocFloats(soh_);
  hidden_error_ = LstmAllocFloats(she_);
  hidden_.get()[hidden_size_ - 1] = 1;
  layer_input_ = LstmAllocFloats((size_t)horizon * num_layers * sli_);
  output_layer_ = LstmAllocFloats((size_t)output_size * soh_);
  output_ = LstmAllocFloats((size_t)horizon * so_);
  h_hist_ = LstmAllocFloats((size_t)horizon * soh_);
  err0_ = LstmAllocFloats(so_);
  bptt_f_ = LstmAllocFloats((size_t)(horizon + 1) * so_);
  bptt_g_ = LstmAllocFloats((size_t)horizon * horizon);
  bptt_v_ = LstmAllocFloats((size_t)horizon * soh_);
  upd_scratch_ = LstmAllocFloats((size_t)16 * soh_);
#if SIMD_ACT_F16
  output_layer_h_ = LstmAllocHalf((size_t)output_size * soh_);
#endif
  const float init_p = 1.0 / output_size;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    for (unsigned int i = 0; i < num_layers; ++i) {
      LayerInputRow(epoch, i)[layer_input_len_[i] - 1] = 1;
    }
    float* out = OutputRow(epoch);
    for (unsigned int i = 0; i < output_size; ++i) out[i] = init_p;
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.emplace_back(layer_input_len_[i] + output_size, input_size_,
        output_size_, num_cells, horizon, gradient_clip, learning_rate);
  }
}

inline Lstm::~Lstm() {
}

inline void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
        LayerInputRow(epoch_, i));
  }
}

// Rebuild, once per burst, everything the output layer owes the BPTT pass.
// The weight snapshots the old code stored per epoch obey
//   slice[e] = W - lr * sum_{k<=e} err_k (x) h_hist_k
// (W = the live matrix, frozen during the burst; err_k / h_hist_k = the
// rank-1 factors recorded at epoch k). The per-epoch BPTT contribution
//   v_e[j] = sum_i err'_e[i] * slice[e][i][j]
// therefore splits into one dense matmul against W plus a triangular Gram
// correction:
//   v_e = (E * W)[e] - lr * sum_{k<=e} (E[e] . F[k]) * h_hist_k
// where F[k] is update k's error row and E[e] = F[e+1] is epoch e's BPTT
// error row (they are the same vectors: the update recorded at epoch k+1
// uses exactly the BPTT error of slice k). All operands are small and
// cache-resident; the matmul loops are tiled 16 epochs wide so W and
// h_hist_ stream once per tile instead of once per epoch.
inline void Lstm::ComputeOutputBptt() {
  const unsigned int hor = horizon_;
  // F[0] = the boundary update's error row, saved in err0_ when it was
  // recorded (its source output row has since been overwritten); F[k>=1] =
  // OutputRow(k-1) - onehot(input_history_[k-1]), all still live. Padding
  // lanes are zero in every source row, so full padded-width loops are
  // exact over the logical row.
  memcpy(FRow(0), err0_.get(), so_ * sizeof(float));
  for (unsigned int k = 1; k <= hor; ++k) {
    memcpy(FRow(k), OutputRow(k - 1), so_ * sizeof(float));
    FRow(k)[input_history_[k - 1]] -= 1.0f;
  }
  // V_base = E * W.
  for (unsigned int e0 = 0; e0 < hor; e0 += 16) {
    const unsigned int ee = e0 + 16 < hor ? e0 + 16 : hor;
    for (unsigned int e = e0; e < ee; ++e) {
      memset(VRow(e), 0, soh_ * sizeof(float));
    }
    for (unsigned int i = 0; i < output_size_; ++i) {
      const float* __restrict w = OutputLayerRow(i);
      for (unsigned int e = e0; e < ee; ++e) {
        const float f = FRow(e + 1)[i];
        float* __restrict v = VRow(e);
        for (unsigned int j = 0; j < soh_; ++j) v[j] += f * w[j];
      }
    }
  }
  // Triangular Gram: G[e][k] = E[e] . F[k] = F[e+1] . F[k], k <= e.
  for (unsigned int e = 0; e < hor; ++e) {
    const float* __restrict a = FRow(e + 1);
    float* __restrict g = bptt_g_.get() + (size_t)e * hor;
    for (unsigned int k = 0; k <= e; ++k) {
      const float* __restrict b = FRow(k);
      float d = 0;
      for (unsigned int j = 0; j < so_; ++j) d += a[j] * b[j];
      g[k] = d;
    }
  }
  // V -= lr * tri(G) * h_hist_.
  for (unsigned int e0 = 0; e0 < hor; e0 += 16) {
    const unsigned int ee = e0 + 16 < hor ? e0 + 16 : hor;
    for (unsigned int k = 0; k < ee; ++k) {
      const float* __restrict hk = HHistRow(k);
      for (unsigned int e = k > e0 ? k : e0; e < ee; ++e) {
        const float s = learning_rate_ * bptt_g_.get()[(size_t)e * hor + k];
        float* __restrict v = VRow(e);
        for (unsigned int j = 0; j < soh_; ++j) v[j] -= s * hk[j];
      }
    }
  }
}

// Fold the finished burst's pending rank-1 updates into the live matrix in
// one blocked pass (W -= lr * F^T * h_hist_, 16 W rows per tile with the
// delta tile L1-resident) and refresh each row's fp16 shadow while the row
// is hot. Replaces the old per-byte full-matrix copy+update and its
// per-byte fp16 re-encode.
inline void Lstm::ApplyOutputUpdates() {
  const unsigned int hor = horizon_;
  for (unsigned int i0 = 0; i0 < output_size_; i0 += 16) {
    const unsigned int ie = i0 + 16 < output_size_ ? i0 + 16 : output_size_;
    float* __restrict d = upd_scratch_.get();
    memset(d, 0, (size_t)(ie - i0) * soh_ * sizeof(float));
    for (unsigned int k = 0; k < hor; ++k) {
      const float* __restrict hk = HHistRow(k);
      const float* fk = FRow(k);
      for (unsigned int r = 0; r < ie - i0; ++r) {
        const float f = fk[i0 + r];
        float* __restrict dr = d + (size_t)r * soh_;
        for (unsigned int j = 0; j < soh_; ++j) dr[j] += f * hk[j];
      }
    }
    for (unsigned int r = 0; r < ie - i0; ++r) {
      float* __restrict w = OutputLayerRow(i0 + r);
      const float* __restrict dr = d + (size_t)r * soh_;
      for (unsigned int j = 0; j < soh_; ++j) w[j] -= learning_rate_ * dr[j];
#if SIMD_ACT_F16
      simd_act::F16EncodeRow(w, OutputLayerRowH(i0 + r), soh_);
#endif
    }
  }
}

inline const float* Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    ComputeOutputBptt();
    float* he = hidden_error_.get();
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        const float* v = VRow(epoch) + offset;
        for (unsigned int j = 0; j < num_cells_; ++j) he[j] += v[j];
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer].BackwardPass(LayerInputRow(epoch, layer), epoch, layer,
            input_symbol, he);
      }
    }
    // Every snapshot of the finished burst has been consumed; fold its
    // pending updates into W before recording the new burst's first update.
    ApplyOutputUpdates();
  }
  // Record this byte's output-layer SGD step as a pending rank-1 factor
  // instead of materializing a new weight-matrix snapshot: error row =
  // OutputRow(last_epoch) with the just-arrived byte as target (copied into
  // err0_ at the boundary, where OutputRow(horizon-1) is overwritten before
  // its last use), hidden vector = hidden_ as of now (copied before
  // ForwardPass mutates it).
  if (epoch_ == 0) {
    const float* outl = OutputRow(horizon_ - 1);
    float* e0 = err0_.get();
    for (unsigned int i = 0; i < output_size_; ++i) e0[i] = outl[i];
    e0[input] -= 1.0f;
  }
  memcpy(HHistRow(epoch_), hidden_.get(), soh_ * sizeof(float));
  return Predict(input);
}

inline const float* Lstm::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    const float* start = hidden_.get() + i * num_cells_;
    float* row = LayerInputRow(epoch_, i);
    std::copy(start, start + num_cells_, row + input_size_);
    layers_[i].ForwardPass(row, input, hidden_.get(), i * num_cells_);
    if (i < layers_.size() - 1) {
      std::copy(start, start + num_cells_,
          LayerInputRow(epoch_, i + 1) + num_cells_ + input_size_);
    }
  }
  float* out = OutputRow(epoch_);
  const float* h = hidden_.get();
  for (unsigned int i = 0; i < output_size_; ++i) {
#if SIMD_ACT_F16
    // hidden_'s padding lanes and the shadow row's padding lanes are both
    // zero, so the full-width padded dot product is exact over the logical
    // row.
    out[i] = simd_act::F16Dot(h, OutputLayerRowH(i), soh_);
#else
    float sum = 0;
    const float* ol = OutputLayerRow(i);
    for (unsigned int j = 0; j < hidden_size_; ++j) {
      sum += h[j] * ol[j];
    }
    out[i] = sum;
#endif
  }
  // W is frozen during the burst, so add what the pending updates would
  // have folded into it:
  //   out[i] -= lr * err_k[i] * (h_hist_k . h)   for k = 0..epoch_
  // -- exactly the terms the old per-epoch snapshot carried. err_k is err0_
  // for k = 0 and OutputRow(k-1) - onehot(input_history_[k-1]) otherwise;
  // every operand stays cache-resident. out's padding lanes take -s*0 and
  // remain zero.
  for (unsigned int k = 0; k <= epoch_; ++k) {
    const float* __restrict hk = HHistRow(k);
    float dsum = 0;
    for (unsigned int j = 0; j < soh_; ++j) dsum += hk[j] * h[j];
    const float s = learning_rate_ * dsum;
    if (k == 0) {
      const float* __restrict e0 = err0_.get();
      for (unsigned int j = 0; j < so_; ++j) out[j] -= s * e0[j];
    } else {
      const float* __restrict er = OutputRow(k - 1);
      for (unsigned int j = 0; j < so_; ++j) out[j] -= s * er[j];
      out[input_history_[k - 1]] += s;
    }
  }
  simd_act::Exp(out, output_size_);
  float total = 0;
  for (unsigned int i = 0; i < output_size_; ++i) {
    total += out[i];
  }
  for (unsigned int i = 0; i < output_size_; ++i) {
    out[i] /= total;
  }
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  last_input_ = input;
  return OutputRow(epoch);
}
