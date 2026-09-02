#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>
#include <vector>
#include <memory>
#include <string>

#include "lstm-layer.h"

#include "../ds/emhash_set.hpp"
#include "../ds/SmallVector.h"

class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  ~Lstm();
  const float* Perceive(unsigned int input);
  const float* Predict(unsigned int input);
  void SetInput(const std::valarray<float>& input);

 private:
  float* LayerInputRow(unsigned int epoch, unsigned int layer) {
    return layer_input_.get() + ((size_t)epoch * num_layers_ + layer) * sli_;
  }
  // The output layer is ONE live weight matrix (output_size_ x soh_) plus a
  // same-geometry fp16 shadow, instead of the previous per-epoch snapshot
  // array (horizon x output_size x soh_). Consecutive snapshots differed by
  // exactly one rank-1 SGD step, so the matrix is kept read-only during a
  // burst, each byte's step is recorded as a rank-1 pending factor, and the
  // burst boundary folds them all in with one batched pass. See Perceive /
  // Predict / ComputeOutputBptt / ApplyOutputUpdates.
  float* OutputLayerRow(unsigned int i) {
    return output_layer_.get() + (size_t)i * soh_;
  }
  uint16_t* OutputLayerRowH(unsigned int i) {
    return output_layer_h_.get() + (size_t)i * soh_;
  }
  float* OutputRow(unsigned int epoch) {
    return output_.get() + (size_t)epoch * so_;
  }
  float* HHistRow(unsigned int k) {
    return h_hist_.get() + (size_t)k * soh_;
  }
  float* FRow(unsigned int k) {
    return bptt_f_.get() + (size_t)k * so_;
  }
  float* VRow(unsigned int e) {
    return bptt_v_.get() + (size_t)e * soh_;
  }
  void ComputeOutputBptt();
  void ApplyOutputUpdates();

  llvm::SmallVector<LstmLayer, 1> layers_;
  std::vector<uint8_t> input_history_; // horizon
  std::vector<unsigned int> layer_input_len_;
  size_t sli_, soh_, so_; // padded row strides of the flat blocks below
  size_t she_; // padded width of hidden_error_ (LstmPad16(num_cells))
  // hidden_ (soh_ wide, logical size hidden_size_ = num_cells*num_layers+1)
  // and hidden_error_ (she_ wide, logical size num_cells) are flat padded
  // 64B-aligned buffers so the fp16 kernels can run full-width with all
  // padding lanes zero.
  LstmFloatBuf layer_input_, output_layer_, output_, hidden_, hidden_error_;
  // Rank-1 pending-update history for the current burst plus burst-boundary
  // scratch: h_hist_ = the hidden vector each epoch's update used (horizon x
  // soh_); err0_ = explicit copy of the boundary update's error row (its
  // source output row is overwritten mid-burst, all other error rows are
  // reconstructed from output_/input_history_ on demand); bptt_f_ = the
  // horizon+1 error rows of a finished burst; bptt_g_ = their triangular
  // Gram matrix; bptt_v_ = per-epoch output-layer BPTT contributions;
  // upd_scratch_ = 16-row delta tile for the batched weight update.
  LstmFloatBuf h_hist_, err0_, bptt_f_, bptt_g_, bptt_v_, upd_scratch_;
  // fp16 shadow of output_layer_; re-encoded row-by-row in the batched
  // boundary update, read by Predict's matvec.
  LstmHalfBuf output_layer_h_;
  float learning_rate_;
  unsigned int num_cells_, num_layers_, epoch_, horizon_, input_size_,
      output_size_, hidden_size_;
  int last_input_ = -1;
};
#include "lstm.hpp"
#endif
