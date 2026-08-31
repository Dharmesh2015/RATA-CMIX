#ifndef RESIDUAL_LSTM96_HEAD_H
#define RESIDUAL_LSTM96_HEAD_H

// Decoder-first CPU residual corrector. The LSTM advances once per completed
// byte and predicts eight corrections for the following byte. Current-bit
// PPMd/LSTM/FXCM signals are handled by a tiny linear/context head, keeping a
// 96-cell recurrent model practical on the full Hutter stream.

#if FX4_RESIDUAL_LSTM96

#include <cstddef>

class KhResidualLstm96Head {
 public:
  explicit KhResidualLstm96Head(const char* blob_path);
  KhResidualLstm96Head(const unsigned char* blob_data, size_t blob_size);
  KhResidualLstm96Head(const KhResidualLstm96Head& other);
  KhResidualLstm96Head& operator=(const KhResidualLstm96Head&) = delete;
  ~KhResidualLstm96Head();

  bool ok() const { return ok_; }

  // current_p is the accepted terminal probability, after BitLSTM32. base_p
  // is the pre-BitLSTM32 probability and is used only as a causal feature.
  // All other inputs are produced by the same Predictor::Predict() call.
  unsigned int Adjust(unsigned int base_p, unsigned int current_p,
      float ppmd_probability, float lstm_probability,
      float fxcm_probability, unsigned int bit_position,
      unsigned int ppmd_order, unsigned int escape_depth,
      unsigned int match_length, unsigned int stream_class,
      int override_active);

  // Called exactly once after Adjust(), with the coded/decoded bit.
  void Observe(int bit);

 private:
  bool ok_ = false;
  struct Impl;
  Impl* impl_ = nullptr;
};

#endif  // FX4_RESIDUAL_LSTM96
#endif  // RESIDUAL_LSTM96_HEAD_H
