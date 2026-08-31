#ifndef DECODER_H
#define DECODER_H

#include <fstream>

#include "../predictor.h"

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

class Decoder {
 public:
  Decoder(std::ifstream* is, Predictor* p);
  int Decode();
  int DecodeRawBit(unsigned int p = 32768);
  void ObserveKnownBit(int bit);
  void ObserveKnownByte(unsigned int byte);

 private:
  int ReadByte();
  unsigned int Discretize(float p);

  std::ifstream* is_;
  unsigned int x1_, x2_, x_;
  Predictor* p_;

#ifdef KH_BITLSTM32
  // Mirror of Encoder::bitlstm_: identical construction, identical per-coded-
  // bit call sequence (Adjust before the range split, Observe after the bit
  // is known), so encode and decode see identical head state.
  std::unique_ptr<KhBitLstm32Head> bitlstm_;
#endif
#if FX4_RESIDUAL_LSTM96
  std::unique_ptr<KhResidualLstm96Head> residual_lstm96_;
#endif
};

#endif
