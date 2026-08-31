#include "decoder.h"

#if defined(KH_BITLSTM32) || FX4_RESIDUAL_LSTM96
#include <cstdlib>
#endif

#ifdef KH_BITLSTM32_EMBED
extern "C" const unsigned char _binary_kh_blob_embed_bin_start[];
extern "C" const unsigned char _binary_kh_blob_embed_bin_end[];
#endif
#if FX4_RESIDUAL_LSTM96 && defined(KH_RESIDUAL_LSTM96_EMBED)
extern "C" const unsigned char _binary_fxrl96_blob_embed_bin_start[];
extern "C" const unsigned char _binary_fxrl96_blob_embed_bin_end[];
#endif

Decoder::Decoder(std::ifstream* is, Predictor* p) : is_(is), x1_(0),
    x2_(0xffffffff), x_(0), p_(p) {
#ifdef KH_BITLSTM32
  if (const char* blob = std::getenv("KH_BITLSTM32")) {
    if (blob[0]) {
      bitlstm_.reset(new KhBitLstm32Head(blob));
      if (!bitlstm_->ok()) bitlstm_.reset();
    }
  }
#ifdef KH_BITLSTM32_EMBED
  else {
    bitlstm_.reset(new KhBitLstm32Head(
        _binary_kh_blob_embed_bin_start,
        static_cast<size_t>(_binary_kh_blob_embed_bin_end -
                            _binary_kh_blob_embed_bin_start)));
    if (!bitlstm_->ok()) bitlstm_.reset();
  }
#elif defined(KH_BITLSTM32_ARCHIVE)
  else {
    bitlstm_.reset(new KhBitLstm32Head(".head_blob_decomp"));
    if (!bitlstm_->ok()) bitlstm_.reset();
  }
#endif
#if defined(KH_BITLSTM32_EMBED) && defined(KH_BITLSTM32_ARCHIVE)
#error "KH_BITLSTM32_EMBED and KH_BITLSTM32_ARCHIVE are mutually exclusive"
#endif
#ifdef KH_BITLSTM32_REQUIRED
  if (!bitlstm_) {
    std::fprintf(stderr,
        "FX4 target build requires the packaged BitLSTM32 model\n");
    std::exit(2);
  }
#endif
#endif
#if FX4_RESIDUAL_LSTM96
  if (const char* blob = std::getenv("KH_RESIDUAL_LSTM96")) {
    if (blob[0]) {
      residual_lstm96_.reset(new KhResidualLstm96Head(blob));
      if (!residual_lstm96_->ok()) residual_lstm96_.reset();
    }
  }
#ifdef KH_RESIDUAL_LSTM96_EMBED
  else {
    residual_lstm96_.reset(new KhResidualLstm96Head(
        _binary_fxrl96_blob_embed_bin_start,
        static_cast<size_t>(_binary_fxrl96_blob_embed_bin_end -
                            _binary_fxrl96_blob_embed_bin_start)));
    if (!residual_lstm96_->ok()) residual_lstm96_.reset();
  }
#elif defined(KH_RESIDUAL_LSTM96_ARCHIVE)
  else {
    residual_lstm96_.reset(
        new KhResidualLstm96Head(".residual_lstm96_blob_decomp"));
    if (!residual_lstm96_->ok()) residual_lstm96_.reset();
  }
#endif
#if defined(KH_RESIDUAL_LSTM96_EMBED) && \
    defined(KH_RESIDUAL_LSTM96_ARCHIVE)
#error "KH_RESIDUAL_LSTM96_EMBED and KH_RESIDUAL_LSTM96_ARCHIVE conflict"
#endif
#endif
  for (int i = 0; i < 4; ++i) {
    x_ = (x_ << 8) + (ReadByte() & 0xff);
  }
}

int Decoder::ReadByte() {
  int byte = (unsigned char)(is_->get());
  if (!is_->good()) return 0;
  return byte;
}

unsigned int Decoder::Discretize(float p) {
  return 1 + 65534 * p;
}

int Decoder::Decode() {
  unsigned int p = Discretize(p_->Predict());
#if FX4_RESIDUAL_LSTM96
  const unsigned int base_p = p;
#endif
#ifdef KH_BITLSTM32
  if (bitlstm_) {
    p = bitlstm_->Adjust(p, p_->kh_stage1_in_, p_->kh_stage1_n_,
        p_->kh_m1raw_, p_->kh_override_);
  }
#endif
#if FX4_RESIDUAL_LSTM96
  if (residual_lstm96_) {
    p = residual_lstm96_->Adjust(base_p, p,
        p_->trace_ppmd_probability_, p_->trace_lstm_probability_,
        p_->trace_fxcm_probability_, p_->trace_bit_position_,
        p_->trace_ppmd_order_, p_->trace_escape_depth_,
        p_->trace_match_length_, p_->trace_stream_class_, p_->kh_override_);
  }
#endif
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  int bit = 0;
  if (x_ <= xmid) {
    bit = 1;
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
#ifdef KH_BITLSTM32
  if (bitlstm_) bitlstm_->Observe(bit);
#endif
#if FX4_RESIDUAL_LSTM96
  if (residual_lstm96_) residual_lstm96_->Observe(bit);
#endif
  p_->Perceive(bit);

  while (((x1_^x2_) & 0xff000000) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
    x_ = (x_ << 8) + ReadByte();
  }
  return bit;
}

int Decoder::DecodeRawBit(unsigned int p) {
  if (p < 1) p = 1;
  if (p > 65534) p = 65534;
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  int bit = 0;
  if (x_ <= xmid) {
    bit = 1;
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }

  while (((x1_^x2_) & 0xff000000) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
    x_ = (x_ << 8) + ReadByte();
  }
  return bit;
}

void Decoder::ObserveKnownBit(int bit) {
  p_->Predict();
  p_->Perceive(bit);
}

void Decoder::ObserveKnownByte(unsigned int byte) {
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
}
