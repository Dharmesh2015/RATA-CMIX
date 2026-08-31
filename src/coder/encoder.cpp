#include "encoder.h"

#include <algorithm>
#include <cmath>
#if defined(KH_BITLSTM32) || FX4_RESIDUAL_ORACLE_TRACE || \
    FX4_RESIDUAL_LSTM96
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

namespace {

void WriteU16(std::ostream* output, std::uint16_t value) {
  output->put(static_cast<char>(value));
  output->put(static_cast<char>(value >> 8));
}

void WriteU64(std::ostream* output, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    output->put(static_cast<char>(value >> shift));
  }
}

#if FX4_RESIDUAL_ORACLE_TRACE
#pragma pack(push, 1)
struct Fx4OracleRecord {
  std::uint16_t base_p;
  std::uint16_t final_p;
  std::uint8_t ppmd_logit;
  std::uint8_t lstm_logit;
  std::uint8_t fxcm_logit;
  std::uint8_t flags;
  std::uint8_t ppm_meta;
  std::uint8_t match_length;
};
#pragma pack(pop)
static_assert(sizeof(Fx4OracleRecord) == 10,
    "FX4 oracle record must remain 10 bytes");

std::uint8_t QuantizeLogit(float p) {
  p = std::max(0.00033535f, std::min(0.99966465f, p));
  const float z = std::log(p / (1.0f - p));
  const int q = static_cast<int>(std::lround((z + 8.0f) * (255.0f / 16.0f)));
  return static_cast<std::uint8_t>(std::max(0, std::min(255, q)));
}
#endif

}  // namespace

Encoder::Encoder(std::ofstream* os, Predictor* p)
    : Encoder(os, p, true) {}

Encoder::Encoder(std::ofstream* os, Predictor* p, bool load_terminal_head)
    : os_(os), x1_(0), x2_(0xffffffff), p_(p) {
#ifdef KH_BITLSTM32
  if (!load_terminal_head) return;
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
#endif
#ifdef KH_BITLSTM32_REQUIRED
  if (!bitlstm_) {
    std::fprintf(stderr,
        "FX4 target build requires the packaged BitLSTM32 model\n");
    std::exit(2);
  }
#endif
#else
  (void)load_terminal_head;
#endif
#if FX4_RESIDUAL_LSTM96
  if (!load_terminal_head) return;
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
#endif
#endif
}

void Encoder::WriteByte(unsigned int byte) {
  if (count_only_) {
    ++count_only_bytes_;
  } else {
    out_.push_back(byte);
  }
}

unsigned int Encoder::Discretize(float p) {
  return 1 + 65534 * p;
}

unsigned int Encoder::DiscretizeProbability(float p) {
  return 1 + 65534 * p;
}

Encoder Encoder::Clone() const {
  Encoder branch(os_, p_, false);
  branch.x1_ = x1_;
  branch.x2_ = x2_;
  branch.out_ = out_;
  branch.count_only_ = count_only_;
  branch.count_only_bytes_ = count_only_bytes_;
#ifdef KH_BITLSTM32
  if (bitlstm_) branch.bitlstm_.reset(new KhBitLstm32Head(*bitlstm_));
#endif
#if FX4_RESIDUAL_LSTM96
  if (residual_lstm96_) {
    branch.residual_lstm96_.reset(
        new KhResidualLstm96Head(*residual_lstm96_));
  }
#endif
  return branch;
}

Encoder Encoder::CloneCountOnly() const {
  Encoder branch(os_, p_, false);
  branch.x1_ = x1_;
  branch.x2_ = x2_;
  branch.count_only_ = true;
  branch.count_only_bytes_ = OutputSize();
#ifdef KH_BITLSTM32
  if (bitlstm_) branch.bitlstm_.reset(new KhBitLstm32Head(*bitlstm_));
#endif
#if FX4_RESIDUAL_LSTM96
  if (residual_lstm96_) {
    branch.residual_lstm96_.reset(
        new KhResidualLstm96Head(*residual_lstm96_));
  }
#endif
  return branch;
}

void Encoder::Encode(int bit) {
  unsigned int p = Discretize(p_->Predict());
  const unsigned int base_p = p;
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
#if FX4_RESIDUAL_ORACLE_TRACE
  WriteOracleRecord(bit, base_p, p);
#endif
  if (trace_byte_active_) {
    const unsigned int mass = bit ? p : 65536u - p;
    trace_cost_bits_ -= std::log2(static_cast<double>(mass) / 65536.0);
  }
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
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
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::EncodeRawBit(int bit, unsigned int p) {
  if (p < 1) p = 1;
  if (p > 65534) p = 65534;
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }

  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::ObserveKnownBit(int bit) {
  p_->Predict();
  p_->Perceive(bit);
}

void Encoder::ObserveKnownByte(unsigned int byte) {
  for (int bit = 7; bit >= 0; --bit) {
    ObserveKnownBit((byte >> bit) & 1);
  }
}

double Encoder::ObserveKnownBitCost(int bit) {
  const float probability = p_->Predict();
  const double mass = bit ? probability : 1.0 - probability;
  p_->Perceive(bit);
  return -std::log2(mass);
}

double Encoder::ObserveKnownByteCost(unsigned int byte) {
  double cost = 0.0;
  for (int bit = 7; bit >= 0; --bit) {
    cost += ObserveKnownBitCost((byte >> bit) & 1);
  }
  return cost;
}

void Encoder::BeginTraceByte(unsigned long long offset, unsigned int actual_byte,
    unsigned int prev4) {
  (void)offset;
  (void)actual_byte;
  (void)prev4;
  if (cost_trace_.is_open()) {
    trace_cost_bits_ = 0.0;
    trace_byte_active_ = true;
  }
}

void Encoder::EndTraceByte() {
  if (!trace_byte_active_) return;
  const float cost = static_cast<float>(trace_cost_bits_);
  cost_trace_.write(reinterpret_cast<const char*>(&cost), sizeof(cost));
  trace_byte_active_ = false;
}

bool Encoder::StartCostTrace(const char* path, std::uint64_t stream_size) {
  if (!path || !*path) return true;
  cost_trace_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!cost_trace_.is_open()) return false;
  cost_trace_.write("F4TC", 4);
  WriteU16(&cost_trace_, 1);
  WriteU16(&cost_trace_, 0);
  WriteU64(&cost_trace_, stream_size);
  return cost_trace_.good();
}

#if FX4_RESIDUAL_ORACLE_TRACE
bool Encoder::StartOracleTrace(std::uint64_t stream_size) {
  const char* path = std::getenv("FX4_ORACLE_TRACE");
  if (!path || !path[0]) return true;
  std::uint64_t minimum = 100000000ull;
  if (const char* value = std::getenv("FX4_ORACLE_TRACE_MIN_BYTES")) {
    if (value[0]) minimum = std::strtoull(value, nullptr, 10);
  }
  if (stream_size < minimum) return true;
  std::uint64_t limit_bytes = stream_size;
  if (const char* value = std::getenv("FX4_ORACLE_TRACE_BYTES")) {
    if (value[0]) limit_bytes = std::min<std::uint64_t>(
        stream_size, std::strtoull(value, nullptr, 10));
  }
  if (limit_bytes == 0) return false;
  oracle_trace_.open(path, std::ios::binary | std::ios::trunc);
  if (!oracle_trace_) return false;
  oracle_trace_.write("FXOT", 4);
  WriteU16(&oracle_trace_, 1);
  WriteU16(&oracle_trace_, sizeof(Fx4OracleRecord));
  WriteU64(&oracle_trace_, stream_size);
  WriteU64(&oracle_trace_, limit_bytes);
  WriteU64(&oracle_trace_, 0);
  oracle_record_limit_ = limit_bytes * 8;
  return oracle_trace_.good();
}

void Encoder::WriteOracleRecord(int bit, unsigned int base_p,
    unsigned int final_p) {
  if (!oracle_trace_ || oracle_record_count_ >= oracle_record_limit_) return;
  Fx4OracleRecord record{};
  record.base_p = static_cast<std::uint16_t>(base_p);
  record.final_p = static_cast<std::uint16_t>(final_p);
  record.ppmd_logit = QuantizeLogit(p_->trace_ppmd_probability_);
  record.lstm_logit = QuantizeLogit(p_->trace_lstm_probability_);
  record.fxcm_logit = QuantizeLogit(p_->trace_fxcm_probability_);
  record.flags = static_cast<std::uint8_t>((bit & 1) |
      ((p_->kh_override_ ? 1u : 0u) << 1) |
      ((p_->trace_bit_position_ & 7u) << 2) |
      ((p_->trace_stream_class_ & 7u) << 5));
  record.ppm_meta = static_cast<std::uint8_t>(
      (p_->trace_ppmd_order_ << 3) | (p_->trace_escape_depth_ & 7u));
  record.match_length = p_->trace_match_length_;
  oracle_trace_.write(reinterpret_cast<const char*>(&record), sizeof(record));
  ++oracle_record_count_;
}
#endif

void Encoder::Flush() {
  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
  WriteByte(x2_ >> 24);

  auto* data = reinterpret_cast<const char*>(out_.data());
  os_->write(data, out_.size());
  if (cost_trace_.is_open()) cost_trace_.flush();
#if FX4_RESIDUAL_ORACLE_TRACE
  if (oracle_trace_.is_open()) {
    const std::streampos end = oracle_trace_.tellp();
    oracle_trace_.seekp(24, std::ios::beg);
    WriteU64(&oracle_trace_, oracle_record_count_);
    oracle_trace_.seekp(end);
    oracle_trace_.close();
  }
#endif
}

size_t Encoder::ProjectedFinalOutputSize() const {
  // Identical renormalization logic to Flush(), operating on local copies
  // of x1_/x2_ so the live encoder state is untouched.
  unsigned int x1 = x1_;
  unsigned int x2 = x2_;
  size_t pending_bytes = 0;
  while (((x1 ^ x2) & 0xff000000) == 0) {
    ++pending_bytes;
    x1 <<= 8;
    x2 = (x2 << 8) + 255;
  }
  ++pending_bytes;  // Flush()'s unconditional final WriteByte(x2 >> 24).
  return OutputSize() + pending_bytes;
}
