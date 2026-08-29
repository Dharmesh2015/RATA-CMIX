#include "encoder.h"

#include <cmath>

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

}  // namespace

Encoder::Encoder(std::ofstream* os, Predictor* p) : os_(os), x1_(0),
    x2_(0xffffffff), p_(p) {}

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
  Encoder branch(os_, p_);
  branch.x1_ = x1_;
  branch.x2_ = x2_;
  branch.out_ = out_;
  branch.count_only_ = count_only_;
  branch.count_only_bytes_ = count_only_bytes_;
  return branch;
}

Encoder Encoder::CloneCountOnly() const {
  Encoder branch(os_, p_);
  branch.x1_ = x1_;
  branch.x2_ = x2_;
  branch.count_only_ = true;
  branch.count_only_bytes_ = OutputSize();
  return branch;
}

void Encoder::Encode(int bit) {
  const unsigned int p = Discretize(p_->Predict());
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
