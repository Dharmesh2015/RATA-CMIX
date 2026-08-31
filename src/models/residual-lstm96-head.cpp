#include "../fx4_config.h"

#if FX4_RESIDUAL_LSTM96

#include "residual-lstm96-head.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr int kInput = 32;
constexpr int kHidden = 96;
constexpr int kOutput = 8;
constexpr int kLinear = 8;
constexpr int kContexts = 512;
constexpr int kConfidenceStates = 7;

#pragma pack(push, 1)
struct BlobHeader {
  char magic[8];                 // {'F','X','R','L','9','6',1,0}
  std::uint16_t input;
  std::uint16_t hidden;
  std::uint16_t output;
  std::uint16_t flags;           // bit 0: symmetric int8 row weights
  float correction_limit;
  std::uint32_t payload_bytes;
  std::uint32_t payload_crc32;
  std::uint32_t reset_bytes;
};
#pragma pack(pop)
static_assert(sizeof(BlobHeader) == 32, "FXRL96 header size changed");

std::uint32_t Crc32(const unsigned char* data, size_t size) {
  std::uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
  }
  return ~crc;
}

float HalfToFloat(std::uint16_t h) {
#if defined(__F16C__)
  return _cvtsh_ss(h);
#else
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t mant = h & 0x3ffu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
      bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

float ProbabilityLogit(float p) {
  p = std::max(1.0f / 65536.0f, std::min(65535.0f / 65536.0f, p));
  return std::log(p) - std::log1p(-p);
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

#if defined(__AVX2__) && defined(__FMA__)
__m256 Exp8(__m256 x) {
  const __m256 hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
  const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
  const __m256 one = _mm256_set1_ps(1.0f);
  x = _mm256_max_ps(lo, _mm256_min_ps(hi, x));
  __m256 fx = _mm256_floor_ps(
      _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, c1, x);
  x = _mm256_fnmadd_ps(fx, c2, x);
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = p0;
  y = _mm256_fmadd_ps(y, x, p1);
  y = _mm256_fmadd_ps(y, x, p2);
  y = _mm256_fmadd_ps(y, x, p3);
  y = _mm256_fmadd_ps(y, x, p4);
  y = _mm256_fmadd_ps(y, x, p5);
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, one);
  const __m256i exponent = _mm256_add_epi32(_mm256_cvttps_epi32(fx),
      _mm256_set1_epi32(0x7f));
  return _mm256_mul_ps(y, _mm256_castsi256_ps(
      _mm256_slli_epi32(exponent, 23)));
}

__m256 Sigmoid8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_div_ps(one, _mm256_add_ps(one,
      Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

__m256 Tanh8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 e = Exp8(_mm256_mul_ps(x, _mm256_set1_ps(-2.0f)));
  return _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
}
#endif

float Dot(const float* weights, const float* values, int count) {
#if defined(__AVX2__) && defined(__FMA__)
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < count; i += 8) {
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(weights + i),
                          _mm256_loadu_ps(values + i), sum);
  }
  __m128 lanes = _mm_add_ps(_mm256_castps256_ps128(sum),
      _mm256_extractf128_ps(sum, 1));
  lanes = _mm_add_ps(lanes, _mm_movehl_ps(lanes, lanes));
  lanes = _mm_add_ss(lanes, _mm_shuffle_ps(lanes, lanes, 1));
  return _mm_cvtss_f32(lanes);
#else
  float sums[8] = {0};
  for (int i = 0; i < count; i += 8) {
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += weights[i + lane] * values[i + lane];
    }
  }
  return ((sums[0] + sums[4]) + (sums[2] + sums[6])) +
         ((sums[1] + sums[5]) + (sums[3] + sums[7]));
#endif
}

int ConfidenceState(float absolute_logit) {
  static constexpr float boundary[6] = {0.5f, 1.0f, 2.0f,
                                         3.0f, 4.0f, 6.0f};
  int state = 0;
  while (state < 6 && absolute_logit >= boundary[state]) ++state;
  return state;
}

}  // namespace

struct KhResidualLstm96Head::Impl {
  float wih[4 * kHidden * kInput];
  float whh[4 * kHidden * kHidden];
  float bias[4 * kHidden];
  float wout[kOutput * kHidden];
  float bout[kOutput];
  float wlinear[kOutput * kLinear];
  float context_delta[kContexts];
  float confidence_gate[kConfidenceStates];
  float correction_limit = 0.0f;
  unsigned int reset_bytes = 256;

  float hidden[kHidden] = {0};
  float cell[kHidden] = {0};
  float next_delta[kOutput] = {0};
  unsigned long long bit_index = 0;

  float pending_p = 0.5f;
  float pending_final_z = 0.0f;
  float pending_ppmd_z = 0.0f;
  float pending_lstm_z = 0.0f;
  float pending_fxcm_z = 0.0f;
  unsigned int pending_order = 0;
  unsigned int pending_escape = 0;
  unsigned int pending_match = 0;
  unsigned int pending_stream = 0;
  unsigned int pending_bit_position = 0;

  float byte_final_z[8] = {0};
  float byte_residual[8] = {0};
  float byte_ppmd_gap[8] = {0};
  float byte_lstm_gap[8] = {0};
  float byte_fxcm_gap[8] = {0};
  float byte_bits[8] = {0};
  float byte_order_sum = 0.0f;
  float byte_escape_sum = 0.0f;
  unsigned int byte_match_max = 0;
  unsigned int byte_stream = 0;

  Impl() = default;
  Impl(const Impl& other) { std::memcpy(this, &other, sizeof(*this)); }

  bool Load(const char* path);
  bool LoadMemory(const unsigned char* data, size_t size);
  void StepByte();
};

bool KhResidualLstm96Head::Impl::Load(const char* path) {
  FILE* file = std::fopen(path, "rb");
  if (!file) return false;
  if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); return false; }
  const long length = std::ftell(file);
  if (length <= 0) { std::fclose(file); return false; }
  std::rewind(file);
  unsigned char* bytes = static_cast<unsigned char*>(
      std::malloc(static_cast<size_t>(length)));
  if (!bytes) { std::fclose(file); return false; }
  const bool read_ok = std::fread(bytes, 1, static_cast<size_t>(length), file) ==
      static_cast<size_t>(length);
  std::fclose(file);
  const bool ok = read_ok && LoadMemory(bytes, static_cast<size_t>(length));
  std::free(bytes);
  return ok;
}

bool KhResidualLstm96Head::Impl::LoadMemory(const unsigned char* data,
                                            size_t size) {
  if (!data || size < sizeof(BlobHeader)) return false;
  BlobHeader header;
  std::memcpy(&header, data, sizeof(header));
  const unsigned char expected_magic[8] = {'F','X','R','L','9','6',1,0};
  if (std::memcmp(header.magic, expected_magic, sizeof(expected_magic)) != 0 ||
      header.input != kInput || header.hidden != kHidden ||
      header.output != kOutput || header.flags != 1u ||
      header.payload_bytes != size - sizeof(header) ||
      header.reset_bytes == 0 || header.reset_bytes > (1u << 20) ||
      !std::isfinite(header.correction_limit) ||
      header.correction_limit <= 0.0f || header.correction_limit > 8.0f) {
    return false;
  }
  const unsigned char* payload = data + sizeof(header);
  if (Crc32(payload, header.payload_bytes) != header.payload_crc32) return false;
  size_t position = 0;
  auto read_half = [&](float* output, size_t count) -> bool {
    if (header.payload_bytes - position < count * sizeof(std::uint16_t)) {
      return false;
    }
    for (size_t i = 0; i < count; ++i) {
      std::uint16_t value;
      std::memcpy(&value, payload + position + i * 2, 2);
      output[i] = HalfToFloat(value);
      if (!std::isfinite(output[i])) return false;
    }
    position += count * 2;
    return true;
  };
  auto read_qmatrix = [&](float* output, int rows, int columns) -> bool {
    float* scales = new float[rows];
    if (!read_half(scales, rows) ||
        header.payload_bytes - position < static_cast<size_t>(rows * columns)) {
      delete[] scales;
      return false;
    }
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < columns; ++column) {
        const std::int8_t q = static_cast<std::int8_t>(
            payload[position + row * columns + column]);
        output[row * columns + column] = scales[row] * static_cast<float>(q);
      }
    }
    position += static_cast<size_t>(rows * columns);
    delete[] scales;
    return true;
  };

  bool ok = read_qmatrix(wih, 4 * kHidden, kInput) &&
      read_qmatrix(whh, 4 * kHidden, kHidden) &&
      read_half(bias, 4 * kHidden) &&
      read_qmatrix(wout, kOutput, kHidden) &&
      read_half(bout, kOutput) &&
      read_qmatrix(wlinear, kOutput, kLinear) &&
      read_half(context_delta, kContexts) &&
      read_half(confidence_gate, kConfidenceStates);
  if (!ok || position != header.payload_bytes) return false;
  correction_limit = header.correction_limit;
  reset_bytes = header.reset_bytes;
  return true;
}

void KhResidualLstm96Head::Impl::StepByte() {
  const unsigned long long source_byte = (bit_index >> 3) - 1;
  if ((source_byte % reset_bytes) == 0) {
    std::memset(hidden, 0, sizeof(hidden));
    std::memset(cell, 0, sizeof(cell));
  }
  float input[kInput];
  input[0] = 1.0f;
  for (int bit = 0; bit < 8; ++bit) {
    input[1 + bit] = byte_bits[bit];
    input[9 + bit] = std::max(-1.0f,
        std::min(1.0f, byte_final_z[bit] * 0.125f));
    input[17 + bit] = byte_residual[bit] * 4.0f;
  }
  float ppmd_gap = 0.0f, lstm_gap = 0.0f, fxcm_gap = 0.0f;
  for (int bit = 0; bit < 8; ++bit) {
    ppmd_gap += byte_ppmd_gap[bit];
    lstm_gap += byte_lstm_gap[bit];
    fxcm_gap += byte_fxcm_gap[bit];
  }
  input[25] = std::max(-1.0f, std::min(1.0f, ppmd_gap / 48.0f));
  input[26] = std::max(-1.0f, std::min(1.0f, lstm_gap / 48.0f));
  input[27] = std::max(-1.0f, std::min(1.0f, fxcm_gap / 48.0f));
  input[28] = byte_order_sum / (8.0f * 31.0f);
  input[29] = byte_escape_sum / (8.0f * 7.0f);
  input[30] = std::log1p(static_cast<float>(byte_match_max)) /
      std::log(256.0f);
  input[31] = static_cast<float>(std::min(byte_stream, 7u)) / 7.0f;

  float pre[4 * kHidden];
  for (int row = 0; row < 4 * kHidden; ++row) {
    pre[row] = Dot(wih + row * kInput, input, kInput) +
        Dot(whh + row * kHidden, hidden, kHidden) + bias[row];
  }
#if defined(__AVX2__) && defined(__FMA__)
  for (int row = 0; row < kHidden; row += 8) {
    const __m256 input_gate = Sigmoid8(
        _mm256_loadu_ps(pre + row));
    const __m256 forget_gate = Sigmoid8(
        _mm256_loadu_ps(pre + kHidden + row));
    const __m256 candidate = Tanh8(
        _mm256_loadu_ps(pre + 2 * kHidden + row));
    const __m256 output_gate = Sigmoid8(
        _mm256_loadu_ps(pre + 3 * kHidden + row));
    const __m256 next_cell = _mm256_fmadd_ps(forget_gate,
        _mm256_loadu_ps(cell + row), _mm256_mul_ps(input_gate, candidate));
    _mm256_storeu_ps(cell + row, next_cell);
    _mm256_storeu_ps(hidden + row,
        _mm256_mul_ps(output_gate, Tanh8(next_cell)));
  }
#else
  for (int row = 0; row < kHidden; ++row) {
    const float input_gate = Sigmoid(pre[row]);
    const float forget_gate = Sigmoid(pre[kHidden + row]);
    const float candidate = std::tanh(pre[2 * kHidden + row]);
    const float output_gate = Sigmoid(pre[3 * kHidden + row]);
    cell[row] = forget_gate * cell[row] + input_gate * candidate;
    hidden[row] = output_gate * std::tanh(cell[row]);
  }
#endif
  for (int bit = 0; bit < kOutput; ++bit) {
    next_delta[bit] = Dot(wout + bit * kHidden, hidden, kHidden) + bout[bit];
  }

  byte_order_sum = 0.0f;
  byte_escape_sum = 0.0f;
  byte_match_max = 0;
}

KhResidualLstm96Head::KhResidualLstm96Head(const char* blob_path) {
  impl_ = new Impl();
  if (!impl_->Load(blob_path)) {
    std::fprintf(stderr, "\n*** FXRL96 WARNING: cannot load '%s'; disabled ***\n",
        blob_path ? blob_path : "");
    delete impl_;
    impl_ = nullptr;
    return;
  }
  ok_ = true;
}

KhResidualLstm96Head::KhResidualLstm96Head(const unsigned char* blob_data,
                                           size_t blob_size) {
  impl_ = new Impl();
  if (!impl_->LoadMemory(blob_data, blob_size)) {
    std::fprintf(stderr, "\n*** FXRL96 WARNING: invalid embedded blob; disabled ***\n");
    delete impl_;
    impl_ = nullptr;
    return;
  }
  ok_ = true;
}

KhResidualLstm96Head::KhResidualLstm96Head(
    const KhResidualLstm96Head& other) : ok_(other.ok_) {
  if (other.impl_) impl_ = new Impl(*other.impl_);
}

KhResidualLstm96Head::~KhResidualLstm96Head() { delete impl_; }

unsigned int KhResidualLstm96Head::Adjust(unsigned int base_p,
    unsigned int current_p, float ppmd_probability, float lstm_probability,
    float fxcm_probability, unsigned int bit_position,
    unsigned int ppmd_order, unsigned int escape_depth,
    unsigned int match_length, unsigned int stream_class,
    int override_active) {
  Impl& state = *impl_;
  bit_position &= 7u;
  ppmd_order = std::min(ppmd_order, 31u);
  escape_depth = std::min(escape_depth, 7u);
  stream_class = std::min(stream_class, 7u);
  match_length = std::min(match_length, 255u);

  const float base_z = ProbabilityLogit(static_cast<float>(base_p) / 65536.0f);
  const float final_p = static_cast<float>(current_p) / 65536.0f;
  const float final_z = ProbabilityLogit(final_p);
  const float ppmd_z = ProbabilityLogit(ppmd_probability);
  const float lstm_z = ProbabilityLogit(lstm_probability);
  const float fxcm_z = ProbabilityLogit(fxcm_probability);
  const float mean = (ppmd_z + lstm_z + fxcm_z + final_z) * 0.25f;
  const float variance = ((ppmd_z - mean) * (ppmd_z - mean) +
      (lstm_z - mean) * (lstm_z - mean) +
      (fxcm_z - mean) * (fxcm_z - mean) +
      (final_z - mean) * (final_z - mean)) * 0.25f;
  float linear[kLinear] = {
      std::max(-1.0f, std::min(1.0f, (ppmd_z - final_z) / 6.0f)),
      std::max(-1.0f, std::min(1.0f, (lstm_z - final_z) / 6.0f)),
      std::max(-1.0f, std::min(1.0f, (fxcm_z - final_z) / 6.0f)),
      static_cast<float>(ppmd_order) / 31.0f,
      static_cast<float>(escape_depth) / 7.0f,
      std::log1p(static_cast<float>(match_length)) / std::log(256.0f),
      std::min(1.0f, std::sqrt(variance) / 4.0f),
      std::max(-1.0f, std::min(1.0f, (final_z - base_z) / 4.0f)),
  };
  const int context = static_cast<int>(bit_position +
      8u * std::min(ppmd_order / 4u, 7u) + 64u * stream_class);
  const int confidence = ConfidenceState(std::fabs(final_z));
  float correction = state.confidence_gate[confidence] *
      (state.next_delta[bit_position] +
       Dot(state.wlinear + bit_position * kLinear, linear, kLinear)) +
      state.context_delta[context];
  correction = std::max(-state.correction_limit,
      std::min(state.correction_limit, correction));

  state.pending_p = final_p;
  state.pending_final_z = final_z;
  state.pending_ppmd_z = ppmd_z;
  state.pending_lstm_z = lstm_z;
  state.pending_fxcm_z = fxcm_z;
  state.pending_order = ppmd_order;
  state.pending_escape = escape_depth;
  state.pending_match = match_length;
  state.pending_stream = stream_class;
  state.pending_bit_position = bit_position;

  if (override_active || correction == 0.0f) return current_p;
  const float adjusted = Sigmoid(final_z + correction);
  unsigned int result = static_cast<unsigned int>(1.0f + 65534.0f * adjusted);
  return std::max(1u, std::min(65535u, result));
}

void KhResidualLstm96Head::Observe(int bit) {
  Impl& state = *impl_;
  const unsigned int position = state.pending_bit_position & 7u;
  const float actual = bit ? 1.0f : 0.0f;
  state.byte_bits[position] = bit ? 1.0f : -1.0f;
  state.byte_final_z[position] = state.pending_final_z;
  state.byte_residual[position] = actual - state.pending_p;
  state.byte_ppmd_gap[position] = state.pending_ppmd_z - state.pending_final_z;
  state.byte_lstm_gap[position] = state.pending_lstm_z - state.pending_final_z;
  state.byte_fxcm_gap[position] = state.pending_fxcm_z - state.pending_final_z;
  state.byte_order_sum += static_cast<float>(state.pending_order);
  state.byte_escape_sum += static_cast<float>(state.pending_escape);
  state.byte_match_max = std::max(state.byte_match_max, state.pending_match);
  if (position == 0u) state.byte_stream = state.pending_stream;
  ++state.bit_index;
  if (position == 7u) state.StepByte();
}

#endif  // FX4_RESIDUAL_LSTM96
