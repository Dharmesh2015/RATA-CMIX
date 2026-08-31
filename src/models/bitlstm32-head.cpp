// KH_BITLSTM32 head implementation. See BITLSTM32_SPEC.md for the exact
// feature contract this file implements (it mirrors head_blocks/
// full_stream_head.py FeatureState.build + continuous_shuffled_lstm.py).
//
// This translation unit MUST be compiled with value-safe floating point
// (-ffp-model=precise, see makefile rule) and its entry points are noinline:
// encode and decode then execute the identical machine code, so the head is
// deterministic across the two directions by construction.

#ifdef KH_BITLSTM32

#include "bitlstm32-head.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr float kTScale = 12.203f;
constexpr int kNin = 92;
constexpr int kHid = 32;
constexpr int kSeqReset = 64;

// Trailing-window families (bit counts), exactly the trainer's RES_W / SUR_W.
constexpr int kResW[5] = {32, 256, 2048, 16384, 131072};
constexpr int kSurW[6] = {16, 64, 512, 4096, 65536, 1048576};
constexpr int kResCap = 131072;
constexpr int kSurCap = 1048576;
constexpr int kSdCap = 256;
constexpr int kErelCap = 512;
constexpr int kShockW = 256;
// The 7 "top" expert columns whose per-expert surprisal is window-averaged.
constexpr int kTop[7] = {0, 5, 11, 17, 22, 23, 24};

// enwik9 coded-stream zone constants (bytes), as in the trainer. Stored as
// float, matching torch.tensor(..., dtype=float32).
constexpr float kZB[4] = {541126651.0f, 554726452.0f, 571499539.0f,
                          586459321.0f};
constexpr float kZLo[5] = {0.0f, 541126651.0f, 554726452.0f, 571499539.0f,
                           586459321.0f};
constexpr float kZHi[5] = {541126651.0f, 554726452.0f, 571499539.0f,
                           586459321.0f, 587138826.0f};
constexpr float kCorpusBits = 4697110608.0f;  // 587,138,826 * 8 as float32

// float -> f16 -> float round trip (round-to-nearest-even), reproducing the
// KH_TRACE quantization the training data went through.
#if defined(__F16C__)
inline float F16RoundTrip(float f) {
  return _cvtsh_ss(_cvtss_sh(f, (_MM_FROUND_TO_NEAREST_INT |
                                 _MM_FROUND_NO_EXC)));
}
#else
inline uint16_t F16FromF32(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t abs_exp = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;
  if (abs_exp == 0xffu) {
    uint16_t h = (uint16_t)(sign | 0x7c00u);
    if (mant) h |= (uint16_t)(0x0200u | (mant >> 13));
    return h;
  }
  int exp = (int)abs_exp - 127 + 15;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t hm = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1u))) ++hm;
    return (uint16_t)(sign | hm);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return h;
}
inline float F32FromF16(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal
      int e = -1;
      uint32_t m = mant;
      while (!(m & 0x400u)) { m <<= 1; --e; }
      m &= 0x3ffu;
      x = sign | ((uint32_t)(127 - 15 + e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}
inline float F16RoundTrip(float f) { return F32FromF16(F16FromF32(f)); }
#endif

// np.nan_to_num(nan=0, posinf=T_SCALE, neginf=-T_SCALE) applied after the
// f16 decode, exactly as the trainer does.
inline float MapExpert(float v) {
  if (std::isnan(v)) return 0.0f;
  if (std::isinf(v)) return v > 0 ? kTScale : -kTScale;
  return v;
}

inline float Sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Sgn(float x) { return (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

#if defined(__AVX2__) && defined(__FMA__)
// 8-lane expf, Cephes polynomial (max ~2 ulp). Fixed coefficients and fixed
// operation order -> deterministic. Used for the ~192 gate activations per
// coded bit; feature-side transcendentals stay scalar libm.
inline __m256 Exp8(__m256 x) {
  const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
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
  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);
  __m256 fx = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
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
  const __m256i imm = _mm256_add_epi32(_mm256_cvttps_epi32(fx),
                                       _mm256_set1_epi32(0x7f));
  const __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(imm, 23));
  return _mm256_mul_ps(y, pow2n);
}
inline __m256 Sigmoid8(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_div_ps(one, _mm256_add_ps(one,
      Exp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}
inline __m256 Tanh8(__m256 x) {
  // tanh(x) = (1 - e^{-2x}) / (1 + e^{-2x}); Exp8's +-88 clamp keeps the
  // ratio finite and saturating at +-1.
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 e = Exp8(_mm256_mul_ps(x, _mm256_set1_ps(-2.0f)));
  return _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
}
inline __m256 Silu8(__m256 x) { return _mm256_mul_ps(x, Sigmoid8(x)); }
#endif

// Deterministic dense matvec y[i] = sum_j W[i*stride+j]*x[j] + b[i].
// The AVX2 path uses a fixed 8-lane accumulator and a fixed-order horizontal
// reduction, so results are identical on every call with the same inputs.
#if defined(__AVX2__) && defined(__FMA__)
inline float DotPadded(const float* w, const float* x, int n_padded) {
  __m256 acc = _mm256_setzero_ps();
  for (int j = 0; j < n_padded; j += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(x + j), acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}
#else
inline float DotPadded(const float* w, const float* x, int n_padded) {
  float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  float acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
  for (int j = 0; j < n_padded; j += 8) {
    acc0 += w[j] * x[j];         acc1 += w[j + 1] * x[j + 1];
    acc2 += w[j + 2] * x[j + 2]; acc3 += w[j + 3] * x[j + 3];
    acc4 += w[j + 4] * x[j + 4]; acc5 += w[j + 5] * x[j + 5];
    acc6 += w[j + 6] * x[j + 6]; acc7 += w[j + 7] * x[j + 7];
  }
  return ((acc0 + acc4) + (acc2 + acc6)) + ((acc1 + acc5) + (acc3 + acc7));
}
#endif

struct WindowRing {
  float* buf = nullptr;
  int cap = 0;
  void Init(int c) { cap = c; buf = new float[c](); }
  ~WindowRing() { delete[] buf; }
};

#pragma pack(push, 1)
struct BlobHeader {
  char magic[8];      // "KHBL32\x01\0"
  uint32_t nin, hid, seq_reset, reserved;
};
#pragma pack(pop)

}  // namespace

struct KhBitLstm32Head::Impl {
  // ---- weights (padded row strides for the fixed-order matvec) ----
  // in_proj: 32 rows x 93 cols, padded to 96. lstm ih/hh: 128 rows x 32 cols.
  // out: 1 row x 32 cols.
  static constexpr int kInPad = 96;
  float win[kHid * kInPad];      // in_proj.weight, zero padded
  float bin[kHid];               // in_proj.bias
  float wih[4 * kHid * kHid];    // lstm.weight_ih_l0 (rows: i,f,g,o)
  float whh[4 * kHid * kHid];    // lstm.weight_hh_l0
  float bih[4 * kHid];           // lstm.bias_ih_l0
  float bhh[4 * kHid];           // lstm.bias_hh_l0
  float wout[kHid];              // out.weight
  float bout;                    // out.bias

  // ---- LSTM stream state ----
  float h[kHid] = {0};
  float c[kHid] = {0};
  float prev_bit = 0.5f;

  // ---- causal feature state ----
  unsigned long long i = 0;      // coded-bit index (records seen)
  WindowRing res_ring, sur_ring, sd_ring, erel_ring;  // erel: 7 interleaved
  double res_sum[5] = {0};
  double sur_sum[6] = {0};
  double sd_sum = 0;
  double erel_sum[7] = {0};
  int shock_count = 0;                    // count of sur>4 in last kShockW bits
  unsigned long long last_shock = 0;      // trainer-initial value 0
  uint8_t hist8[8] = {0};
  unsigned long long byte_count = 0;
  unsigned int cur_byte = 0;

  // ---- pending values between Adjust() and Observe() ----
  float pend_p1 = 0.5f;
  float pend_sd = 0.0f;
  float pend_pe[7] = {0};

  Impl() = default;
  Impl(const Impl& o) {
    std::memcpy(win, o.win, sizeof(win));
    std::memcpy(bin, o.bin, sizeof(bin));
    std::memcpy(wih, o.wih, sizeof(wih));
    std::memcpy(whh, o.whh, sizeof(whh));
    std::memcpy(bih, o.bih, sizeof(bih));
    std::memcpy(bhh, o.bhh, sizeof(bhh));
    std::memcpy(wout, o.wout, sizeof(wout));
    bout = o.bout;
    std::memcpy(h, o.h, sizeof(h));
    std::memcpy(c, o.c, sizeof(c));
    prev_bit = o.prev_bit;
    i = o.i;
    res_ring.Init(o.res_ring.cap);
    sur_ring.Init(o.sur_ring.cap);
    sd_ring.Init(o.sd_ring.cap);
    erel_ring.Init(o.erel_ring.cap);
    std::memcpy(res_ring.buf, o.res_ring.buf,
        sizeof(float) * o.res_ring.cap);
    std::memcpy(sur_ring.buf, o.sur_ring.buf,
        sizeof(float) * o.sur_ring.cap);
    std::memcpy(sd_ring.buf, o.sd_ring.buf,
        sizeof(float) * o.sd_ring.cap);
    std::memcpy(erel_ring.buf, o.erel_ring.buf,
        sizeof(float) * o.erel_ring.cap);
    std::memcpy(res_sum, o.res_sum, sizeof(res_sum));
    std::memcpy(sur_sum, o.sur_sum, sizeof(sur_sum));
    sd_sum = o.sd_sum;
    std::memcpy(erel_sum, o.erel_sum, sizeof(erel_sum));
    shock_count = o.shock_count;
    last_shock = o.last_shock;
    std::memcpy(hist8, o.hist8, sizeof(hist8));
    byte_count = o.byte_count;
    cur_byte = o.cur_byte;
    pend_p1 = o.pend_p1;
    pend_sd = o.pend_sd;
    std::memcpy(pend_pe, o.pend_pe, sizeof(pend_pe));
  }

  bool LoadBlob(const char* path);
  bool LoadBlobMem(const unsigned char* data, size_t size);
  void BuildFeatures(unsigned int base_p, const float* stage1, int n_stage1,
                     float m1raw, float* x, float* t_out);
  float RunNet(const float* x, float prev);
};

// Half -> float widening for v2 (f16-stored) blobs. Exact and deterministic:
// every f16 value maps to one f32 value, identically on encode and decode.
static inline float KhF32FromF16(uint16_t h) {
#if defined(__F16C__)
  return _cvtsh_ss(h);
#else
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x3ffu;
      x = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &x, sizeof(out));
  return out;
#endif
}

// File-path loader: slurps the blob and defers to the memory parser, so the
// file and embedded (KH_BITLSTM32_EMBED) sources run the IDENTICAL parse and
// widen code and therefore produce identical weight arrays.
bool KhBitLstm32Head::Impl::LoadBlob(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
  long sz = std::ftell(f);
  if (sz < 0) { std::fclose(f); return false; }
  std::rewind(f);
  unsigned char* buf = (unsigned char*)std::malloc((size_t)sz ? (size_t)sz : 1);
  if (!buf) { std::fclose(f); return false; }
  const bool read_ok = std::fread(buf, 1, (size_t)sz, f) == (size_t)sz;
  std::fclose(f);
  const bool ok = read_ok && LoadBlobMem(buf, (size_t)sz);
  std::free(buf);
  return ok;
}

bool KhBitLstm32Head::Impl::LoadBlobMem(const unsigned char* data,
                                        size_t size) {
  if (data == nullptr) return false;
  size_t pos = 0;
  BlobHeader hd;
  bool ok = size >= sizeof(hd);
  if (ok) { std::memcpy(&hd, data, sizeof(hd)); pos = sizeof(hd); }
  ok = ok &&
       (std::memcmp(hd.magic, "KHBL32\x01", 8) == 0 ||
        std::memcmp(hd.magic, "KHBL32\x02", 8) == 0) &&
       hd.nin == (uint32_t)kNin && hd.hid == (uint32_t)kHid &&
       hd.seq_reset == (uint32_t)kSeqReset;
  const bool f16_payload = ok && hd.magic[6] == '\x02';
  // v2 blobs store the identical layout as f16 (half the shipped bytes);
  // widen to f32 once at load. Reads via this helper so both versions share
  // the layout code below.
  auto readf = [&](float* dst, size_t n) -> bool {
    if (!f16_payload) {
      if (size - pos < n * sizeof(float)) return false;
      std::memcpy(dst, data + pos, n * sizeof(float));
      pos += n * sizeof(float);
      return true;
    }
    if (size - pos < n * sizeof(uint16_t)) return false;
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, data + pos + i * sizeof(uint16_t), sizeof(uint16_t));
      dst[i] = KhF32FromF16(h);
    }
    pos += n * sizeof(uint16_t);
    return true;
  };
  if (ok) {
    // Blob layout: in_proj.weight [32][93], in_proj.bias [32],
    // weight_ih_l0 [128][32], weight_hh_l0 [128][32], bias_ih_l0 [128],
    // bias_hh_l0 [128], out.weight [32], out.bias [1]. All f32 LE.
    std::memset(win, 0, sizeof(win));
    for (int r = 0; r < kHid && ok; ++r) {
      ok = readf(win + r * kInPad, kNin + 1);
    }
    ok = ok && readf(bin, kHid);
    ok = ok && readf(wih, 4 * kHid * kHid);
    ok = ok && readf(whh, 4 * kHid * kHid);
    ok = ok && readf(bih, 4 * kHid);
    ok = ok && readf(bhh, 4 * kHid);
    ok = ok && readf(wout, kHid);
    ok = ok && readf(&bout, 1);
    // Exactly at end of blob now.
    ok = ok && pos == size;
  }
  if (ok) {
    res_ring.Init(kResCap);
    sur_ring.Init(kSurCap);
    sd_ring.Init(kSdCap);
    erel_ring.Init(kErelCap * 7);
  }
  return ok;
}

// Builds the exact 92-feature row (BITLSTM32_SPEC.md) for the CURRENT bit,
// before it is coded. All window features cover bits j < i only.
__attribute__((noinline))
void KhBitLstm32Head::Impl::BuildFeatures(unsigned int base_p,
                                          const float* stage1, int n_stage1,
                                          float m1raw, float* x, float* t_out) {
  // e[0..24]: f16-quantized usable expert row (skips dead fxcm slot 23).
  float e[25];
  for (int j = 0; j < 23; ++j) {
    const float v = (j < n_stage1) ? stage1[j] : 0.0f;
    e[j] = MapExpert(F16RoundTrip(v));
  }
  e[23] = MapExpert(F16RoundTrip(n_stage1 > 24 ? stage1[24] : 0.0f));
  e[24] = MapExpert(F16RoundTrip(m1raw));

  const float p1 = (float)base_p / 65536.0f;  // clamp(1e-6,..) never binds
  const float t = std::log(p1 / (1.0f - p1));
  *t_out = t;

  for (int k = 0; k < kNin; ++k) x[k] = 0.0f;
  for (int j = 0; j < 25; ++j) x[j] = e[j];

  x[25 + (int)(i & 7)] = 1.0f;                       // bitpos one-hot

  const float tl = t / kTScale;                      // tfeat
  x[33] = tl;
  x[34] = tl * tl;
  x[35] = Sgn(t);

  // Trailing means. den = min(i, W) clamped to >= 1; empty window -> 0.
  for (int k = 0; k < 5; ++k) {
    const double den = (double)((i < (unsigned long long)kResW[k])
                                    ? (i ? i : 1) : (unsigned long long)kResW[k]);
    x[36 + k] = (float)(res_sum[k] / den) * 4.0f;
  }
  float surm[6];
  for (int k = 0; k < 6; ++k) {
    const double den = (double)((i < (unsigned long long)kSurW[k])
                                    ? (i ? i : 1) : (unsigned long long)kSurW[k]);
    surm[k] = (float)(sur_sum[k] / den);
    x[41 + k] = surm[k];
  }
  x[47] = (surm[1] - surm[3]) * 4.0f;                // regime
  x[48] = (surm[2] - surm[5]) * 4.0f;
  x[49] = std::log1p((float)(i - last_shock)) / 14.0f;  // shock clock
  {
    const double den = (double)((i < (unsigned long long)kShockW)
                                    ? (i ? i : 1) : (unsigned long long)kShockW);
    x[50] = (float)((double)shock_count / den) * 8.0f;  // qmean
  }
  const float sd = t - e[24];                        // ssed
  x[51] = sd / kTScale;
  {
    const double den = (double)((i < (unsigned long long)kSdCap)
                                    ? (i ? i : 1) : (unsigned long long)kSdCap);
    x[52] = (float)(sd_sum / den) / kTScale;
  }
  for (int k = 0; k < 7; ++k) {                      // erel means (window 512)
    const double den = (double)((i < (unsigned long long)kErelCap)
                                    ? (i ? i : 1) : (unsigned long long)kErelCap);
    x[53 + k] = (float)(erel_sum[k] / den);
  }

  // disp: mean/std(unbiased)/range/sign-agreement/quantiles of e.
  float mean = 0;
  for (int j = 0; j < 25; ++j) mean += e[j];
  mean /= 25.0f;
  float var = 0;
  for (int j = 0; j < 25; ++j) { const float d = e[j] - mean; var += d * d; }
  const float sdev = std::sqrt(var / 24.0f);
  float srt[25];
  for (int j = 0; j < 25; ++j) srt[j] = e[j];
  for (int a = 1; a < 25; ++a) {                      // insertion sort
    const float v = srt[a];
    int b = a - 1;
    while (b >= 0 && srt[b] > v) { srt[b + 1] = srt[b]; --b; }
    srt[b + 1] = v;
  }
  int agree = 0;
  const float st = Sgn(t);
  for (int j = 0; j < 25; ++j) agree += (Sgn(e[j]) == st);
  x[60] = mean;
  x[61] = sdev;
  x[62] = srt[24] - srt[0];
  x[63] = (float)agree / 25.0f;
  x[64] = srt[0]; x[65] = srt[6]; x[66] = srt[12]; x[67] = srt[18];
  x[68] = srt[24];

  // x[69..74] stay 0: xml features are zeroed in the all_no_xml feature set.

  // ctx8: previous 8 completed stream bytes / 255, oldest -> newest,
  // zero-padded at stream start.
  {
    const int have = (byte_count < 8) ? (int)byte_count : 8;
    const int pad = 8 - have;
    for (int q = 0; q < 8; ++q) {
      x[75 + q] = (q < pad)
          ? 0.0f
          : (float)hist8[(byte_count - 8 + q) & 7] / 255.0f;
    }
  }

  // zone one-hot + within-zone fraction; corpus progress + 10-cycle phase.
  const float bytepos = (float)((double)i / 8.0);
  int z = 4;
  for (int k = 0; k < 4; ++k) {
    if (bytepos <= kZB[k]) { z = k; break; }
  }
  x[83 + z] = 1.0f;
  x[88] = (bytepos - kZLo[z]) / (kZHi[z] - kZLo[z]);
  float progress = (float)((double)i) / kCorpusBits;
  if (progress > 1.0f) progress = 1.0f;
  x[89] = progress;
  const float phase = 62.831853071795864769f * progress;  // 2*pi*10
  x[90] = std::sin(phase);
  x[91] = std::cos(phase);

  // Pending values consumed by Observe() once the bit is known.
  pend_p1 = p1;
  pend_sd = sd;
  for (int k = 0; k < 7; ++k) {
    float pe = Sigmoidf(e[kTop[k]]);
    if (pe < 1e-6f) pe = 1e-6f;
    if (pe > 1.0f - 1e-6f) pe = 1.0f - 1e-6f;
    pend_pe[k] = pe;
  }
}

__attribute__((noinline))
float KhBitLstm32Head::Impl::RunNet(const float* x, float prev) {
  float x93[kInPad];
  for (int j = 0; j < kNin; ++j) x93[j] = x[j];
  x93[kNin] = prev;
  for (int j = kNin + 1; j < kInPad; ++j) x93[j] = 0.0f;

  float h0[kHid];
  for (int r = 0; r < kHid; ++r) {
    h0[r] = DotPadded(win + r * kInPad, x93, kInPad) + bin[r];
  }
  // Gate pre-activations, PyTorch order (i, f, g, o) x 32 rows.
  float pre[4 * kHid];
#if defined(__AVX2__) && defined(__FMA__)
  for (int r = 0; r < kHid; r += 8) {
    _mm256_storeu_ps(h0 + r, Silu8(_mm256_loadu_ps(h0 + r)));
  }
  for (int r = 0; r < 4 * kHid; ++r) {
    pre[r] = DotPadded(wih + r * kHid, h0, kHid) + bih[r] +
             DotPadded(whh + r * kHid, h, kHid) + bhh[r];
  }
  for (int r = 0; r < kHid; r += 8) {
    const __m256 gi = Sigmoid8(_mm256_loadu_ps(pre + 0 * kHid + r));
    const __m256 gf = Sigmoid8(_mm256_loadu_ps(pre + 1 * kHid + r));
    const __m256 gg = Tanh8(_mm256_loadu_ps(pre + 2 * kHid + r));
    const __m256 go = Sigmoid8(_mm256_loadu_ps(pre + 3 * kHid + r));
    const __m256 cn = _mm256_fmadd_ps(gf, _mm256_loadu_ps(c + r),
                                      _mm256_mul_ps(gi, gg));
    _mm256_storeu_ps(c + r, cn);
    _mm256_storeu_ps(h + r, _mm256_mul_ps(go, Tanh8(cn)));
  }
#else
  for (int r = 0; r < kHid; ++r) h0[r] = h0[r] * Sigmoidf(h0[r]);  // SiLU
  for (int r = 0; r < 4 * kHid; ++r) {
    pre[r] = DotPadded(wih + r * kHid, h0, kHid) + bih[r] +
             DotPadded(whh + r * kHid, h, kHid) + bhh[r];
  }
  for (int r = 0; r < kHid; ++r) {
    const float gi = Sigmoidf(pre[0 * kHid + r]);
    const float gf = Sigmoidf(pre[1 * kHid + r]);
    const float gg = std::tanh(pre[2 * kHid + r]);
    const float go = Sigmoidf(pre[3 * kHid + r]);
    const float cn = gf * c[r] + gi * gg;
    c[r] = cn;
    h[r] = go * std::tanh(cn);
  }
#endif
  return DotPadded(wout, h, kHid) + bout;
}

KhBitLstm32Head::KhBitLstm32Head(const char* blob_path) {
  impl_ = new Impl();
  if (!impl_->LoadBlob(blob_path)) {
    std::fprintf(stderr, "\n*** KH_BITLSTM32 WARNING: cannot load weight blob "
                 "'%s' -- head disabled ***\n", blob_path ? blob_path : "");
    delete impl_;
    impl_ = nullptr;
    ok_ = false;
    return;
  }
  ok_ = true;
}

KhBitLstm32Head::KhBitLstm32Head(const unsigned char* blob_data,
                                 size_t blob_size) {
  impl_ = new Impl();
  if (!impl_->LoadBlobMem(blob_data, blob_size)) {
    std::fprintf(stderr, "\n*** KH_BITLSTM32 WARNING: cannot load embedded "
                 "weight blob (%zu bytes) -- head disabled ***\n", blob_size);
    delete impl_;
    impl_ = nullptr;
    ok_ = false;
    return;
  }
  ok_ = true;
}

KhBitLstm32Head::KhBitLstm32Head(const KhBitLstm32Head& other)
    : ok_(other.ok_) {
  if (other.impl_ != nullptr) impl_ = new Impl(*other.impl_);
}

KhBitLstm32Head::~KhBitLstm32Head() { delete impl_; }

__attribute__((noinline))
unsigned int KhBitLstm32Head::Adjust(unsigned int base_p, const float* stage1,
                                     int n_stage1, float m1raw,
                                     int override_active) {
  Impl& s = *impl_;
  // seq64 TBPTT training reset: fresh recurrent state every 64 coded bits.
  if ((s.i % kSeqReset) == 0) {
    for (int r = 0; r < kHid; ++r) { s.h[r] = 0.0f; s.c[r] = 0.0f; }
  }
  float x[kNin];
  float t;
  s.BuildFeatures(base_p, stage1, n_stage1, m1raw, x, &t);
  const float delta = s.RunNet(x, s.prev_bit);
  if (override_active) {
    // Byte-mixer 0/1 override bits were loss-masked (w=0) during training:
    // the head's output there is unconstrained, so keep the coder's p.
    return base_p;
  }
  const float pnew = Sigmoidf(t + delta);
  unsigned int q = (unsigned int)(1.0f + 65534.0f * pnew);  // Discretize()
  if (q < 1) q = 1;
  if (q > 65535) q = 65535;
  return q;
}

__attribute__((noinline))
void KhBitLstm32Head::Observe(int bit) {
  Impl& s = *impl_;
  const unsigned long long i = s.i;
  const float y = (float)bit;

  // res family
  {
    const float v = y - s.pend_p1;
    for (int k = 0; k < 5; ++k) {
      if (i >= (unsigned long long)kResW[k]) {
        s.res_sum[k] -= (double)s.res_ring.buf[(i - kResW[k]) % kResCap];
      }
      s.res_sum[k] += (double)v;
    }
    s.res_ring.buf[i % kResCap] = v;
  }
  // sur family + shock window + shock clock
  {
    float pc = bit ? s.pend_p1 : (1.0f - s.pend_p1);
    if (pc < 1e-6f) pc = 1e-6f;
    const float v = -std::log2(pc);
    for (int k = 0; k < 6; ++k) {
      if (i >= (unsigned long long)kSurW[k]) {
        s.sur_sum[k] -= (double)s.sur_ring.buf[(i - kSurW[k]) % kSurCap];
      }
      s.sur_sum[k] += (double)v;
    }
    if (i >= (unsigned long long)kShockW) {
      s.shock_count -= (s.sur_ring.buf[(i - kShockW) % kSurCap] > 4.0f);
    }
    s.shock_count += (v > 4.0f);
    if (v > 4.0f) s.last_shock = i;
    s.sur_ring.buf[i % kSurCap] = v;
  }
  // |sd| window
  {
    const float v = std::fabs(s.pend_sd);
    if (i >= (unsigned long long)kSdCap) {
      s.sd_sum -= (double)s.sd_ring.buf[(i - kSdCap) % kSdCap];
    }
    s.sd_sum += (double)v;
    s.sd_ring.buf[i % kSdCap] = v;
  }
  // per-expert surprisal windows
  for (int k = 0; k < 7; ++k) {
    const float pe = bit ? s.pend_pe[k] : (1.0f - s.pend_pe[k]);
    const float v = -std::log2(pe);
    if (i >= (unsigned long long)kErelCap) {
      s.erel_sum[k] -= (double)s.erel_ring.buf[((i - kErelCap) % kErelCap) * 7 + k];
    }
    s.erel_sum[k] += (double)v;
    s.erel_ring.buf[(i % kErelCap) * 7 + k] = v;
  }
  // previous-bit input and byte history
  s.prev_bit = y;
  s.cur_byte = ((s.cur_byte << 1) | (unsigned)bit) & 0xFFu;
  if ((i & 7) == 7) {
    s.hist8[s.byte_count & 7] = (uint8_t)s.cur_byte;
    ++s.byte_count;
    s.cur_byte = 0;
  }
  s.i = i + 1;
}

#endif  // KH_BITLSTM32
