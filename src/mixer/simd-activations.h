#ifndef SIMD_ACTIVATIONS_H
#define SIMD_ACTIVATIONS_H

// Batched branch-free exp / logistic-sigmoid / tanh over contiguous float
// arrays for the LSTM hot path. Replaces per-cell scalar libm calls whose
// range-reduction branches dominate branch mispredicts.
//
// ISA selection is compile-time from the build's -march flags:
//   __AVX512F__ -> 16-wide AVX-512 (dev box, -march=native on Zen 4/5)
//   __AVX2__+__FMA__ -> 8-wide AVX2 (official AMD test machine, make ZEN2=1)
//   otherwise -> scalar polynomial (still branch-free, no libm)
// so no build of this header can execute an instruction its -march does not
// guarantee. All paths evaluate the same per-element operation sequence
// (Cephes-style expf: round-to-nearest-even range reduction, degree-5
// polynomial, FMA where the target has it).

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace simd_act {

// Clamp bounds keep the 2^n exponent scale in [-126, 127] so the bit-trick
// scale never overflows into the exponent-255 range.
constexpr float kExpLo = -87.3365447505531f;
constexpr float kExpHi = 88.3762626647950f;
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kLn2Hi = 0.693359375f;
constexpr float kLn2Lo = -2.12194440e-4f;
constexpr float kExpC0 = 1.9875691500e-4f;
constexpr float kExpC1 = 1.3981999507e-3f;
constexpr float kExpC2 = 8.3334519073e-3f;
constexpr float kExpC3 = 4.1665795894e-2f;
constexpr float kExpC4 = 1.6666665459e-1f;
constexpr float kExpC5 = 5.0000001201e-1f;

#if defined(__AVX512F__) && !defined(SIMD_ACT_FORCE_AVX2)

static inline __m512 VExp(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(kExpLo)),
      _mm512_set1_ps(kExpHi));
  __m512 fn = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(kLog2e)),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(fn, _mm512_set1_ps(kLn2Hi), x);
  r = _mm512_fnmadd_ps(fn, _mm512_set1_ps(kLn2Lo), r);
  __m512 y = _mm512_set1_ps(kExpC0);
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(kExpC1));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(kExpC2));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(kExpC3));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(kExpC4));
  y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(kExpC5));
  y = _mm512_fmadd_ps(y, _mm512_mul_ps(r, r), r);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));
  __m512i n = _mm512_cvtps_epi32(fn);
  n = _mm512_slli_epi32(_mm512_add_epi32(n, _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(n));
}

static inline __m512 VLogistic(__m512 x) {
  __m512 e = VExp(_mm512_sub_ps(_mm512_setzero_ps(), x));
  return _mm512_div_ps(_mm512_set1_ps(1.0f),
      _mm512_add_ps(_mm512_set1_ps(1.0f), e));
}

static inline __m512 VTanh(__m512 x) {
  const __m512i sign = _mm512_set1_epi32(0x80000000);
  __m512i xi = _mm512_castps_si512(x);
  __m512 a = _mm512_castsi512_ps(_mm512_andnot_si512(sign, xi));
  __m512 e = VExp(_mm512_mul_ps(a, _mm512_set1_ps(-2.0f)));
  __m512 t = _mm512_div_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), e),
      _mm512_add_ps(_mm512_set1_ps(1.0f), e));
  __m512i ti = _mm512_castps_si512(t);
  ti = _mm512_or_si512(ti, _mm512_and_si512(sign, xi));
  return _mm512_castsi512_ps(ti);
}

inline void Exp(float* p, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(p + i, VExp(_mm512_loadu_ps(p + i)));
  }
  if (i < n) {
    __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
    _mm512_mask_storeu_ps(p + i, m, VExp(_mm512_maskz_loadu_ps(m, p + i)));
  }
}

inline void Logistic(float* p, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(p + i, VLogistic(_mm512_loadu_ps(p + i)));
  }
  if (i < n) {
    __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
    _mm512_mask_storeu_ps(p + i, m, VLogistic(_mm512_maskz_loadu_ps(m, p + i)));
  }
}

inline void Tanh(float* dst, const float* src, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i, VTanh(_mm512_loadu_ps(src + i)));
  }
  if (i < n) {
    __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
    _mm512_mask_storeu_ps(dst + i, m, VTanh(_mm512_maskz_loadu_ps(m, src + i)));
  }
}

#elif defined(__AVX2__) && defined(__FMA__)

alignas(32) static const int32_t kTailMask[16] = {
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

static inline __m256i TailMask(size_t rem) {
  return _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(kTailMask + 8 - rem));
}

static inline __m256 VExp(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(kExpLo)),
      _mm256_set1_ps(kExpHi));
  __m256 fn = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(kLn2Hi), x);
  r = _mm256_fnmadd_ps(fn, _mm256_set1_ps(kLn2Lo), r);
  __m256 y = _mm256_set1_ps(kExpC0);
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(kExpC1));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(kExpC2));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(kExpC3));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(kExpC4));
  y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(kExpC5));
  y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
  __m256i n = _mm256_cvtps_epi32(fn);
  n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

static inline __m256 VLogistic(__m256 x) {
  __m256 e = VExp(_mm256_sub_ps(_mm256_setzero_ps(), x));
  return _mm256_div_ps(_mm256_set1_ps(1.0f),
      _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline __m256 VTanh(__m256 x) {
  const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
  __m256 a = _mm256_andnot_ps(sign, x);
  __m256 e = VExp(_mm256_mul_ps(a, _mm256_set1_ps(-2.0f)));
  __m256 t = _mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), e),
      _mm256_add_ps(_mm256_set1_ps(1.0f), e));
  return _mm256_or_ps(t, _mm256_and_ps(sign, x));
}

inline void Exp(float* p, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(p + i, VExp(_mm256_loadu_ps(p + i)));
  }
  if (i < n) {
    __m256i m = TailMask(n - i);
    _mm256_maskstore_ps(p + i, m, VExp(_mm256_maskload_ps(p + i, m)));
  }
}

inline void Logistic(float* p, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(p + i, VLogistic(_mm256_loadu_ps(p + i)));
  }
  if (i < n) {
    __m256i m = TailMask(n - i);
    _mm256_maskstore_ps(p + i, m, VLogistic(_mm256_maskload_ps(p + i, m)));
  }
}

inline void Tanh(float* dst, const float* src, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, VTanh(_mm256_loadu_ps(src + i)));
  }
  if (i < n) {
    __m256i m = TailMask(n - i);
    _mm256_maskstore_ps(dst + i, m, VTanh(_mm256_maskload_ps(src + i, m)));
  }
}

#else  // scalar fallback: same polynomial, branch-free, autovectorizable

#if defined(__FMA__)
static inline float SFma(float a, float b, float c) {
  return __builtin_fmaf(a, b, c);
}
#else
static inline float SFma(float a, float b, float c) { return a * b + c; }
#endif

static inline float VExp(float x) {
  x = x < kExpLo ? kExpLo : x;
  x = x > kExpHi ? kExpHi : x;
  float fn = __builtin_rintf(x * kLog2e);
  float r = SFma(fn, -kLn2Hi, x);
  r = SFma(fn, -kLn2Lo, r);
  float y = kExpC0;
  y = SFma(y, r, kExpC1);
  y = SFma(y, r, kExpC2);
  y = SFma(y, r, kExpC3);
  y = SFma(y, r, kExpC4);
  y = SFma(y, r, kExpC5);
  y = SFma(y, r * r, r);
  y += 1.0f;
  int32_t bits = ((int32_t)fn + 127) << 23;
  float scale;
  memcpy(&scale, &bits, sizeof(scale));
  return y * scale;
}

static inline float VLogistic(float x) {
  return 1.0f / (1.0f + VExp(-x));
}

static inline float VTanh(float x) {
  float a = __builtin_fabsf(x);
  float e = VExp(-2.0f * a);
  float t = (1.0f - e) / (1.0f + e);
  return x < 0.0f ? -t : t;
}

inline void Exp(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = VExp(p[i]);
}

inline void Logistic(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) p[i] = VLogistic(p[i]);
}

inline void Tanh(float* dst, const float* src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = VTanh(src[i]);
}

#endif

inline void Tanh(float* p, size_t n) { Tanh(p, p, n); }

// ---------------------------------------------------------------------------
// fp16 weight-shadow helpers (step_002_003). The LSTM stores its read-heavy
// weight streams twice: the fp32 master (which Adam / the SGD output-layer
// update evolves, so quantization error never compounds) and an fp16 shadow
// re-encoded whenever the master changes. The per-byte forward matvecs read
// only the shadow, halving the streamed bytes; every dot product accumulates
// in fp32. Hardware conversion: _mm512_cvtph_ps (AVX-512F) or
// _mm256_cvtph_ps (F16C) - both official Hutter machines (i7-1165G7 and
// Zen 2/3 Ryzen 7) have F16C+AVX2+FMA. Without F16C, SIMD_ACT_F16 stays 0
// and the callers keep their fp32 paths.
// All row lengths passed in are padded strides (multiples of 16 floats)
// whose padding lanes are zero in both the master and the shadow.

// SIMD_ACT_DISABLE_F16 forces the fp32 path even where F16C exists.
//
// The fp16 shadow weights are the one part of the LSTM rework that is
// NOT bit-exact by construction: a dot product taken at half precision
// cannot reproduce the fp32 result, so the archive moves. Whether it
// moves by 2 bytes or 200 is a question about this stream, and without
// a switch there is no way to ask it -- the path turns itself on
// wherever the ISA allows, which is every build that matters.
//
// Off is not the default. The measured deltas are small and in the
// favourable direction, and the speedup is large. This exists so the
// trade can be quantified rather than assumed.
#if defined(SIMD_ACT_DISABLE_F16)
#define SIMD_ACT_F16 0
#elif (defined(__AVX512F__) && !defined(SIMD_ACT_FORCE_AVX2)) || \
    (defined(__AVX2__) && defined(__FMA__) && defined(__F16C__))
#define SIMD_ACT_F16 1
#else
#define SIMD_ACT_F16 0
#endif

#if SIMD_ACT_F16
#if defined(__AVX512F__) && !defined(SIMD_ACT_FORCE_AVX2)

static inline __m512 F16Load16(const uint16_t* p) {
  return _mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

static inline void F16Store16(uint16_t* p, __m512 v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
      _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

inline void F16EncodeRow(const float* src, uint16_t* dst, size_t n) {
  for (size_t j = 0; j < n; j += 16) {
    F16Store16(dst + j, _mm512_loadu_ps(src + j));
  }
}

// Widen an fp16 row into an fp32 row (step_003_006: fp16-only per-epoch
// LSTM histories are decoded once per BPTT epoch into fp32 scratch).
inline void F16DecodeRow(const uint16_t* src, float* dst, size_t n) {
  for (size_t j = 0; j < n; j += 16) {
    _mm512_storeu_ps(dst + j, F16Load16(src + j));
  }
}

// d{0,1,2} = dot(x, r{0,1,2}), fp16 rows, fp32 accumulation.
inline void F16Dot3(const float* x, const uint16_t* r0, const uint16_t* r1,
    const uint16_t* r2, size_t n, float* d0, float* d1, float* d2) {
  __m512 a0 = _mm512_setzero_ps();
  __m512 a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps();
  for (size_t j = 0; j < n; j += 16) {
    const __m512 v = _mm512_loadu_ps(x + j);
    a0 = _mm512_fmadd_ps(v, F16Load16(r0 + j), a0);
    a1 = _mm512_fmadd_ps(v, F16Load16(r1 + j), a1);
    a2 = _mm512_fmadd_ps(v, F16Load16(r2 + j), a2);
  }
  *d0 = _mm512_reduce_add_ps(a0);
  *d1 = _mm512_reduce_add_ps(a1);
  *d2 = _mm512_reduce_add_ps(a2);
}

inline float F16Dot(const float* x, const uint16_t* r, size_t n) {
  __m512 a = _mm512_setzero_ps();
  for (size_t j = 0; j < n; j += 16) {
    a = _mm512_fmadd_ps(_mm512_loadu_ps(x + j), F16Load16(r + j), a);
  }
  return _mm512_reduce_add_ps(a);
}

// dst[j] = src[j] - s*h[j], emitting the fp16 image of dst in the same pass.
inline void F16UpdateEncodeRow(float* dst, uint16_t* dsth, const float* src,
    float s, const float* h, size_t n) {
  const __m512 vs = _mm512_set1_ps(s);
  for (size_t j = 0; j < n; j += 16) {
    const __m512 v = _mm512_fnmadd_ps(vs, _mm512_loadu_ps(h + j),
        _mm512_loadu_ps(src + j));
    _mm512_storeu_ps(dst + j, v);
    F16Store16(dsth + j, v);
  }
}

// acc[j] += e * r[j], fp16 row.
inline void F16Axpy(float* acc, const uint16_t* r, float e, size_t n) {
  const __m512 ve = _mm512_set1_ps(e);
  for (size_t j = 0; j < n; j += 16) {
    _mm512_storeu_ps(acc + j,
        _mm512_fmadd_ps(ve, F16Load16(r + j), _mm512_loadu_ps(acc + j)));
  }
}

#else  // AVX2 + FMA + F16C

static inline __m256 F16Load8(const uint16_t* p) {
  return _mm256_cvtph_ps(
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

static inline void F16Store8(uint16_t* p, __m256 v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p),
      _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline float F16Reduce8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
      _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

inline void F16EncodeRow(const float* src, uint16_t* dst, size_t n) {
  for (size_t j = 0; j < n; j += 8) {
    F16Store8(dst + j, _mm256_loadu_ps(src + j));
  }
}

inline void F16DecodeRow(const uint16_t* src, float* dst, size_t n) {
  for (size_t j = 0; j < n; j += 8) {
    _mm256_storeu_ps(dst + j, F16Load8(src + j));
  }
}

inline void F16Dot3(const float* x, const uint16_t* r0, const uint16_t* r1,
    const uint16_t* r2, size_t n, float* d0, float* d1, float* d2) {
  __m256 a0 = _mm256_setzero_ps();
  __m256 a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps();
  for (size_t j = 0; j < n; j += 8) {
    const __m256 v = _mm256_loadu_ps(x + j);
    a0 = _mm256_fmadd_ps(v, F16Load8(r0 + j), a0);
    a1 = _mm256_fmadd_ps(v, F16Load8(r1 + j), a1);
    a2 = _mm256_fmadd_ps(v, F16Load8(r2 + j), a2);
  }
  *d0 = F16Reduce8(a0);
  *d1 = F16Reduce8(a1);
  *d2 = F16Reduce8(a2);
}

inline float F16Dot(const float* x, const uint16_t* r, size_t n) {
  __m256 a = _mm256_setzero_ps();
  for (size_t j = 0; j < n; j += 8) {
    a = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), F16Load8(r + j), a);
  }
  return F16Reduce8(a);
}

inline void F16UpdateEncodeRow(float* dst, uint16_t* dsth, const float* src,
    float s, const float* h, size_t n) {
  const __m256 vs = _mm256_set1_ps(s);
  for (size_t j = 0; j < n; j += 8) {
    const __m256 v = _mm256_fnmadd_ps(vs, _mm256_loadu_ps(h + j),
        _mm256_loadu_ps(src + j));
    _mm256_storeu_ps(dst + j, v);
    F16Store8(dsth + j, v);
  }
}

inline void F16Axpy(float* acc, const uint16_t* r, float e, size_t n) {
  const __m256 ve = _mm256_set1_ps(e);
  for (size_t j = 0; j < n; j += 8) {
    _mm256_storeu_ps(acc + j,
        _mm256_fmadd_ps(ve, F16Load8(r + j), _mm256_loadu_ps(acc + j)));
  }
}

#endif
#endif  // SIMD_ACT_F16

}  // namespace simd_act

#endif  // SIMD_ACTIVATIONS_H
