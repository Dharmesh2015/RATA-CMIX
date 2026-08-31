// KH_OBIAS upstream implementation. Mirrors, op for op, the torch modules
// PpmdFeatureNet + ObiasUpstream(mode="up") in
// tools/train_cmix_ppmd16_stack_torch.py (pytorch-cmix-lstm worktree):
//
//   e_s   = embedding[lag_byte(s)] (zeroed while position < lag)
//   b     = lag_w @ concat(e_1,e_2,e_4,e_8) + lag_b + pos_w @ pos9 + pos_b
//   xp    = clamp(ln(max(p, 1e-12)) - mean_j ln(max(p_j,1e-12)), +-12)
//   q     = ppmd_w @ xp + ppmd_b                       (low-rank 256->32)
//   x     = concat(b[7], q[32])                        (LSTM input, 39)
//   i,f,g,o = torch LSTM gate order; h,c update; z = tanh(bottle @ h + bb)
//   gate  = g0 + gate_w . z + gate_b
//   bias_i = gate * ln(max(p_i, 1e-6)) + sym_w[i] . z
//
// Position features: identical to DirectLstm16Feature / the trainer's
// position_features (zone one-hot 5 + zone progress + global progress +
// sin/cos of 20*pi*progress), evaluated at the position being predicted
// (= bytes completed so far).
//
// This TU must be compiled with -ffp-model=precise (see makefile) and the
// public entry points are noinline: encode and decode then run identical
// machine code, deterministic by construction.

#ifdef KH_OBIAS

#include "obias-prior.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint64_t kCorpusBytes = 587138826ULL;
constexpr int kEmb = 7, kProj = 32, kHid = 32, kBot = 16;
constexpr int kLstmIn = kEmb + kProj;  // 39

struct BlobHeader {
  char magic[8];        // "OBIASUP\x01"
  uint32_t emb, proj, hid, bottleneck;
  uint64_t corpus_bytes;
  uint32_t half_count;
  uint32_t reserved[7];
};  // 64 bytes

uint32_t HalfCountExpected() {
  return 256 * kEmb + kEmb * (4 * kEmb) + kEmb + kEmb * 9 + kEmb +
         kProj * 256 + kProj + (4 * kHid) * kLstmIn + (4 * kHid) * kHid +
         4 * kHid + 4 * kHid + kBot * kHid + kBot + kBot + 1 + 256 * kBot +
         1;
}

// f16 -> f32 widening, identical to the trainer's numpy astype(float32)
// (IEEE 754 round-trip; same routine family as bitlstm32-head).
float F32FromF16(uint16_t h) {
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

inline float Sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

}  // namespace

struct KhObiasPrior::Impl {
  // obias_sched mode (magic "OBIASSC\x01"): 3-param scheduled scalar gate,
  //   gate(t) = a + b*exp(-(t/corpus_bytes)/exp(log_tau))
  //   bias_i  = gate(t)*ln(max(p_i,1e-6))
  // t = bytes completed after Advance (same position the obias_up position
  // features use). float32 math. Sidecar: obias_handoff/obias_sched.blob.json.
  bool sched = false;
  float sched_a = 0.0f, sched_b = 0.0f, sched_tau = 1.0f;

  std::vector<float> w;
  const float *embedding, *lag_w, *lag_b, *pos_w, *pos_b;
  const float *ppmd_w, *ppmd_b, *w_ih, *w_hh, *b_ih, *b_hh;
  const float *bottle_w, *bottle_b, *gate_w, *gate_b, *sym_w, *g0;
  unsigned char history[8] = {};
  float h[kHid] = {};
  float c[kHid] = {};
  float bias[256] = {};
  float z16[kBot] = {};   // parity hook
  float gate_v = 0.0f;    // parity hook
  uint64_t completed = 0;

  bool Load(const char* path) {
    if (!path || !path[0]) return false;
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    BlobHeader hd;
    bool ok = fread(&hd, 1, sizeof(hd), f) == sizeof(hd);
    if (ok && memcmp(hd.magic, "OBIASSC\x01", 8) == 0) {
      // Scheduled-gate blob: header dims zero, half_count slot = f32 count.
      ok = hd.emb == 0 && hd.proj == 0 && hd.hid == 0 && hd.bottleneck == 0 &&
           hd.corpus_bytes == kCorpusBytes && hd.half_count == 3;
      float prm[3];
      ok = ok && fread(prm, sizeof(float), 3, f) == 3;
      const int extra_sc = fgetc(f);
      fclose(f);
      if (!ok || extra_sc != EOF) return false;
      sched = true;
      sched_a = prm[0];
      sched_b = prm[1];
      sched_tau = expf(prm[2]);
      return true;
    }
    ok = ok && memcmp(hd.magic, "OBIASUP\x01", 8) == 0;
    ok = ok && hd.emb == (uint32_t)kEmb && hd.proj == (uint32_t)kProj &&
         hd.hid == (uint32_t)kHid && hd.bottleneck == (uint32_t)kBot;
    ok = ok && hd.corpus_bytes == kCorpusBytes;
    ok = ok && hd.half_count == HalfCountExpected();
    if (!ok) { fclose(f); return false; }
    std::vector<uint16_t> halves(hd.half_count);
    ok = fread(halves.data(), sizeof(uint16_t), halves.size(), f) ==
         halves.size();
    const int extra = fgetc(f);
    fclose(f);
    if (!ok || extra != EOF) return false;
    w.resize(halves.size());
    for (size_t i = 0; i < halves.size(); ++i) w[i] = F32FromF16(halves[i]);
    size_t at = 0;
    auto take = [&](size_t n) { const float* p = w.data() + at; at += n;
                                return p; };
    embedding = take(256 * kEmb);
    lag_w = take((size_t)kEmb * 4 * kEmb);
    lag_b = take(kEmb);
    pos_w = take((size_t)kEmb * 9);
    pos_b = take(kEmb);
    ppmd_w = take((size_t)kProj * 256);
    ppmd_b = take(kProj);
    w_ih = take((size_t)4 * kHid * kLstmIn);
    w_hh = take((size_t)4 * kHid * kHid);
    b_ih = take(4 * kHid);
    b_hh = take(4 * kHid);
    bottle_w = take((size_t)kBot * kHid);
    bottle_b = take(kBot);
    gate_w = take(kBot);
    gate_b = take(1);
    sym_w = take((size_t)256 * kBot);
    g0 = take(1);
    if (at != w.size()) return false;
    return true;
  }

  void PositionFeatures(uint64_t position, float* f) const {
    const float p = (float)position;
    const float bounds[4] = {541126651.0f, 554726452.0f, 571499539.0f,
                             586459321.0f};
    const float lo[5] = {0.0f, bounds[0], bounds[1], bounds[2], bounds[3]};
    const float hi[5] = {bounds[0], bounds[1], bounds[2], bounds[3],
                         (float)kCorpusBytes};
    // std::lower_bound semantics (== torch.bucketize right=False), matching
    // the deployed, parity-gated DirectLstm16Feature.
    int zone = 0;
    while (zone < 4 && p > bounds[zone]) ++zone;
    for (int i = 0; i < 5; ++i) f[i] = i == zone ? 1.0f : 0.0f;
    f[5] = (p - lo[zone]) / (hi[zone] - lo[zone]);
    f[6] = p / (float)kCorpusBytes;
    const float phase = (float)(20.0 * 3.14159265358979323846) * f[6];
    f[7] = sinf(phase);
    f[8] = cosf(phase);
  }

  void Step(const float* p256) {
    if (sched) {
      const float progress = (float)completed / (float)kCorpusBytes;
      const float gate = sched_a + sched_b * expf(-progress / sched_tau);
      gate_v = gate;
      for (int i = 0; i < 256; ++i) {
        float pv = p256[i];
        if (pv < 1e-6f) pv = 1e-6f;
        bias[i] = gate * logf(pv);
      }
      return;
    }
    // Lag/position stem (position = completed, the byte being predicted).
    float lagcat[4 * kEmb] = {};
    const unsigned int lags[4] = {1, 2, 4, 8};
    for (int s = 0; s < 4; ++s) {
      if (completed < lags[s]) continue;
      const unsigned char byte = history[(completed - lags[s]) & 7u];
      const float* e = embedding + (size_t)byte * kEmb;
      for (int j = 0; j < kEmb; ++j) lagcat[s * kEmb + j] = e[j];
    }
    float pos[9];
    PositionFeatures(completed, pos);
    float x[kLstmIn];
    for (int i = 0; i < kEmb; ++i) {
      float a = lag_b[i], b = pos_b[i];
      for (int j = 0; j < 4 * kEmb; ++j)
        a += lag_w[i * 4 * kEmb + j] * lagcat[j];
      for (int j = 0; j < 9; ++j) b += pos_w[i * 9 + j] * pos[j];
      x[i] = a + b;
    }
    // Low-rank projection of the clamped centered log distribution.
    float xp[256];
    float mean = 0.0f;
    for (int j = 0; j < 256; ++j) {
      float pv = p256[j];
      if (pv < 1e-12f) pv = 1e-12f;
      xp[j] = logf(pv);
      mean += xp[j];
    }
    mean /= 256.0f;
    for (int j = 0; j < 256; ++j) {
      float v = xp[j] - mean;
      if (v > 12.0f) v = 12.0f;
      if (v < -12.0f) v = -12.0f;
      xp[j] = v;
    }
    for (int i = 0; i < kProj; ++i) {
      float a = ppmd_b[i];
      const float* row = ppmd_w + (size_t)i * 256;
      for (int j = 0; j < 256; ++j) a += row[j] * xp[j];
      x[kEmb + i] = a;
    }
    // LSTM cell, torch gate order (i, f, g, o).
    float gates[4 * kHid];
    for (int i = 0; i < 4 * kHid; ++i) {
      float a = b_ih[i] + b_hh[i];
      const float* ri = w_ih + (size_t)i * kLstmIn;
      const float* rh = w_hh + (size_t)i * kHid;
      for (int j = 0; j < kLstmIn; ++j) a += ri[j] * x[j];
      for (int j = 0; j < kHid; ++j) a += rh[j] * h[j];
      gates[i] = a;
    }
    float nh[kHid];
    for (int j = 0; j < kHid; ++j) {
      const float ig = Sigmoid(gates[j]);
      const float fg = Sigmoid(gates[kHid + j]);
      const float gg = tanhf(gates[2 * kHid + j]);
      const float og = Sigmoid(gates[3 * kHid + j]);
      c[j] = fg * c[j] + ig * gg;
      nh[j] = og * tanhf(c[j]);
    }
    std::memcpy(h, nh, sizeof(nh));
    float* z = z16;
    for (int i = 0; i < kBot; ++i) {
      float a = bottle_b[i];
      const float* row = bottle_w + (size_t)i * kHid;
      for (int j = 0; j < kHid; ++j) a += row[j] * h[j];
      z[i] = tanhf(a);
    }
    float gate = g0[0] + gate_b[0];
    for (int j = 0; j < kBot; ++j) gate += gate_w[j] * z[j];
    gate_v = gate;
    for (int i = 0; i < 256; ++i) {
      float pv = p256[i];
      if (pv < 1e-6f) pv = 1e-6f;
      float a = gate * logf(pv);
      const float* row = sym_w + (size_t)i * kBot;
      for (int j = 0; j < kBot; ++j) a += row[j] * z[j];
      bias[i] = a;
    }
  }
};

__attribute__((noinline)) KhObiasPrior::KhObiasPrior(const char* blob_path) {
  impl_ = new Impl();
#ifdef KH_OBIAS_CONST_GATE
  if (blob_path == nullptr) {
    impl_->sched = true;
    impl_->sched_a = (float)(KH_OBIAS_CONST_GATE);
    impl_->sched_b = 0.0f;
    impl_->sched_tau = 1.0f;
    ok_ = true;
    std::fprintf(stderr, "kh-obias: compiled constant gate g=%.6f\n",
                 impl_->sched_a);
    return;
  }
#endif
  if (!impl_->Load(blob_path)) {
    std::fprintf(stderr, "\n*** KH_OBIAS WARNING: cannot load weight blob "
                 "'%s' -- prior disabled ***\n",
                 blob_path ? blob_path : "");
    return;
  }
  ok_ = true;
  if (impl_->sched) {
    std::fprintf(stderr, "kh-obias: loaded %s (sched a=%.6f b=%.6f tau=%.8f)\n",
                 blob_path, impl_->sched_a, impl_->sched_b, impl_->sched_tau);
  } else {
    std::fprintf(stderr, "kh-obias: loaded %s (%zu fp32 params)\n", blob_path,
                 impl_->w.size());
  }
}

KhObiasPrior::~KhObiasPrior() { delete impl_; }

__attribute__((noinline)) void KhObiasPrior::Advance(unsigned char byte,
                                                     const float* p256) {
  if (!ok_) return;
  Impl& im = *impl_;
  im.history[im.completed & 7u] = byte;
  ++im.completed;
  im.Step(p256);
}

__attribute__((noinline)) const float* KhObiasPrior::Bias() const {
  return impl_->bias;
}

uint64_t KhObiasPrior::BytesCompleted() const { return impl_->completed; }

const float* KhObiasPrior::Z16() const { return impl_->z16; }

float KhObiasPrior::Gate() const { return impl_->gate_v; }

#endif  // KH_OBIAS
