#include "lstm-layer.h"

#include "sigmoid.h"
#include <math.h>
#include <algorithm>
#include <numeric>

#define FAST_TANH tanh //fast_tanh
// Fuse the forget/input/output gate matvecs into one pass over the shared
// input vector. Each gate keeps its own single accumulator walking the row
// in the same element order as before, so per-gate results are unchanged.
#define LSTM_FUSED_GATE_MATVEC 1

namespace {

// alpha / bias-correction terms depend only on the step counter, so they are
// computed once per burst instead of once per weight row. Same expressions
// (and therefore same float values) as the old per-call code.
struct AdamParams { float alpha, bc1, bc2; };

inline AdamParams MakeAdamParams(float learning_rate, float t) {
  const float beta1 = 0.025, beta2 = 0.9999;
  AdamParams p;
  if (t < UPDATE_LIMIT) {
    p.alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    p.bc1 = (float)(1.0f - pow(beta1, t));
    p.bc2 = (float)(1.0f - pow(beta2, t));
  } else {
    p.alpha = learning_rate * 0.1f / sqrt(5e-5f * UPDATE_LIMIT + 1.0f);
    p.bc1 = (float)(1.0f - pow(beta1, UPDATE_LIMIT));
    p.bc2 = (float)(1.0f - pow(beta2, UPDATE_LIMIT));
  }
  return p;
}

// The old five elementwise loops fused into one pass: every element was
// independent across loops, and the per-element operation sequence below is
// exactly the old one, so this only changes memory traffic (each of g/m/v/w
// is now streamed once instead of m 4x / v 4x / g 3x / w 2x).
inline void Adam(const AdamParams& p, float* __restrict g, float* __restrict m,
    float* __restrict v, float* __restrict w, size_t n) {
  const float beta1 = 0.025, beta2 = 0.9999, eps = 1e-6f;
  for (size_t k = 0; k < n; ++k) {
    const float gk = g[k];
    float mk = m[k] * beta1;
    mk += (1.0f - beta1) * gk;
    float vk = v[k] * beta2;
    vk += (1.0f - beta2) * gk * gk;
    m[k] = mk;
    v[k] = vk;
    w[k] -= p.alpha * ((mk / p.bc1) / (std::sqrt(vk / p.bc2 + eps)));
  }
}

}

inline LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    state_(LstmAllocFloats(LstmPad16(num_cells))),
    state_error_(num_cells), stored_error_(num_cells),
    input_ptrs_(horizon),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    dense_width_(input_size - output_size),
    scell_(LstmPad16(num_cells)),
    sdense_(LstmPad16(input_size - output_size)),
    tanh_state_(LstmAllocHist((size_t)horizon * LstmPad16(num_cells))),
    input_gate_state_(LstmAllocHist((size_t)horizon * LstmPad16(num_cells))),
    last_state_(LstmAllocHist((size_t)horizon * LstmPad16(num_cells))),
    inp_hist_(LstmAllocHist(SIMD_ACT_F16
        ? (size_t)horizon * LstmPad16(input_size - output_size) : 0)),
    scratch_(LstmAllocFloats(SIMD_ACT_F16
        ? (size_t)8 * LstmPad16(num_cells) : 0)),
    forget_gate_(input_size, output_size, num_cells, horizon,
        output_size + auxiliary_input_size),
    input_node_(input_size, output_size, num_cells, horizon,
        output_size + auxiliary_input_size),
    output_gate_(input_size, output_size, num_cells, horizon,
        output_size + auxiliary_input_size) {
  // combo_llif12 (env KH_LSTM_LLIF12): Xavier bound x init_scale. Disabled the
  // scale is 1.0f, whose multiply is IEEE-exact -> identical golden init.
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  // Same Rand() call order as before: (forget, input, output) per (i, j),
  // j spanning the full logical row (symbol block then dense block).
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < input_size; ++j) {
      float wf = low + Rand() * range;
      float wi = low + Rand() * range;
      float wo = low + Rand() * range;
      if (j < output_size_) {
        forget_gate_.wsymt(j)[i] = wf;
        input_node_.wsymt(j)[i] = wi;
        output_gate_.wsymt(j)[i] = wo;
      } else {
        forget_gate_.wdense(i)[j - output_size_] = wf;
        input_node_.wdense(i)[j - output_size_] = wi;
        output_gate_.wdense(i)[j - output_size_] = wo;
      }
    }
    // combo_llif12: forget-gate dense bias init (1.0 golden, 0.0 with
    // KH_LSTM_LLIF12).
    forget_gate_.wdense(i)[dense_width_ - 1] = 1.0f;
  }
#if SIMD_ACT_F16
  {
    const size_t sdense = forget_gate_.sdense_;
    wdense3h_ = LstmAllocHalf((size_t)num_cells_ * 3 * sdense);
    forget_gate_.whd_ = wdense3h_.get();
    input_node_.whd_ = wdense3h_.get() + sdense;
    output_gate_.whd_ = wdense3h_.get() + 2 * sdense;
    forget_gate_.whd_stride_ = 3 * sdense;
    input_node_.whd_stride_ = 3 * sdense;
    output_gate_.whd_stride_ = 3 * sdense;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      simd_act::F16EncodeRow(forget_gate_.wdense(i), forget_gate_.whdense(i),
          sdense);
      simd_act::F16EncodeRow(input_node_.wdense(i), input_node_.whdense(i),
          sdense);
      simd_act::F16EncodeRow(output_gate_.wdense(i), output_gate_.whdense(i),
          sdense);
    }
  }
#endif
}

inline void LstmLayer::LayerNorm(NeuronLayer& neurons, float* nrm, float* st) {
  float f = 0;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    f += nrm[i] * nrm[i];
  }
  neurons.ivar_[epoch_] = 1.0f / sqrt((f / num_cells_) + 1e-5f);
  const float ivar = neurons.ivar_[epoch_];
  for (unsigned int i = 0; i < num_cells_; ++i) {
    nrm[i] *= ivar;
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    st[i] = nrm[i] * neurons.gamma_[i] + neurons.beta_[i];
  }
}

inline void LstmLayer::ForwardPass(const float* input, int input_symbol,
    float* hidden, int hidden_start) {
  float* cs = state_.get();
#if SIMD_ACT_F16
  // step_003_006: this epoch's input row and pre-update cell state go
  // straight into their fp16 histories; the gate values computed below live
  // in fp32 scratch rows (all padding lanes zero) that the forward math
  // consumes at full precision, and are quantized into the fp16 histories
  // once finalized. BackwardPass is the histories' only reader.
  simd_act::F16EncodeRow(input, inp_hist_.get() + (size_t)epoch_ * sdense_,
      sdense_);
  simd_act::F16EncodeRow(cs, last_state_.get() + (size_t)epoch_ * scell_,
      scell_);
  float* nf = scratch_.get();
  float* ni = nf + scell_;
  float* no = ni + scell_;
  float* fgs = no + scell_;
  float* ins = fgs + scell_;
  float* ogs = ins + scell_;
  float* igs = ogs + scell_;
  float* ts = igs + scell_;
#else
  memcpy(last_state_.get() + (size_t)epoch_ * scell_, cs,
      num_cells_ * sizeof(float));
  float* nf = forget_gate_.norm(epoch_);
  float* ni = input_node_.norm(epoch_);
  float* no = output_gate_.norm(epoch_);
  float* fgs = forget_gate_.state(epoch_);
  float* ins = input_node_.state(epoch_);
  float* ogs = output_gate_.state(epoch_);
  float* igs = input_gate_state_.get() + (size_t)epoch_ * scell_;
  float* ts = tanh_state_.get() + (size_t)epoch_ * scell_;
#endif
#if LSTM_FUSED_GATE_MATVEC
  {
    const float* wsf = forget_gate_.wsymt(input_symbol);
    const float* wsi = input_node_.wsymt(input_symbol);
    const float* wso = output_gate_.wsymt(input_symbol);
#if SIMD_ACT_F16
    // Reads the fused fp16 shadow ([wf|wi|wo] per cell, one contiguous
    // half-width stream) over the padded row width; the input row's padding
    // lanes up to sdense_ are zero (sdense_ <= sli_), so the extra lanes
    // contribute nothing.
    const size_t sdense = forget_gate_.sdense_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float ff, fi, fo;
      const uint16_t* rf = forget_gate_.whdense(i);
      simd_act::F16Dot3(input, rf, rf + sdense, rf + 2 * sdense, sdense,
          &ff, &fi, &fo);
      nf[i] = wsf[i] + ff;
      ni[i] = wsi[i] + fi;
      no[i] = wso[i] + fo;
    }
#else
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float ff = wsf[i];
      float fi = wsi[i];
      float fo = wso[i];
      const float* rf = forget_gate_.wdense(i);
      const float* ri = input_node_.wdense(i);
      const float* ro = output_gate_.wdense(i);
      for (unsigned int j = 0; j < dense_width_; ++j) {
        ff += input[j] * rf[j];
        fi += input[j] * ri[j];
        fo += input[j] * ro[j];
      }
      nf[i] = ff;
      ni[i] = fi;
      no[i] = fo;
    }
#endif
  }
#else
  {
    NeuronLayer* gates[3] = {&forget_gate_, &input_node_, &output_gate_};
    float* nrms[3] = {nf, ni, no};
    for (int g = 0; g < 3; ++g) {
      NeuronLayer& neurons = *gates[g];
      float* nrm = nrms[g];
      const float* ws = neurons.wsymt(input_symbol);
      for (unsigned int i = 0; i < num_cells_; ++i) {
        float f = ws[i];
        const float* row = neurons.wdense(i);
        for (unsigned int j = 0; j < dense_width_; ++j) {
          f += input[j] * row[j];
        }
        nrm[i] = f;
      }
    }
  }
#endif
  LayerNorm(forget_gate_, nf, fgs);
  LayerNorm(input_node_, ni, ins);
  LayerNorm(output_gate_, no, ogs);
#if SIMD_ACT_F16
  // BackwardPass reads the normalized (pre-gamma/beta) rows.
  simd_act::F16EncodeRow(nf, forget_gate_.norm(epoch_), scell_);
  simd_act::F16EncodeRow(ni, input_node_.norm(epoch_), scell_);
  simd_act::F16EncodeRow(no, output_gate_.norm(epoch_), scell_);
#endif
  // step_001_002: batched branch-free vector activations over the whole
  // cell array replace per-cell scalar libm calls. Float results drift
  // slightly from libm; validated by roundtrip + compressed-size gate.
  simd_act::Logistic(fgs, num_cells_);
  simd_act::Tanh(ins, num_cells_);
  simd_act::Logistic(ogs, num_cells_);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    igs[i] = 1.0f - fgs[i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    cs[i] *= fgs[i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    cs[i] += ins[i] * igs[i];
  }
  simd_act::Tanh(ts, cs, num_cells_);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    hidden[hidden_start + i] = ogs[i] * ts[i];
  }
#if SIMD_ACT_F16
  simd_act::F16EncodeRow(fgs, forget_gate_.state(epoch_), scell_);
  simd_act::F16EncodeRow(ins, input_node_.state(epoch_), scell_);
  simd_act::F16EncodeRow(ogs, output_gate_.state(epoch_), scell_);
  simd_act::F16EncodeRow(igs,
      input_gate_state_.get() + (size_t)epoch_ * scell_, scell_);
  simd_act::F16EncodeRow(ts,
      tanh_state_.get() + (size_t)epoch_ * scell_, scell_);
#endif
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

inline void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
  }
}

inline void LstmLayer::ClipGradients(float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (p[i] < -gradient_clip_) p[i] = -gradient_clip_;
    else if (p[i] > gradient_clip_) p[i] = gradient_clip_;
  }
}

inline void LstmLayer::BackwardPass(const float* input, int epoch,
    int layer, int input_symbol, float* hidden_error) {
#if SIMD_ACT_F16
  // Decode this epoch's fp16 histories once into fp32 scratch rows; the
  // loop bodies below are then textually identical to the fp32 code (the
  // operands carry fp16 quantization - drift-gated, not bit-exact). The
  // scratch rows are free here: ForwardPass only uses them mid-call.
  (void)input;
  float* tsd = scratch_.get();
  float* igsd = tsd + scell_;
  float* lsd = igsd + scell_;
  float* fgsd = lsd + scell_;
  float* insd = fgsd + scell_;
  float* ogsd = insd + scell_;
  simd_act::F16DecodeRow(tanh_state_.get() + (size_t)epoch * scell_, tsd,
      scell_);
  simd_act::F16DecodeRow(input_gate_state_.get() + (size_t)epoch * scell_,
      igsd, scell_);
  simd_act::F16DecodeRow(last_state_.get() + (size_t)epoch * scell_, lsd,
      scell_);
  simd_act::F16DecodeRow(forget_gate_.state(epoch), fgsd, scell_);
  simd_act::F16DecodeRow(input_node_.state(epoch), insd, scell_);
  simd_act::F16DecodeRow(output_gate_.state(epoch), ogsd, scell_);
  const float* ts = tsd;
  const float* igs = igsd;
  const float* ls = lsd;
  const float* fgs = fgsd;
  const float* ins = insd;
  const float* ogs = ogsd;
#else
  const float* ts = tanh_state_.get() + (size_t)epoch * scell_;
  const float* igs = input_gate_state_.get() + (size_t)epoch * scell_;
  const float* ls = last_state_.get() + (size_t)epoch * scell_;
  const float* fgs = forget_gate_.state(epoch);
  const float* ins = input_node_.state(epoch);
  const float* ogs = output_gate_.state(epoch);
  input_ptrs_[epoch] = input;
#endif
  // Gate errors are written straight into this epoch's row of the per-gate
  // err_ buffer (the old error_ scratch vector), so the fully post-processed
  // value is retained for the deferred dense-gradient outer-product.
  float* oge = output_gate_.err(epoch);
  float* ine = input_node_.err(epoch);
  float* fge = forget_gate_.err(epoch);
  if (epoch == (int)horizon_ - 1) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      stored_error_[i] = hidden_error[i];
    }
    state_error_ = 0;
  } else {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      stored_error_[i] += hidden_error[i];
    }
  }

  for (unsigned int i = 0; i < num_cells_; ++i) {
    oge[i] = ts[i] * stored_error_[i] * ogs[i] *
        (1.0f - ogs[i]);
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    state_error_[i] += stored_error_[i] * ogs[i] * (1.0f - (ts[i] * ts[i]));
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    ine[i] = state_error_[i] * igs[i] *
        (1.0f - (ins[i] * ins[i]));
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    fge[i] = (ls[i] - ins[i]) * state_error_[i] * fgs[i] *
        igs[i];
  }

  for (unsigned int i = 0; i < num_cells_; ++i) {
    hidden_error[i] = 0;
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      state_error_[i] *= fgs[i];
    }
    stored_error_ = 0;
  } else {
    if (update_steps_ < UPDATE_LIMIT) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error, num_cells_);
}

inline void LstmLayer::BackwardPass(NeuronLayer& neurons, const float* input,
    int epoch, int layer, int input_symbol, float* hidden_error) {
  float* err = neurons.err(epoch);
#if SIMD_ACT_F16
  // The gate's normalized row, widened once from its fp16 history. Scratch
  // row 6 is unused by the outer BackwardPass's decodes (rows 0-5), which
  // are all dead by the time the per-gate passes run.
  float* nrm = scratch_.get() + (size_t)6 * scell_;
  simd_act::F16DecodeRow(neurons.norm(epoch), nrm, scell_);
#else
  float* nrm = neurons.norm(epoch);
#endif
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    for (size_t s = 0; s < neurons.sym_width_; ++s) {
      memset(neurons.usymt(s), 0, neurons.scell_ * sizeof(float));
    }
    // Blocked tile transpose: the old per-cell column walk stored at stride
    // scell_ (a fresh cache line per store); 16x16 tiles keep both the
    // wdense source lines and the transpose destination lines hot.
    for (size_t j0 = 0; j0 < neurons.trows_; j0 += 16) {
      const size_t jend = std::min(j0 + 16, neurons.trows_);
      for (size_t i0 = 0; i0 < num_cells_; i0 += 16) {
        const size_t iend = std::min(i0 + 16, (size_t)num_cells_);
        for (size_t j = j0; j < jend; ++j) {
          float* tr = neurons.transpose(j);
          for (size_t i = i0; i < iend; ++i) {
            tr[i] = neurons.wdense(i)[input_size_ + j];
          }
        }
      }
    }
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.beta_u_[i] += err[i];
  }
  for (unsigned int i = 0; i < num_cells_; ++i) {
    neurons.gamma_u_[i] += err[i] * nrm[i];
  }
  {
    const float ivar = neurons.ivar_[epoch];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      err[i] *= neurons.gamma_[i] * ivar;
    }
  }
  {
    float f = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      f += err[i] * nrm[i];
    }
    const float scale = f / num_cells_;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      err[i] -= scale * nrm[i];
    }
  }
  if (layer > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      const float* tr = neurons.transpose(num_cells_ + i);
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err[j] * tr[j];
      }
      hidden_error[i] += f;
    }
  }
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      const float* tr = neurons.transpose(i);
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err[j] * tr[j];
      }
      stored_error_[i] += f;
    }
  }
  // One contiguous row accumulate (symbol block is stored transposed)
  // replaces the per-cell stride-ssym_ column walk.
  {
    float* us = neurons.usymt(input_symbol);
    for (unsigned int i = 0; i < num_cells_; ++i) {
      us[i] += err[i];
    }
  }
  if (epoch == 0) {
    const AdamParams p = MakeAdamParams(learning_rate_, update_steps_);
    // Deferred dense-gradient materialization: u = sum_e err_e (x) input_e,
    // built once per burst in 16-cell blocks (the u block stays L1-resident
    // across the epoch sweep; each input row is streamed once per block)
    // instead of accumulating into the full u slab every epoch. Adam then
    // consumes each block while it is still hot. Summation order over epochs
    // differs from the old epoch-by-epoch accumulation (drift-gated step,
    // not bit-exact). On F16 targets the input rows are read from the fp16
    // history written by ForwardPass (halving the re-streamed bytes); the
    // u accumulation itself stays fp32.
    for (unsigned int i0 = 0; i0 < num_cells_; i0 += 16) {
      const unsigned int iend = std::min(i0 + 16, num_cells_);
      for (unsigned int i = i0; i < iend; ++i) {
        memset(neurons.udense(i), 0, dense_width_ * sizeof(float));
      }
      for (unsigned int e = 0; e < horizon_; ++e) {
        const float* er = neurons.err(e);
#if SIMD_ACT_F16
        const uint16_t* inph = inp_hist_.get() + (size_t)e * sdense_;
        for (unsigned int i = i0; i < iend; ++i) {
          simd_act::F16Axpy(neurons.udense(i), inph, er[i], sdense_);
        }
#else
        const float* __restrict inp = input_ptrs_[e];
        for (unsigned int i = i0; i < iend; ++i) {
          const float ei = er[i];
          float* __restrict ud = neurons.udense(i);
          for (unsigned int j = 0; j < dense_width_; ++j) {
            ud[j] += ei * inp[j];
          }
        }
#endif
      }
      for (unsigned int i = i0; i < iend; ++i) {
        Adam(p, neurons.udense(i), neurons.mdense(i), neurons.vdense(i),
            neurons.wdense(i), neurons.dense_width_);
#if SIMD_ACT_F16
        // Refresh the fp16 shadow while the updated w row is still hot.
        simd_act::F16EncodeRow(neurons.wdense(i), neurons.whdense(i),
            neurons.sdense_);
#endif
      }
    }
    for (size_t s = 0; s < neurons.sym_width_; ++s) {
      Adam(p, neurons.usymt(s), neurons.msymt(s), neurons.vsymt(s),
          neurons.wsymt(s), num_cells_);
    }
    Adam(p, &neurons.gamma_u_[0], &neurons.gamma_m_[0], &neurons.gamma_v_[0],
        &neurons.gamma_[0], num_cells_);
    Adam(p, &neurons.beta_u_[0], &neurons.beta_m_[0], &neurons.beta_v_[0],
        &neurons.beta_[0], num_cells_);
  }
}
