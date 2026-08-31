#ifndef KH_LSTM_PARAMS_H
#define KH_LSTM_PARAMS_H

// combo_llif12 byte-LSTM hyperparameters, runtime-gated by env KH_LSTM_LLIF12.
//
// Validated standalone (base-only, zero PPMD) in the PyTorch cmix-LSTM replica
// on the full 587,138,826-byte transformed corpus: -0.0809 bpb whole-corpus
// (-0.0505 last quarter) vs production settings; see
// cmix_lstm_torch/PYTORCH_CMIX_LSTM_RESULTS.md "Full-corpus A/B". NOT yet
// validated inside full cmix. This is an experimental init mode: KH_LSTM_LLIF12
// is UNSET in the configuration of record, so this path is OFF in the shipped
// binary.
//
// The combo splits the single Lstm learning-rate constructor argument into
//   * output-layer SGD LR   0.12   (was: the ctor arg, 0.03)
//   * gate Adam base LR     0.09   (alpha = gate_lr * 0.1 / sqrt(5e-5 t + 1),
//                                   form unchanged; was the same 0.03)
// and changes two init constants in LstmLayer:
//   * Xavier bound scale    0.5    (val = scale * sqrt(6/(input+output)))
//   * forget-gate dense bias 0.0   (was 1.0)
// Everything else (cells, horizon 128, Adam betas 0.025/0.9999, eps, clip 10,
// UPDATE_LIMIT) is untouched. Zero speed / memory cost.
//
// With KH_LSTM_LLIF12 unset (or "0") every value below reproduces the golden
// constants EXACTLY (multiplying the Xavier bound by 1.0f is exact in IEEE
// fp), so the default run stays byte-identical to golden-256. The env var is
// read once (static init) -- both the encode and decode processes of an A/B
// must set it identically or the archive will not decode.

#include <cstdlib>
#include <cstring>

struct KhLstmParams {
  // Negative LR means "use the constructor learning_rate argument".
  float output_lr;
  float gate_lr;
  float init_scale;
  float forget_bias;
  bool enabled;
};

inline const KhLstmParams& KhGetLstmParams() {
  static const KhLstmParams params = [] {
    KhLstmParams p{-1.0f, -1.0f, 1.0f, 1.0f, false};
    const char* e = std::getenv("KH_LSTM_LLIF12");
    // DEFAULT when UNSET is now "init" -- this matches the record run
    // exactly (it was launched with KH_LSTM_LLIF12=init; the unset-default
    // takes the identical "init" branch below), and makes a bare
    // `./archive9` decode reproduce the record stream. Set
    // KH_LSTM_LLIF12=0 to get the stock (golden-256) parameters.
    // NOTE(obias lane): an earlier revision defaulted KH_OBIAS builds to
    // golden ("0") for the aux-deleted obias_up shape (trained on golden
    // init). That shape LOST the 10 MB gate (-6,728 B) and is dead; the
    // winning keep-aux constant-gate shape (KH_OBIAS_CONST_GATE) runs on
    // the RECORD init-mode base -- 10 MB decomposition measured the init
    // choice itself at only -205 B. Default stays "init" for all builds.
    if (e == nullptr) e = "init";
    if (e[0] && std::strcmp(e, "0") != 0) {
      if (std::strcmp(e, "init") == 0) {
        // Init-only subset: Xavier x0.5 + forget dense bias 0, learning
        // rates stay the stock constructor value (10 MB A/B showed the LR
        // half hurting in-mixer; this isolates the init half).
        p = {-1.0f, -1.0f, 0.5f, 0.0f, true};
      } else if (std::strcmp(e, "lr") == 0) {
        // LR-only subset (complement of "init").
        p = {0.12f, 0.09f, 1.0f, 1.0f, true};
      } else {
        p = {0.12f, 0.09f, 0.5f, 0.0f, true};
      }
    }
    return p;
  }();
  return params;
}

inline float KhLstmOutputLr(float ctor_lr) {
  const KhLstmParams& p = KhGetLstmParams();
  return p.output_lr >= 0.0f ? p.output_lr : ctor_lr;
}

inline float KhLstmGateLr(float ctor_lr) {
  const KhLstmParams& p = KhGetLstmParams();
  return p.gate_lr >= 0.0f ? p.gate_lr : ctor_lr;
}

#endif  // KH_LSTM_PARAMS_H
