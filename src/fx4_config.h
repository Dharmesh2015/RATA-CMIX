#ifndef FX4_CONFIG_H
#define FX4_CONFIG_H

// This branch has one supported model configuration. The makefile defines the
// production feature set; this header owns only numeric runtime constants.

#define FX4_PROGRESS_STEPS 100
#define FX4_PROGRESS_LOG 0
#define FX4_STDERR_PROGRESS 0
#define FX4_IO_BUFFER_BYTES (1u << 20)

#define FX4_PPMD_ORDER 25
#define FX4_PPMD_MEMORY_MB 14000
#define FX4_PPMD_MMAP_TO_DISK 1
#define FX4_PPMD_REMAP_INTERVAL 65536ull

#define FX4_LSTM_GRADIENT_CLIP 10.0f
#define FX4_SPECIALIST_LEARNING_RATE 0.0005f

#if !FX4_TARGET93_CANONICAL || !FX4_TRANSFORMER6M || \
    !FX4_TRANSFORMER6M_REQUIRED || !FX4_TRANSFORMER_REPLACES_LSTM || \
    !FX4_DEEPMIX_CONTEXTS || !FX4_GRAMMAR_MATCH || !FX4_ESN_NLMS || \
    !FX4_SPECIALIST_CORRECTOR
#error The clean branch supports only the canonical target93 production build
#endif

#if defined(FX4_DONOR_PLAN) || defined(FX4_DONOR_FORK_DISCOVERY) || \
    defined(FX4_VIRTUAL_REPLAY) || defined(FX4_SCR2) || \
    defined(FX4_SELECTIVE_POSTR1) || defined(FX4_MINI_CMIX) || \
    defined(FX4_SHADOW_LSTM200) || defined(FX4_ALTXS_M3_M5) || \
    defined(FX4_POSTR1_TRANSFORM) || defined(FX4_RESIDUAL_ORACLE_TRACE) || \
    defined(FX4_RESIDUAL_LSTM96) || defined(KH_OBIAS) || \
    defined(KH_BITLSTM32)
#error Discovery and experimental model switches are unavailable on this branch
#endif

#endif
