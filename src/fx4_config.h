#ifndef FX4_CONFIG_H
#define FX4_CONFIG_H

#ifndef FX4_PROGRESS_STEPS
#define FX4_PROGRESS_STEPS 100
#endif

#ifndef FX4_PROGRESS_LOG
#define FX4_PROGRESS_LOG 0
#endif

#ifndef FX4_STDERR_PROGRESS
#define FX4_STDERR_PROGRESS 0
#endif

// Set only by the single packaged CPU target. The checks at the end of this
// file make accidental research switches a compile-time error.
#ifndef FX4_TARGET93_CANONICAL
#define FX4_TARGET93_CANONICAL 0
#endif

#ifndef FX4_IO_BUFFER_BYTES
#define FX4_IO_BUFFER_BYTES (1u << 20)
#endif

// Accepted cmix-lex-compatible model configuration. Keep these values in one
// place so experimental builds cannot silently select a different predictor.
#ifndef FX4_PPMD_ORDER
#define FX4_PPMD_ORDER 25
#endif

#ifndef FX4_PPMD_MEMORY_MB
#define FX4_PPMD_MEMORY_MB 14000
#endif

#ifndef FX4_PPMD_MMAP_TO_DISK
#define FX4_PPMD_MMAP_TO_DISK 1
#endif

// cmix-lex cadence retained for the 14 GB file-backed heap and 10 GB RSS cap.
#ifndef FX4_PPMD_REMAP_INTERVAL
#define FX4_PPMD_REMAP_INTERVAL 65536ull
#endif

// altxs M3+M5 outer transform. M3 densifies PHDA9 before WRT and M5 replaces
// payload_lex/R1 with a reversible structural-key + SimHash block order. The
// complete inverse side data is sealed into the entropy-coded product.
#ifndef FX4_ALTXS_M3_M5
#define FX4_ALTXS_M3_M5 0
#endif

// Optional fx2-cmix 6M CPU transformer replacement for the online byte LSTM.
// It is deliberately independent from M3/M5: the supplied public weights are
// valid only for the stream/vocabulary on which they were trained.
#ifndef FX4_TRANSFORMER6M
#define FX4_TRANSFORMER6M 0
#endif

// Target93/S1 builds must fail rather than silently falling back to the
// online LSTM when the separately packaged transformer weights are missing.
#ifndef FX4_TRANSFORMER6M_REQUIRED
#define FX4_TRANSFORMER6M_REQUIRED 0
#endif

// The published fx2-cmix-transformer uses the frozen transformer instead of
// the online byte LSTM. Keep this separate from the additive M3/M5 ablation:
// the supplied weights are only validated on the original 205-symbol stream.
#ifndef FX4_TRANSFORMER_REPLACES_LSTM
#define FX4_TRANSFORMER_REPLACES_LSTM 0
#endif

// CPU-only model additions retained from fx-deepmix. These gates are kept
// explicit because their gains were measured on its cmix-lex stream and must
// still be revalidated after FX4's M3/M5 transform.
#ifndef FX4_DEEPMIX_CONTEXTS
#define FX4_DEEPMIX_CONTEXTS 0
#endif

#ifndef FX4_GRAMMAR_MATCH
#define FX4_GRAMMAR_MATCH 0
#endif

// Tiny deterministic echo-state/NLMS correction expert. It has no model
// asset or side data: encoder and decoder learn identical state online from
// already-coded bits. The expert is mixed against the accepted probability
// through an adaptive baseline anchor so weak periods remain near baseline.
#ifndef FX4_ESN_NLMS
#define FX4_ESN_NLMS 0
#endif

// Lightweight Nacrith-inspired causal side expert. It uses bounded byte/WRT
// n-gram counts plus an online log-space bias; no LLM/GPU code is included.
#ifndef FX4_TOKEN_NGRAM_BIAS
#define FX4_TOKEN_NGRAM_BIAS 0
#endif

#ifndef FX4_LSTM_CELLS
#define FX4_LSTM_CELLS 170
#endif

#ifndef FX4_LSTM_LAYERS
#define FX4_LSTM_LAYERS 1
#endif

#ifndef FX4_LSTM_HORIZON
#define FX4_LSTM_HORIZON 128
#endif

#ifndef FX4_LSTM_LEARNING_RATE
#define FX4_LSTM_LEARNING_RATE 0.055f
#endif

#ifndef FX4_LSTM_GRADIENT_CLIP
#define FX4_LSTM_GRADIENT_CLIP 10.0f
#endif

// Research-only LSTM-200 shadow. It is trained continuously from byte zero,
// but its probability is consumed only by explicitly selected post-R1 spans.
// The accepted LSTM-170 and final predictor remain untouched.
#ifndef FX4_SHADOW_LSTM200
#define FX4_SHADOW_LSTM200 0
#endif

#ifndef FX4_SHADOW_LSTM_CELLS
#define FX4_SHADOW_LSTM_CELLS 200
#endif

// Versioned donor-plan support is research-only until exact net archive gain
// exceeds its metadata and executable cost.
#ifndef FX4_DONOR_PLAN
#define FX4_DONOR_PLAN 0
#endif

// Linux-only exact donor discovery. This is never enabled in the accepted
// compressor: it forks at recipient boundaries so baseline and donor trials
// start from identical predictor state.
#ifndef FX4_DONOR_FORK_DISCOVERY
#define FX4_DONOR_FORK_DISCOVERY 0
#endif

// Research-only external donor replay used by the bounded crawler. The
// accepted compressor leaves this at 0, so neither its archive format nor its
// executable contains the bootstrap path. A crawler build sets it to 1 and
// supplies FX4_RESEARCH_DONOR_BOOTSTRAP=<file> to encoder and decoder.
#ifndef FX4_RESEARCH_DONOR_BOOTSTRAP
#define FX4_RESEARCH_DONOR_BOOTSTRAP 0
#endif

// Exact intermediate stream dumps are opt-in research instrumentation. Keep
// them out of production and donor-search binaries unless explicitly enabled.
#ifndef FX4_RESEARCH_STREAM_DUMP
#define FX4_RESEARCH_STREAM_DUMP 0
#endif

// Selective post-R1 virtual replay. A plan is used only when compression sets
// FX4_VR_PLAN; archives carry every pattern and event needed by the decoder.
#ifndef FX4_VIRTUAL_REPLAY
#define FX4_VIRTUAL_REPLAY 0
#endif

// Reversible structural-token transform for the post-R1 predictor stream.
// Compression opts in with FX4_ENABLE_SCR2=1; archives carry their own mode.
#ifndef FX4_SCR2
#define FX4_SCR2 0
#endif

#ifndef FX4_SCR2_DEFAULT
#define FX4_SCR2_DEFAULT 0
#endif

#ifndef FX4_SPECIALIST_CORRECTOR
#define FX4_SPECIALIST_CORRECTOR 1
#endif

#ifndef FX4_SPECIALIST_LEARNING_RATE
#define FX4_SPECIALIST_LEARNING_RATE 0.0005f
#endif


// Selective post-R1 predictor portfolio. Experts are activated only by spans
// carried in the archive plan; with no selected span this path is exactly
// baseline-neutral.
#ifndef FX4_SELECTIVE_POSTR1
#define FX4_SELECTIVE_POSTR1 0
#endif

// Eleven complementary cmix predictors condensed into one post-R1 expert.
// Keep this independent from the other post-R1 experts so donor-only builds
// do not pay its prediction and update cost.
#ifndef FX4_MINI_CMIX
#define FX4_MINI_CMIX 0
#endif
// Gauss/interlacement recurrence expert. Discovery builds enable this to
// measure it page-wise; production builds leave it out unless its aggregate
// archive saving pays for the measured S1 delta.
#ifndef FX4_TOPOLOGY_RECURRENCE
#define FX4_TOPOLOGY_RECURRENCE 0
#endif

// Tiny deterministic causal-convolution residual expert. It is compiled only
// for discovery or after measured aggregate savings pay its final S1 cost.
#ifndef FX4_CAUSAL_CNN
#define FX4_CAUSAL_CNN 0
#endif

// Diagnostic-only compact trace for CPU residual-model/oracle research.
// It never changes a probability or the archive and is excluded from record
// builds. Runtime output is enabled with FX4_ORACLE_TRACE=<path>.
#ifndef FX4_RESIDUAL_ORACLE_TRACE
#define FX4_RESIDUAL_ORACLE_TRACE 0
#endif

// Decoder-first byte-step residual LSTM-96 terminal correction. The model is
// compile- and runtime-gated; without a valid blob the accepted probability
// path remains byte-identical.
#ifndef FX4_RESIDUAL_LSTM96
#define FX4_RESIDUAL_LSTM96 0
#endif



// Reversible block portfolio between R1 and the entropy models. Compression
// opts in with FX4_POSTR1_TRANSFORM_PLAN; the resulting F4PT stream is
// self-contained and restored before the R1 inverse.
#ifndef FX4_POSTR1_TRANSFORM
#define FX4_POSTR1_TRANSFORM 0
#endif

#if FX4_TARGET93_CANONICAL
#if !FX4_TRANSFORMER6M || !FX4_TRANSFORMER_REPLACES_LSTM || \
    !FX4_DEEPMIX_CONTEXTS || !FX4_GRAMMAR_MATCH || !FX4_ESN_NLMS
#error target93 requires transformer replacement, DeepMix/GrammarMatch, and ESN/NLMS
#endif
#if FX4_ALTXS_M3_M5 || FX4_DONOR_PLAN || FX4_DONOR_FORK_DISCOVERY || \
    FX4_RESEARCH_DONOR_BOOTSTRAP || FX4_RESEARCH_STREAM_DUMP || \
    FX4_VIRTUAL_REPLAY || FX4_SCR2 || FX4_SELECTIVE_POSTR1 || \
    FX4_MINI_CMIX || FX4_SHADOW_LSTM200 || FX4_TOKEN_NGRAM_BIAS || \
    FX4_TOPOLOGY_RECURRENCE || FX4_CAUSAL_CNN || \
    FX4_RESIDUAL_ORACLE_TRACE || FX4_RESIDUAL_LSTM96 || \
    FX4_POSTR1_TRANSFORM
#error target93 cannot include stream-changing, discovery, or rejected experts
#endif
#ifdef KH_OBIAS
#error target93 cannot include the online-LSTM obias head
#endif
#endif
#endif
