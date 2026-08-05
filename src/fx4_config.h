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
#define FX4_PPMD_REMAP_INTERVAL 5000ull
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

// Baseline-anchored correction over existing PPMd, LSTM and FXCM predictions.
// It uses only decoder-visible stream state and has no archive side data.
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
#define FX4_VIRTUAL_REPLAY 1
#endif

// Reversible structural-token transform for the post-R1 predictor stream.
// Compression opts in with FX4_ENABLE_SCR2=1; archives carry their own mode.
#ifndef FX4_SCR2
#define FX4_SCR2 0
#endif

#ifndef FX4_SCR2_DEFAULT
#define FX4_SCR2_DEFAULT 0
#endif

// Add three SCR2-only structural experts to the outer context mixer. Raw
// archives do not construct them and retain the accepted predictor shape.
#ifndef FX4_SCR2_SPECIALIST
#define FX4_SCR2_SPECIALIST 1
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
#define FX4_SELECTIVE_POSTR1 1
#endif


// Reversible block portfolio between R1 and the entropy models. Compression
// opts in with FX4_POSTR1_TRANSFORM_PLAN; the resulting F4PT stream is
// self-contained and restored before the R1 inverse.
#ifndef FX4_POSTR1_TRANSFORM
#define FX4_POSTR1_TRANSFORM 1
#endif
#endif