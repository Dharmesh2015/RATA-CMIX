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
#define FX4_LSTM_LEARNING_RATE 0.03f
#endif

#ifndef FX4_LSTM_GRADIENT_CLIP
#define FX4_LSTM_GRADIENT_CLIP 10.0f
#endif

#endif