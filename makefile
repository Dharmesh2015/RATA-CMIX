CXX := clang++
OUT ?= cmix

.DEFAULT_GOAL := target93

# Single production configuration. Research profiles and feature switches live
# only on exp/selective-discovery and cannot be enabled from this branch.
DEFINES := -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG \
	-DFX4_STDERR_PROGRESS=0 -DFX4_PROGRESS_LOG=0 \
	-DFX4_LSTM_MID_BRIDGE=2 \
	-DFX4_TRANSFORMER6M=1 -DFX4_TRANSFORMER6M_REQUIRED=1 \
	-DFX4_TRANSFORMER_REPLACES_LSTM=1 -DKH_TRANSFORMER6M_ARCHIVE \
	-DFX4_LSTM_CELLS=200 -DFX4_LSTM_LAYERS=1 \
	-DFX4_LSTM_HORIZON=128 -DFX4_LSTM_LEARNING_RATE=0.03f \
	-DFX4_DEEPMIX_CONTEXTS=1 -DFX4_GRAMMAR_MATCH=1 -DGM_REVTS=1 \
	-DFX4_ESN_NLMS=1 -DFX4_SPECIALIST_CORRECTOR=1 \
	-DFX4_TARGET93_CANONICAL=1 \
	-DCMIX_PPMD_RSS_BUDGET_MB=8704

ARCH_FLAGS ?= -march=native -mtune=native
COMMON := $(DEFINES) -m64 -Wall -std=c++17 -fno-exceptions \
	-fno-unwind-tables -fno-asynchronous-unwind-tables \
	-fno-threadsafe-statics -Wno-unknown-escape-sequence \
	-Wno-unused-variable -Wno-unneeded-internal-declaration \
	-Wno-unused-but-set-variable -Wno-format $(ARCH_FLAGS) \
	-fdata-sections -ffunction-sections
FAST_FLAGS := $(COMMON) -O3 -ffp-model=fast
SLOW_FLAGS := $(COMMON) -Os -ffp-model=fast
COLD_FLAGS := $(COMMON) -Oz -ffp-model=fast
TRANSFORMER_FLAGS := -m64 -O3 -std=c++17 -Wall -Wextra \
	-fno-math-errno $(ARCH_FLAGS) -fdata-sections -ffunction-sections
LDFLAGS := -m64 -fuse-ld=lld -Wl,--gc-sections -std=c++17

FAST_SOURCES := \
	src/coder/decoder.cpp src/coder/encoder.cpp \
	src/context-manager.cpp \
	src/contexts/bit-context.cpp src/contexts/bracket-context.cpp \
	src/contexts/combined-context.cpp src/contexts/context-hash.cpp \
	src/contexts/indirect-hash.cpp src/contexts/interval-hash.cpp \
	src/contexts/interval.cpp src/contexts/sparse.cpp \
	src/models/bracket.cpp src/models/byte-model.cpp \
	src/models/direct-hash.cpp src/models/direct.cpp src/models/match.cpp \
	src/models/fxcmv1.cpp src/models/grammar-match.cpp \
	src/models/esn-nlms.cpp src/models/ppmd.cpp \
	src/states/nonstationary.cpp src/states/run-map.cpp \
	src/mixer/byte-mixer.cpp src/mixer/mixer-input.cpp \
	src/mixer/mixer.cpp src/mixer/sigmoid.cpp src/mixer/sse.cpp \
	src/predictor.cpp

SLOW_SOURCES := \
	src/preprocess/preprocessor.cpp src/preprocess/dictionary.cpp

COLD_SOURCES := src/r1_reorder_transform.cpp src/runner.cpp

TRANSFORMER_OBJECTS := tf_weights_io_compressed.o \
	tf_qmat_dense.o tf_qmat_sparse.o tf_attn.o tf_kda.o tf_glue.o \
	tf_arena_build.o tf_model_opt.o

.PHONY: target93 cmix fast slow cold transformer_objects clean

target93: cmix

fast:
	$(CXX) $(FAST_FLAGS) $(FAST_SOURCES) -c

slow:
	$(CXX) $(SLOW_FLAGS) $(SLOW_SOURCES) -c

cold:
	$(CXX) $(COLD_FLAGS) $(COLD_SOURCES) -c

tf_weights_io_compressed.o: \
	src/third_party/fx2_transformer/weights_io_compressed.cpp \
	src/third_party/fx2_transformer/weights_io.h
	$(CXX) $(filter-out -O3,$(TRANSFORMER_FLAGS)) -Os \
		-DFX2_TRANSFORMER_COMPRESSED_ONLY=1 -c $< -o $@

tf_qmat_dense.o: src/third_party/fx2_transformer/opt/qmat_dense.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_qmat_sparse.o: src/third_party/fx2_transformer/opt/qmat_sparse.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_attn.o: src/third_party/fx2_transformer/opt/attn.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_kda.o: src/third_party/fx2_transformer/opt/kda.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_glue.o: src/third_party/fx2_transformer/opt/glue.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_arena_build.o: src/third_party/fx2_transformer/opt/arena_build.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -DFX2_TRANSFORMER_COMPRESSED_ONLY=1 \
		-c $< -o $@

tf_model_opt.o: src/third_party/fx2_transformer/opt/model_opt.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

transformer_objects: $(TRANSFORMER_OBJECTS)

cmix: fast slow cold transformer_objects
	$(CXX) $(LDFLAGS) \
		bit-context.o bracket-context.o bracket.o byte-mixer.o \
		byte-model.o combined-context.o context-hash.o context-manager.o \
		decoder.o dictionary.o direct-hash.o direct.o encoder.o \
		esn-nlms.o fxcmv1.o grammar-match.o indirect-hash.o \
		interval-hash.o interval.o match.o mixer-input.o mixer.o \
		nonstationary.o ppmd.o predictor.o preprocessor.o \
		r1_reorder_transform.o run-map.o runner.o sigmoid.o sparse.o sse.o \
		$(TRANSFORMER_OBJECTS) -s -o $(OUT)
	rm -f *.o

clean:
	rm -f *.o cmix cmix_orig
