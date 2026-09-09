CXX := clang++-17
OUT ?= cmix
CC_C := clang-17
STRIP_FLAG ?= -s
PGO ?= use
PGO_PROFILE ?= pgo/default.profdata
PGO_RAW_DIR := pgo-raw

.DEFAULT_GOAL := target93

# Single production configuration. Research profiles and feature switches live
# only on exp/selective-discovery and cannot be enabled from this branch.
# One deterministic configuration. Every model switch is compiled in at its
# measured setting; nothing here is selectable at run time and there are no
# environment variables that change a probability.
#
# The stack is v22p + 521 + T1 + T2 + T2-E + T2-ED + T2-EDG + Stationary +
# GrammarMatch + the grammar mixer context + GM_ARM, with a 2x200 online LSTM
# expert. Those defaults live in src/predictor.h and src/fx4_config.h and are
# not repeated here: a value written in two places is a value that will
# disagree with itself.
# SEED is read by runner.cpp, UPDATE_LIMIT by the LSTM's Adam step.
# Nothing else needs to arrive from here: every model setting is fixed
# in the source it belongs to.
DEFINES := -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG

NATIVE ?= 0
ifeq ($(NATIVE),1)
ARCH_FLAGS := -march=native -mtune=native
else
ARCH_FLAGS ?= -march=x86-64-v3 -mtune=generic
endif

LTO ?= thin
ifeq ($(LTO),full)
LTO_FLAGS := -flto
else ifeq ($(LTO),thin)
LTO_FLAGS := -flto=thin
else ifeq ($(LTO),off)
LTO_FLAGS :=
else
$(error LTO must be full, thin, or off)
endif

# PGO=generate builds the instrumented binary used only by the
# pgo-instrumented target below. The default, PGO=use, applies the
# committed pgo/default.profdata automatically whenever it's present, so
# a plain `make target93` -- exactly what build.sh runs during judging,
# with no extra flags or steps available to it -- already gets the
# profile-guided build. A missing profile falls back to a plain
# optimized (still LTO'd) build with a make-time warning, never a hard
# failure: this repository must still build without the profile checked
# out.
ifeq ($(PGO),generate)
PGO_FLAGS := -fprofile-generate=$(PGO_RAW_DIR)
else ifeq ($(PGO),use)
PGO_FLAGS := $(if $(wildcard $(PGO_PROFILE)),-fprofile-use=$(PGO_PROFILE) -Wno-profile-instr-out-of-date,)
ifeq ($(wildcard $(PGO_PROFILE)),)
$(warning $(PGO_PROFILE) not found; building without profile guidance)
endif
else ifeq ($(PGO),off)
PGO_FLAGS :=
else
$(error PGO must be generate, use, or off)
endif

COMMON := $(DEFINES) -m64 -Wall -std=c++17 -fno-exceptions \
	-fno-unwind-tables -fno-asynchronous-unwind-tables \
	-fno-threadsafe-statics -Wno-unknown-escape-sequence \
	-Wno-unused-variable -Wno-unneeded-internal-declaration \
	-Wno-unused-but-set-variable -Wno-format $(ARCH_FLAGS) \
	-fdata-sections -ffunction-sections -fno-semantic-interposition \
	$(LTO_FLAGS) $(PGO_FLAGS)
FAST_FLAGS := $(COMMON) -O3 -ffp-model=fast
SLOW_FLAGS := $(COMMON) -Os -ffp-model=fast
COLD_FLAGS := $(COMMON) -Oz -ffp-model=fast
TRANSFORMER_FLAGS := -m64 -O3 -std=c++17 -Wall -Wextra \
	-fno-math-errno $(ARCH_FLAGS) -fdata-sections -ffunction-sections \
	$(LTO_FLAGS) $(PGO_FLAGS)
LDFLAGS := -m64 -fuse-ld=lld -Wl,--gc-sections -Wl,--icf=safe \
	-std=c++17 $(LTO_FLAGS) \
	$(PGO_FLAGS)

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
	src/models/scr2-match.cpp \
	src/models/morphology-match.cpp \
	src/models/causal-donor.cpp \
	src/models/ppmd.cpp \
	src/states/nonstationary.cpp src/states/run-map.cpp \
	src/mixer/byte-mixer.cpp src/mixer/mixer-input.cpp \
	src/mixer/mixer.cpp src/mixer/sigmoid.cpp src/mixer/sse.cpp \
	src/predictor.cpp

SLOW_SOURCES := \
	src/preprocess/preprocessor.cpp src/preprocess/dictionary.cpp

COLD_SOURCES := src/runner.cpp

TRANSFORMER_OBJECTS := tf_weights_io_compressed.o \
	tf_qmat_dense.o tf_qmat_sparse.o tf_attn.o tf_kda.o tf_glue.o \
	tf_arena_build.o tf_model_opt.o

.PHONY: target93 cmix fast slow cold transformer_objects clean \
	pgo-instrumented pgo-merge fast-native

target93: cmix

# Local-machine throughput build. The portable judged build remains target93.
# Regenerate the profile on this machine first for the best result.
fast-native:
	$(MAKE) target93 NATIVE=1 LTO=full PGO=use OUT=cmix_fast_native

fast:
	$(CXX) $(FAST_FLAGS) $(FAST_SOURCES) -c

slow:
	$(CXX) $(SLOW_FLAGS) $(SLOW_SOURCES) -c

cold:
	$(CXX) $(COLD_FLAGS) $(COLD_SOURCES) -c

tf_weights_io_compressed.o: \
	cpp_infer/src/weights_io_compressed.cpp \
	cpp_infer/src/weights_io.h
	$(CXX) $(filter-out -O3,$(TRANSFORMER_FLAGS)) -Os \
		-DFX2_TRANSFORMER_COMPRESSED_ONLY=1 -c $< -o $@

tf_qmat_dense.o: cpp_infer/src/opt/qmat_dense.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_qmat_sparse.o: cpp_infer/src/opt/qmat_sparse.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_attn.o: cpp_infer/src/opt/attn.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_kda.o: cpp_infer/src/opt/kda.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_glue.o: cpp_infer/src/opt/glue.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

tf_arena_build.o: cpp_infer/src/opt/arena_build.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -DFX2_TRANSFORMER_COMPRESSED_ONLY=1 \
		-c $< -o $@

tf_model_opt.o: cpp_infer/src/opt/model_opt.cpp
	$(CXX) $(TRANSFORMER_FLAGS) -c $< -o $@

transformer_objects: $(TRANSFORMER_OBJECTS)

cmix: fast slow cold transformer_objects
	$(CXX) $(LDFLAGS) \
		bit-context.o bracket-context.o bracket.o byte-mixer.o \
		byte-model.o combined-context.o context-hash.o context-manager.o \
		decoder.o dictionary.o direct-hash.o direct.o encoder.o \
		fxcmv1.o grammar-match.o scr2-match.o morphology-match.o causal-donor.o indirect-hash.o \
		interval-hash.o interval.o match.o mixer-input.o mixer.o \
		nonstationary.o ppmd.o predictor.o preprocessor.o \
		run-map.o runner.o sigmoid.o sparse.o sse.o \
		$(TRANSFORMER_OBJECTS) $(STRIP_FLAG) -o $(OUT)
	rm -f *.o

# Builds an instrumented binary at cmix_pgo_instrumented. Run it over a
# representative input (prof_input/input) to produce *.profraw files in
# pgo-raw/, then `make pgo-merge` to fold them into pgo/default.profdata.
# See docs/PGO_LTO.md.
pgo-instrumented:
	$(MAKE) clean
	rm -rf $(PGO_RAW_DIR)
	mkdir -p $(PGO_RAW_DIR)
	$(MAKE) cmix PGO=generate OUT=cmix_pgo_instrumented
	rm -f *.o

pgo-merge:
	test -n "$$(ls $(PGO_RAW_DIR)/*.profraw 2>/dev/null)"
	mkdir -p $(dir $(PGO_PROFILE))
	llvm-profdata-17 merge -output=$(PGO_PROFILE) $(PGO_RAW_DIR)/*.profraw

clean:
	rm -f *.o cmix cmix_orig cmix_pgo_instrumented
	rm -rf $(PGO_RAW_DIR)
