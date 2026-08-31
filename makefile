CC := clang++-17
CC_C := clang-17
OUT ?= cmix

# cmix-obias configuration of record, adapted to the FX4 configuration names.
# Set CMIX_OBIAS_RECORD=0 only for a control build.
CMIX_OBIAS_RECORD ?= 1
CFLAGS_DEFINES ?= -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG \
	-DFX4_STDERR_PROGRESS=0 -DFX4_PROGRESS_LOG=0 \
	-DFX4_LSTM_MID_BRIDGE=2
ifeq ($(CMIX_OBIAS_RECORD),1)
override CFLAGS_DEFINES += -DFX4_LSTM_CELLS=256 \
	-DFX4_LSTM_LEARNING_RATE=0.03f -DKH_BITLSTM32 -DKH_OBIAS \
	-DKH_OBIAS_CONST_GATE=0.15f -DKH_BITLSTM32_ARCHIVE
endif

DONOR ?= 0
POSTR1 ?= 0
MINI_CMIX ?= 0
VIRTUAL_REPLAY ?= 0
DONOR_DISCOVERY ?= 0
RESEARCH_DONOR_BOOTSTRAP ?= 0
POSTR1_TRANSFORM ?= 0
ORACLE_TRACE ?= 0
RESIDUAL_LSTM96 ?= 0
ALTXS ?= 0
TRANSFORMER ?= 0
TOKEN_NGRAM ?= 0

DONOR_SOURCE :=
DONOR_HEADER :=
DONOR_OBJECT :=
DONOR_DISCOVERY_SOURCE :=
DONOR_DISCOVERY_HEADER :=
DONOR_DISCOVERY_OBJECT :=
POSTR1_SOURCE :=
POSTR1_HEADER :=
POSTR1_OBJECT :=
ALTXS_CPP_SOURCE :=
ALTXS_CPP_OBJECT :=
ALTXS_C_OBJECTS :=
ALTXS_TARGET :=
TRANSFORMER_OBJECTS :=
TRANSFORMER_TARGET :=
TOKEN_NGRAM_SOURCE :=
TOKEN_NGRAM_OBJECT :=

ifeq ($(DONOR_DISCOVERY),1)
DONOR := 1
POSTR1 := 1
override CFLAGS_DEFINES += -DFX4_DONOR_FORK_DISCOVERY=1
DONOR_DISCOVERY_SOURCE := src/donor_fork_discovery.cpp src/donor_winner_search.cpp
DONOR_DISCOVERY_HEADER := src/donor_fork_discovery.h src/donor_winner_search.h
DONOR_DISCOVERY_OBJECT := donor_fork_discovery.o donor_winner_search.o
endif
ifeq ($(POSTR1),1)
override CFLAGS_DEFINES += -DFX4_SELECTIVE_POSTR1=1
POSTR1_SOURCE := src/models/postr1_experts.cpp
POSTR1_HEADER := src/models/postr1_experts.h
POSTR1_OBJECT := postr1_experts.o
endif
ifeq ($(MINI_CMIX),1)
override CFLAGS_DEFINES += -DFX4_MINI_CMIX=1
endif
ifeq ($(VIRTUAL_REPLAY),1)
override CFLAGS_DEFINES += -DFX4_VIRTUAL_REPLAY=1
endif
ifeq ($(RESEARCH_DONOR_BOOTSTRAP),1)
override CFLAGS_DEFINES += -DFX4_RESEARCH_DONOR_BOOTSTRAP=1
endif
ifeq ($(POSTR1_TRANSFORM),1)
override CFLAGS_DEFINES += -DFX4_POSTR1_TRANSFORM=1
endif
ifeq ($(ORACLE_TRACE),1)
override CFLAGS_DEFINES += -DFX4_RESIDUAL_ORACLE_TRACE=1
endif
ifeq ($(RESIDUAL_LSTM96),1)
override CFLAGS_DEFINES += -DFX4_RESIDUAL_LSTM96=1 \
	-DKH_RESIDUAL_LSTM96_ARCHIVE
endif
ifeq ($(ALTXS),1)
override CFLAGS_DEFINES += -DFX4_ALTXS_M3_M5=1
ALTXS_CPP_SOURCE := src/altxs_transform.cpp
ALTXS_CPP_OBJECT := altxs_transform.o
ALTXS_C_OBJECTS := m3_densify.o m5_payload_sim.o product_seal.o
ALTXS_TARGET := altxs_objects
endif
ifeq ($(TRANSFORMER),1)
override CFLAGS_DEFINES += -DFX4_TRANSFORMER6M=1 \
	-DFX4_TRANSFORMER6M_REQUIRED=1 -DKH_TRANSFORMER6M_ARCHIVE \
	-DKH_BITLSTM32_REQUIRED=1
TRANSFORMER_OBJECTS := tf_weights_io.o tf_weights_io_compressed.o \
	tf_qmat_dense.o tf_qmat_sparse.o tf_attn.o tf_kda.o tf_glue.o \
	tf_arena_build.o tf_model_opt.o
TRANSFORMER_TARGET := transformer_objects
endif
ifeq ($(TOKEN_NGRAM),1)
override CFLAGS_DEFINES += -DFX4_TOKEN_NGRAM_BIAS=1
TOKEN_NGRAM_SOURCE := src/models/token-ngram-bias.cpp
TOKEN_NGRAM_OBJECT := token-ngram-bias.o
endif
ifeq ($(DONOR),1)
override CFLAGS_DEFINES += -DFX4_DONOR_PLAN=1
DONOR_SOURCE := src/donor_plan.cpp
DONOR_HEADER := src/donor_plan.h
DONOR_OBJECT := donor_plan.o
endif

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
COMMON := $(CFLAGS_DEFINES) -m64 -Wall -std=c++17 -fno-exceptions \
	-fno-unwind-tables -fno-asynchronous-unwind-tables \
	-fno-threadsafe-statics -Wno-unknown-escape-sequence \
	-Wno-unused-variable -Wno-unneeded-internal-declaration \
	-Wno-unused-but-set-variable -Wno-format -march=native -mtune=native \
	-fdata-sections -ffunction-sections
FAST_FLAGS := $(COMMON) -O3 -ffp-model=fast
SLOW_FLAGS := $(COMMON) -Os -ffp-model=fast
COLD_FLAGS := $(COMMON) -Oz -ffp-model=fast
C_FLAGS := $(CFLAGS_DEFINES) -m64 -std=c11 -DNDEBUG -D_GNU_SOURCE -O3 \
	-march=native -mtune=native -fdata-sections -ffunction-sections
LFLAGS := -m64 -fuse-ld=lld -Wl,--gc-sections -std=c++17
TRANSFORMER_FLAGS := -m64 -O3 -std=c++17 -Wall -Wextra \
	-fno-math-errno -march=native -mtune=native \
	-fdata-sections -ffunction-sections

prof_gen: FAST_FLAGS += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: SLOW_FLAGS += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: COLD_FLAGS += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: LFLAGS += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: clean cmix

prof_use: FAST_FLAGS += -fprofile-use=$(ROOT_DIR)/pgo_data -flto
prof_use: SLOW_FLAGS += -fprofile-use=$(ROOT_DIR)/pgo_data
prof_use: COLD_FLAGS += -fprofile-use=$(ROOT_DIR)/pgo_data
prof_use: LFLAGS += -fprofile-use=$(ROOT_DIR)/pgo_data -flto
prof_use: clean cmix

slow: src/preprocess/preprocessor.cpp src/preprocess/preprocessor.h \
	src/preprocess/dictionary.cpp src/preprocess/dictionary.h
	$(CC) $(SLOW_FLAGS) src/preprocess/preprocessor.cpp \
		src/preprocess/dictionary.cpp -c

cold: $(DONOR_SOURCE) $(DONOR_HEADER) $(DONOR_DISCOVERY_SOURCE) \
	$(DONOR_DISCOVERY_HEADER) src/r1_reorder_transform.cpp \
	src/r1_reorder_transform.h src/scr2_transform.cpp src/scr2_transform.h \
	src/virtual_replay_plan.cpp src/virtual_replay_plan.h \
	src/postr1_transform.cpp src/postr1_transform.h $(ALTXS_CPP_SOURCE) \
	src/runner.cpp
	$(CC) $(COLD_FLAGS) $(DONOR_SOURCE) $(DONOR_DISCOVERY_SOURCE) \
		src/r1_reorder_transform.cpp src/scr2_transform.cpp \
		src/virtual_replay_plan.cpp src/postr1_transform.cpp \
		$(ALTXS_CPP_SOURCE) src/runner.cpp -c

altxs_objects: src/third_party/altxs/m3_densify.c \
	src/third_party/altxs/m5_payload_sim.c \
	src/third_party/altxs/product_seal.c
	$(CC_C) $(C_FLAGS) -Isrc/third_party/altxs \
		-c src/third_party/altxs/m3_densify.c \
		src/third_party/altxs/m5_payload_sim.c \
		src/third_party/altxs/product_seal.c

tf_weights_io.o: src/third_party/fx2_transformer/weights_io.cpp \
	src/third_party/fx2_transformer/weights_io.h
	$(CC) $(TRANSFORMER_FLAGS) \
		-c src/third_party/fx2_transformer/weights_io.cpp -o $@

tf_weights_io_compressed.o: \
	src/third_party/fx2_transformer/weights_io_compressed.cpp \
	src/third_party/fx2_transformer/weights_io.h
	$(CC) $(filter-out -O3,$(TRANSFORMER_FLAGS)) -Os \
		-c src/third_party/fx2_transformer/weights_io_compressed.cpp -o $@

tf_qmat_dense.o: src/third_party/fx2_transformer/opt/qmat_dense.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_qmat_sparse.o: src/third_party/fx2_transformer/opt/qmat_sparse.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_attn.o: src/third_party/fx2_transformer/opt/attn.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_kda.o: src/third_party/fx2_transformer/opt/kda.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_glue.o: src/third_party/fx2_transformer/opt/glue.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_arena_build.o: src/third_party/fx2_transformer/opt/arena_build.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@
tf_model_opt.o: src/third_party/fx2_transformer/opt/model_opt.cpp
	$(CC) $(TRANSFORMER_FLAGS) -c $< -o $@

transformer_objects: $(TRANSFORMER_OBJECTS)

fast:
	$(CC) $(FAST_FLAGS) src/coder/decoder.cpp src/coder/encoder.cpp \
		src/context-manager.cpp src/contexts/bit-context.cpp \
		src/contexts/bracket-context.cpp src/contexts/combined-context.cpp \
		src/contexts/context-hash.cpp src/contexts/indirect-hash.cpp \
		src/contexts/interval-hash.cpp src/contexts/interval.cpp \
		src/contexts/sparse.cpp src/models/bracket.cpp \
		src/models/byte-model.cpp src/models/direct-hash.cpp \
		src/models/direct.cpp src/models/match.cpp src/models/fxcmv1.cpp \
		$(POSTR1_SOURCE) $(TOKEN_NGRAM_SOURCE) src/models/ppmd.cpp \
		src/states/nonstationary.cpp \
		src/states/run-map.cpp src/mixer/byte-mixer.cpp \
		src/mixer/mixer-input.cpp src/mixer/mixer.cpp \
		src/mixer/sigmoid.cpp src/mixer/sse.cpp -c src/predictor.cpp

# These neural correction units must use the same precise floating-point path
# in encoder and decoder even though the legacy predictor remains fast-math.
head: src/models/bitlstm32-head.cpp src/models/bitlstm32-head.h
	$(CC) $(FAST_FLAGS) -ffp-model=precise -c src/models/bitlstm32-head.cpp

obias: src/models/obias-prior.cpp src/models/obias-prior.h
	$(CC) $(FAST_FLAGS) -ffp-model=precise -c src/models/obias-prior.cpp

residual96: src/models/residual-lstm96-head.cpp \
	src/models/residual-lstm96-head.h
	$(CC) $(FAST_FLAGS) -ffp-model=precise \
		-c src/models/residual-lstm96-head.cpp

cmix: fast slow cold head obias residual96 $(ALTXS_TARGET) \
	$(TRANSFORMER_TARGET)
	$(CC) $(LFLAGS) bit-context.o bracket-context.o bracket.o byte-mixer.o \
		byte-model.o combined-context.o context-hash.o context-manager.o \
		decoder.o dictionary.o direct-hash.o direct.o $(DONOR_OBJECT) \
		$(DONOR_DISCOVERY_OBJECT) encoder.o indirect-hash.o interval-hash.o \
		interval.o match.o mixer-input.o mixer.o nonstationary.o fxcmv1.o \
		$(POSTR1_OBJECT) $(TOKEN_NGRAM_OBJECT) ppmd.o predictor.o preprocessor.o \
		r1_reorder_transform.o scr2_transform.o virtual_replay_plan.o \
		postr1_transform.o $(ALTXS_CPP_OBJECT) $(ALTXS_C_OBJECTS) \
		run-map.o runner.o sigmoid.o sparse.o sse.o \
		bitlstm32-head.o obias-prior.o residual-lstm96-head.o \
		$(TRANSFORMER_OBJECTS) -s -o $(OUT)
	rm -f *.o

.PHONY: selective record altxs_record target93 transformer_objects clean
record: cmix

# Reversible altxs M3+M5 outer transform on top of the cmix-obias predictor.
# This is an experimental archive family and does not silently replace record.
altxs_record:
	$(MAKE) cmix ALTXS=1 OUT=$(OUT)

# Canonical CPU-only 93 MB research path. M3+M5 and the transformer are one
# archive family; the model is packaged into both S1 and S2 and counted.
target93:
	$(MAKE) cmix ALTXS=1 TRANSFORMER=1 TOKEN_NGRAM=$(TOKEN_NGRAM) OUT=$(OUT)

# Compile discovery support without applying any action globally. Donor,
# mini-cmix, SCR2/virtual-replay and post-R1 experts remain plan-gated.
selective:
	$(MAKE) cmix DONOR=1 POSTR1=1 MINI_CMIX=1 VIRTUAL_REPLAY=1 \
		DONOR_DISCOVERY=1 POSTR1_TRANSFORM=1 OUT=$(OUT)

clean:
	rm -f *.o cmix cmix_prof remap
