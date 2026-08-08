CC = clang++-17
# The accepted release model is defined once in src/fx4_config.h.
CFLAGS_DEFINES ?= -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -DFX4_STDERR_PROGRESS=0 -DFX4_PROGRESS_LOG=0 -DFX4_LSTM_MID_BRIDGE=2

STRIP_FLAG := -s
OUT ?= cmix
PAGE_INDEX_OUT ?= postr1_page_index
R1_MAP_OUT ?= emit_r1_map
PAGE_DONOR_RANKER_OUT ?= postr1_page_donor_ranker
DONOR ?= 0
DONOR_DISCOVERY ?= 0
DONOR_DISCOVERY_SOURCE :=
DONOR_DISCOVERY_HEADER :=
DONOR_DISCOVERY_OBJECT :=
POSTR1_SOURCE :=
POSTR1_HEADER :=
POSTR1_OBJECT :=
ifeq ($(DONOR_DISCOVERY),1)
DONOR := 1
CFLAGS_DEFINES += -DFX4_DONOR_FORK_DISCOVERY=1 -DFX4_SELECTIVE_POSTR1=1
DONOR_DISCOVERY_SOURCE := src/donor_fork_discovery.cpp src/donor_winner_search.cpp
DONOR_DISCOVERY_HEADER := src/donor_fork_discovery.h src/donor_winner_search.h
DONOR_DISCOVERY_OBJECT := donor_fork_discovery.o donor_winner_search.o
POSTR1_SOURCE := src/models/postr1_experts.cpp
POSTR1_HEADER := src/models/postr1_experts.h
POSTR1_OBJECT := postr1_experts.o
endif
DONOR_SOURCE :=
DONOR_HEADER :=
DONOR_OBJECT :=
ifeq ($(DONOR),1)
CFLAGS_DEFINES += -DFX4_DONOR_PLAN=1
DONOR_SOURCE := src/donor_plan.cpp
DONOR_HEADER := src/donor_plan.h
DONOR_OBJECT := donor_plan.o
endif

CPPFLAGS_PART-THAT-SHOULD-BE-FAST := $(CFLAGS_DEFINES) -m64 -Wall -std=c++17 -ffp-model=fast -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-threadsafe-statics -Wno-unknown-escape-sequence -Wno-unused-variable -Wno-unneeded-internal-declaration -Wno-unused-but-set-variable -Wno-format 

ifdef COREI7
$(info COREI7 defined)
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=corei7
else
ifdef ZEN2
$(info ZEN2 defined)
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=znver2
else
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=native -mtune=native
$(info native used)
endif
endif

CPPFLAGS_PART-THAT-CAN-BE-SLOW    := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -Os -fdata-sections -ffunction-sections
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -O3 -fdata-sections -ffunction-sections
CPPFLAGS_PART-THAT-IS-COLD        := $(CFLAGS_DEFINES) -m64 -Wall -std=c++17 -ffp-model=fast -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-threadsafe-statics -Wno-unknown-escape-sequence -Wno-unused-variable -Wno-unneeded-internal-declaration -Wno-unused-but-set-variable -Wno-format -Oz -fdata-sections -ffunction-sections

ifdef COREI7
CPPFLAGS_PART-THAT-IS-COLD += -march=corei7
else
ifdef ZEN2
CPPFLAGS_PART-THAT-IS-COLD += -march=znver2
else
CPPFLAGS_PART-THAT-IS-COLD += -march=native -mtune=native
endif
endif

LFLAGS := -m64 -Wl,--gc-sections -std=c++17


slow: src/preprocess/preprocessor.cpp src/preprocess/preprocessor.h src/preprocess/dictionary.cpp src/preprocess/dictionary.h
	$(CC) $(CPPFLAGS_PART-THAT-CAN-BE-SLOW) src/preprocess/preprocessor.cpp src/preprocess/dictionary.cpp -c 

cold: $(DONOR_SOURCE) $(DONOR_HEADER) $(DONOR_DISCOVERY_SOURCE) $(DONOR_DISCOVERY_HEADER) src/r1_reorder_transform.cpp src/r1_reorder_transform.h src/scr2_transform.cpp src/scr2_transform.h src/models/scr2_tokens.h src/virtual_replay_plan.cpp src/virtual_replay_plan.h src/postr1_transform.cpp src/postr1_transform.h src/runner.cpp
	$(CC) $(CPPFLAGS_PART-THAT-IS-COLD) $(DONOR_SOURCE) $(DONOR_DISCOVERY_SOURCE) src/r1_reorder_transform.cpp src/scr2_transform.cpp src/virtual_replay_plan.cpp src/postr1_transform.cpp src/runner.cpp -c

fast: src/coder/decoder.cpp src/coder/decoder.h src/coder/encoder.cpp src/coder/encoder.h src/context-manager.cpp src/context-manager.h src/contexts/bit-context.cpp src/contexts/bit-context.h src/contexts/bracket-context.cpp src/contexts/bracket-context.h src/contexts/combined-context.cpp src/contexts/combined-context.h src/contexts/context-hash.cpp src/contexts/context-hash.h src/contexts/context.h src/contexts/indirect-hash.cpp src/contexts/indirect-hash.h src/contexts/interval-hash.cpp src/contexts/interval-hash.h src/contexts/interval.cpp src/contexts/interval.h src/contexts/sparse.cpp src/contexts/sparse.h  src/models/bracket.cpp src/models/bracket.h src/models/byte-model.cpp src/models/byte-model.h src/models/direct-hash.cpp src/models/direct-hash.h src/models/direct.cpp src/models/direct.h src/models/indirect.h src/models/match.cpp src/models/match.h src/models/model.h src/models/fxcmv1.h $(POSTR1_SOURCE) $(POSTR1_HEADER) src/models/scr2_tokens.h src/models/ppmd.cpp src/models/ppmd.h src/states/nonstationary.cpp src/states/nonstationary.h src/states/run-map.cpp src/states/run-map.h src/states/state.h src/mixer/byte-mixer.cpp src/mixer/byte-mixer.h src/mixer/lstm-layer.h src/mixer/lstm.h src/mixer/mixer-input.cpp src/mixer/mixer-input.h src/mixer/mixer.cpp src/mixer/mixer.h src/mixer/sigmoid.cpp src/mixer/sigmoid.h src/mixer/sse.cpp src/mixer/sse.h src/predictor.h src/predictor.cpp
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) src/coder/decoder.cpp src/coder/encoder.cpp src/context-manager.cpp src/contexts/bit-context.cpp src/contexts/bracket-context.cpp src/contexts/combined-context.cpp src/contexts/context-hash.cpp src/contexts/indirect-hash.cpp src/contexts/interval-hash.cpp src/contexts/interval.cpp src/contexts/sparse.cpp src/models/bracket.cpp src/models/byte-model.cpp src/models/direct-hash.cpp src/models/direct.cpp src/models/match.cpp src/models/fxcmv1.cpp $(POSTR1_SOURCE) src/models/ppmd.cpp src/states/nonstationary.cpp src/states/run-map.cpp src/mixer/byte-mixer.cpp src/mixer/mixer-input.cpp src/mixer/mixer.cpp src/mixer/sigmoid.cpp src/mixer/sse.cpp -c src/predictor.cpp

cmix: fast slow cold
	$(CC) $(LFLAGS) bit-context.o bracket-context.o bracket.o byte-mixer.o byte-model.o combined-context.o context-hash.o context-manager.o decoder.o dictionary.o direct-hash.o direct.o $(DONOR_OBJECT) $(DONOR_DISCOVERY_OBJECT) encoder.o indirect-hash.o interval-hash.o interval.o match.o mixer-input.o mixer.o nonstationary.o fxcmv1.o $(POSTR1_OBJECT) ppmd.o predictor.o preprocessor.o r1_reorder_transform.o scr2_transform.o virtual_replay_plan.o postr1_transform.o run-map.o runner.o sigmoid.o sparse.o sse.o $(STRIP_FLAG) -o $(OUT)
	rm -f *.o

remap: src/readalike_prepr/article_remap.cpp
	$(CC) src/readalike_prepr/article_remap.cpp -o remap

page_index: tools/postr1_page_index.cpp src/preprocess/dictionary.cpp src/preprocess/dictionary.h
	$(CC) -m64 -std=c++17 -O3 -march=native -DNDEBUG \
		tools/postr1_page_index.cpp src/preprocess/dictionary.cpp \
		-o $(PAGE_INDEX_OUT)

r1_map: tools/emit_r1_map.cpp src/r1_reorder_transform.cpp src/r1_reorder_transform.h
	$(CC) -m64 -std=c++17 -O3 -march=native -DNDEBUG \
		tools/emit_r1_map.cpp src/r1_reorder_transform.cpp \
		-o $(R1_MAP_OUT)

page_donor_ranker: tools/rank_page_donors.cpp
	$(CC) -m64 -std=c++17 -O3 -march=native -DNDEBUG \
		tools/rank_page_donors.cpp -o $(PAGE_DONOR_RANKER_OUT)

clean:
	rm -f *.o
	rm -f cmix
	rm -f remap
	rm -f $(PAGE_INDEX_OUT)
	rm -f $(R1_MAP_OUT)
	rm -f $(PAGE_DONOR_RANKER_OUT)

all: cmix remap
