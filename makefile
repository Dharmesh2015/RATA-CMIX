CC = clang++-17
# The accepted release model is defined once in src/fx4_config.h.
CFLAGS_DEFINES ?= -DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -DFX4_STDERR_PROGRESS=0 -DFX4_PROGRESS_LOG=0

STRIP_FLAG := -s
OUT ?= cmix

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

cold: src/r1_reorder_transform.cpp src/r1_reorder_transform.h src/runner.cpp
	$(CC) $(CPPFLAGS_PART-THAT-IS-COLD) src/r1_reorder_transform.cpp src/runner.cpp -c

fast: src/coder/decoder.cpp src/coder/decoder.h src/coder/encoder.cpp src/coder/encoder.h src/context-manager.cpp src/context-manager.h src/contexts/bit-context.cpp src/contexts/bit-context.h src/contexts/bracket-context.cpp src/contexts/bracket-context.h src/contexts/combined-context.cpp src/contexts/combined-context.h src/contexts/context-hash.cpp src/contexts/context-hash.h src/contexts/context.h src/contexts/indirect-hash.cpp src/contexts/indirect-hash.h src/contexts/interval-hash.cpp src/contexts/interval-hash.h src/contexts/interval.cpp src/contexts/interval.h src/contexts/sparse.cpp src/contexts/sparse.h  src/models/bracket.cpp src/models/bracket.h src/models/byte-model.cpp src/models/byte-model.h src/models/direct-hash.cpp src/models/direct-hash.h src/models/direct.cpp src/models/direct.h src/models/indirect.h src/models/match.cpp src/models/match.h src/models/model.h src/models/fxcmv1.h src/models/ppmd.cpp src/models/ppmd.h src/states/nonstationary.cpp src/states/nonstationary.h src/states/run-map.cpp src/states/run-map.h src/states/state.h src/mixer/byte-mixer.cpp src/mixer/byte-mixer.h src/mixer/lstm-layer.h src/mixer/lstm.h src/mixer/mixer-input.cpp src/mixer/mixer-input.h src/mixer/mixer.cpp src/mixer/mixer.h src/mixer/sigmoid.cpp src/mixer/sigmoid.h src/mixer/sse.cpp src/mixer/sse.h src/predictor.h src/predictor.cpp
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) src/coder/decoder.cpp src/coder/encoder.cpp src/context-manager.cpp src/contexts/bit-context.cpp src/contexts/bracket-context.cpp src/contexts/combined-context.cpp src/contexts/context-hash.cpp src/contexts/indirect-hash.cpp src/contexts/interval-hash.cpp src/contexts/interval.cpp src/contexts/sparse.cpp src/models/bracket.cpp src/models/byte-model.cpp src/models/direct-hash.cpp src/models/direct.cpp src/models/match.cpp src/models/fxcmv1.cpp src/models/ppmd.cpp src/states/nonstationary.cpp src/states/run-map.cpp src/mixer/byte-mixer.cpp src/mixer/mixer-input.cpp src/mixer/mixer.cpp src/mixer/sigmoid.cpp src/mixer/sse.cpp -c src/predictor.cpp

cmix: fast slow cold
	$(CC) $(LFLAGS) bit-context.o bracket-context.o bracket.o byte-mixer.o byte-model.o combined-context.o context-hash.o context-manager.o decoder.o dictionary.o direct-hash.o direct.o encoder.o indirect-hash.o interval-hash.o interval.o match.o mixer-input.o mixer.o nonstationary.o fxcmv1.o ppmd.o predictor.o preprocessor.o r1_reorder_transform.o run-map.o runner.o sigmoid.o sparse.o sse.o $(STRIP_FLAG) -o $(OUT)
	rm -f *.o

remap: src/readalike_prepr/article_remap.cpp
	$(CC) src/readalike_prepr/article_remap.cpp -o remap

clean:
	rm -f *.o
	rm -f cmix
	rm -f remap

all: cmix remap
