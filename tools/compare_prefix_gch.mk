# 1% benchmark build for release/google-cloud-hutter and its descendants.
#
# Builds tools/compare_prefix.cpp against this branch's model objects and links
# a prefix_bench that codes a prefix of the post-WRT stream and reports the
# archive size. It reuses the production makefile's own fast/slow/cold/
# transformer targets, so what it measures is what target93 compiles -- not a
# parallel build that could drift from it.
#
# THREE DELIBERATE DIFFERENCES FROM target93, all so the number is comparable
# to fx2-cmix-transformer-v521's completed 673,799:
#
#   no LTO       -flto=thin measured NOT bit-exact on this project. A run with
#                it is measuring a different archive.
#   no PGO       the committed profile was trained on the pre-port code and
#                currently loses 40 functions to hash mismatches, including
#                Predictor::Predict and Mixer::Mix. Leaving it out keeps the
#                bench from depending on how stale the profile happens to be.
#   fixed ISA    -march=x86-64-v3 -mtune=generic, the baseline every v521
#                measurement used.
#
# BENCH_LTO=1 and BENCH_PGO=1 put them back, for measuring the shipped
# configuration rather than a comparable one.
include makefile

BENCH_ARCH ?= -march=x86-64-v3 -mtune=generic

# BENCH_F16=0 forces the LSTM's fp32 dot products. The fp16 shadow path
# is not bit-exact -- half-precision cannot reproduce an fp32 dot
# product -- so this is how its cost in archive bytes gets measured
# instead of assumed.
ifeq ($(BENCH_F16),0)
BENCH_EXTRA += -DSIMD_ACT_DISABLE_F16
endif

# The production flags are computed with := before this file is read, so they
# have to be filtered rather than redefined.
BENCH_STRIP := -flto=thin -Wno-profile-instr-out-of-date
ifneq ($(BENCH_LTO),1)
BENCH_DROP += -flto=thin
endif
ifneq ($(BENCH_PGO),1)
BENCH_DROP += -fprofile-use=$(PGO_PROFILE) -Wno-profile-instr-out-of-date
endif
BENCH_DROP += -march=x86-64-v3 -mtune=generic

clean_flags = $(filter-out $(BENCH_DROP),$(1)) $(BENCH_ARCH)

FAST_FLAGS := $(call clean_flags,$(FAST_FLAGS)) $(BENCH_EXTRA)
SLOW_FLAGS := $(call clean_flags,$(SLOW_FLAGS)) $(BENCH_EXTRA)
COLD_FLAGS := $(call clean_flags,$(COLD_FLAGS)) $(BENCH_EXTRA)
TRANSFORMER_FLAGS := $(call clean_flags,$(TRANSFORMER_FLAGS))
LDFLAGS := $(filter-out $(BENCH_DROP),$(LDFLAGS))

.PHONY: prefix-bench
prefix-bench: fast slow cold transformer_objects
	$(CXX) $(FAST_FLAGS) -DPREFIX_FX2 -Isrc -c tools/compare_prefix.cpp \
		-o prefix_driver.o
	$(CXX) $(LDFLAGS) prefix_driver.o \
		$(filter-out runner.o prefix_driver.o,$(wildcard *.o)) -o prefix_bench
