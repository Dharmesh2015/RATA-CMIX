#include <fstream>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <malloc.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"
#include "donor_plan.h"
#include "donor_fork_discovery.h"
#include "virtual_replay_plan.h"
#include "postr1_transform.h"
#include "altxs_transform.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/self_extract.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/misc.h"
#include "r1_reorder_transform.h"
#include "scr2_transform.h"
#include "fx4_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
const int kMinVocabFileSize = 10000;

#if FX4_RESIDUAL_ORACLE_TRACE
bool oracle_trace_stopped = false;
#endif

#if FX4_ALTXS_M3_M5
altxs::ProductMeta altxs_decode_meta;
bool altxs_decode_used = false;
constexpr const char* kAltxsM3Side = ".altxs_m3_side";
constexpr const char* kAltxsM3SideDecomp = ".altxs_m3_side_decomp";
#endif

#if FX4_RESEARCH_DONOR_BOOTSTRAP
std::vector<std::uint8_t> LoadResearchDonorBootstrap() {
  const char* path = std::getenv("FX4_RESEARCH_DONOR_BOOTSTRAP");
  if (!path || !*path) return {};
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    fprintf(stderr, "cannot open FX4_RESEARCH_DONOR_BOOTSTRAP: %s\n", path);
    std::exit(2);
  }
  input.seekg(0, std::ios::end);
  const std::streamoff length = input.tellg();
  input.seekg(0, std::ios::beg);
  if (length <= 0 || length > (1 << 20)) {
    fprintf(stderr, "research donor bootstrap must be 1..1048576 bytes\n");
    std::exit(2);
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  input.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!input) {
    fprintf(stderr, "cannot read complete research donor bootstrap: %s\n",
        path);
    std::exit(2);
  }
  return bytes;
}

void AddResearchDonorVocabulary(const std::vector<std::uint8_t>& bytes,
    std::vector<bool>* vocab) {
  for (std::uint8_t byte : bytes) (*vocab)[byte] = true;
}

void ReplayResearchDonor(const std::vector<std::uint8_t>& bytes,
    Predictor* predictor) {
  for (std::uint8_t byte : bytes) {
    for (int bit = 7; bit >= 0; --bit) {
      predictor->Predict();
      predictor->Perceive((byte >> bit) & 1);
    }
  }
}
#endif

#if FX4_RESEARCH_STREAM_DUMP || FX4_VIRTUAL_REPLAY || FX4_DONOR_PLAN
bool CopyResearchStream(const std::string& source, const char* destination) {
  if (!destination || !*destination) return true;
  if (source == destination) return false;
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(destination,
      std::ios::binary | std::ios::out | std::ios::trunc);
  if (!input.is_open() || !output.is_open()) return false;
  std::vector<char> buffer(1u << 20);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) output.write(buffer.data(), count);
  }
  return input.eof() && output.good();
}
#endif

#if FX4_RESEARCH_STREAM_DUMP || FX4_SCR2 || FX4_DONOR_PLAN || \
    FX4_POSTR1_TRANSFORM || FX4_RESEARCH_DONOR_BOOTSTRAP
bool EnvironmentEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
}

#if FX4_RESEARCH_DONOR_BOOTSTRAP
std::vector<std::uint32_t> ResearchDonorSegments(std::size_t total) {
  const char* specification =
      std::getenv("FX4_RESEARCH_DONOR_SEGMENTS");
  if (!specification || !*specification) {
    return {static_cast<std::uint32_t>(total)};
  }
  std::vector<std::uint32_t> segments;
  std::size_t sum = 0;
  const char* cursor = specification;
  while (*cursor) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(cursor, &end, 10);
    if (end == cursor || value == 0 || value > total ||
        (*end != '\0' && *end != ',')) {
      std::fprintf(stderr, "invalid FX4_RESEARCH_DONOR_SEGMENTS\n");
      std::exit(2);
    }
    segments.push_back(static_cast<std::uint32_t>(value));
    sum += value;
    cursor = *end == ',' ? end + 1 : end;
  }
  if (sum != total) {
    std::fprintf(stderr,
        "FX4_RESEARCH_DONOR_SEGMENTS total does not match donor bytes\n");
    std::exit(2);
  }
  return segments;
}

void ConfigureResearchDonorProfile(
    const std::vector<std::uint8_t>& bytes, Predictor* predictor) {
  if (!EnvironmentEnabled("FX4_RESEARCH_DONOR_PROFILE")) return;
  predictor->SetPostR1DonorProfile(bytes, ResearchDonorSegments(bytes.size()));
}
#endif
#endif
}

int Help() {
  printf("cmix-lex\n");
  printf("Compress:\n");
  printf("    to compress enwik9: cmix -e enwik9 [output]\n");
  printf("    to create a header for hutter prize: cmix -h comp_dict_size comp_new_order_size decomp_input_size [transformer_size] [bitlstm32_size]\n");
  printf("    with dictionary:    cmix -c [dictionary] [input] [output]\n");
    printf("    without dictionary: cmix -c [input] [output]\n");
    printf("    no preprocessing:   cmix -n [input] [output]\n");
    printf("    only preprocessing: cmix -s [dictionary] [input] [output]\n");
    printf("                        cmix -s [input] [output]\n");
  printf("Decompress:\n");
    printf("    with dictionary:    cmix -d [dictionary] [input] [output]\n");
    printf("    without dictionary: cmix -d [input] [output]\n");
  return -1;
}

size_t getFileSize(const std::string& path) {
  // // get the size of the output file
  FILE *f = fopen(path.c_str(), "rb");
  if (f == NULL) {
    printf("can't open file for measuring its size");
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteHeader(unsigned long long length, const std::vector<bool>& vocab,
    bool dictionary_used, bool donor_plan_used, bool scr2_used,
    bool postr1_transform_used, bool virtual_replay_used, bool altxs_used,
    std::ofstream* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x0F;
      if (dictionary_used) c |= 0x80;
      if (donor_plan_used) c |= 0x40;
      if (scr2_used || postr1_transform_used || altxs_used) c |= 0x20;
      if (virtual_replay_used) c |= 0x10;
    }
    os->put(c);
  }
  if (scr2_used) {
    os->put(static_cast<char>(scr2::kArchiveVersion));
  } else if (postr1_transform_used) {
    os->put(static_cast<char>(postr1::kArchiveVersion));
  } else if (altxs_used) {
    os->put(static_cast<char>(altxs::kArchiveVersion));
  }
  if (length < kMinVocabFileSize) return;
  for (int i = 0; i < 32; ++i) {
    unsigned char c = 0;
    for (int j = 0; j < 8; ++j) {
      if (vocab[i * 8 + j]) c += 1<<j;
    }
    os->put(c);
  }
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

bool ReadHeader(std::ifstream* is, unsigned long long* length,
    bool* dictionary_used, bool* donor_plan_used, bool* scr2_used,
    bool* postr1_transform_used, bool* virtual_replay_used, bool* altxs_used,
    std::vector<bool>* vocab) {
  *length = 0;
  *scr2_used = false;
  *postr1_transform_used = false;
  *altxs_used = false;
  bool transform_used = false;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    if (i == 0) {
      *dictionary_used = (c & 0x80) != 0;
      *donor_plan_used = (c & 0x40) != 0;
      transform_used = (c & 0x20) != 0;
      *virtual_replay_used = (c & 0x10) != 0;
      c &= 0x0F;
    }
    *length += c;
  }
  if (transform_used) {
    const int version = is->get();
    if (version == scr2::kArchiveVersion) {
#if FX4_SCR2
      *scr2_used = true;
#else
      return false;
#endif
    } else if (version == postr1::kArchiveVersion) {
#if FX4_POSTR1_TRANSFORM
      *postr1_transform_used = true;
#else
      return false;
#endif
    } else if (version == altxs::kArchiveVersion) {
#if FX4_ALTXS_M3_M5
      *altxs_used = true;
#else
      return false;
#endif
    } else {
      return false;
    }
  }
  if (*length == 0) return true;
  if (*length < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
    return true;
  }
  for (int i = 0; i < 32; ++i) {
    unsigned char c = is->get();
    for (int j = 0; j < 8; ++j) {
      if (c & (1<<j)) (*vocab)[i * 8 + j] = true;
    }
  }
  return is->good();
}

void ExtractVocab(unsigned long long num_bytes, std::ifstream* is,
    std::vector<bool>* vocab) {
  for (size_t pos = 0; pos < num_bytes; ++pos) {
    unsigned char c = is->get();
    (*vocab)[c] = true;
  }
  assert(num_bytes >= 2);
  std::valarray<int> byte_map(0, 256);
  uint16_t offset = 0;
  for (int i = 0; i < 256; ++i) {
    byte_map[i] = offset;
    if ((*vocab)[i]) ++offset;
  }
}

void ClearOutput() {
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\r                     \r");
  fflush(stderr);
#endif
}

class CausalScr2RangeCursor {
 public:
  CausalScr2RangeCursor(const VirtualReplayPlan* plan,
      std::uint64_t stream_size)
      : plan_(plan), stream_size_(stream_size) {}

  bool Active(std::uint64_t position, std::uint64_t* begin,
      std::uint64_t* end, bool* entered) {
    *entered = false;
    if (!plan_ || !plan_->causal_scr2()) return false;
    if (plan_->causal_all()) {
      *begin = 0;
      *end = stream_size_;
      if (!all_entered_) {
        all_entered_ = true;
        *entered = true;
      }
      return position < stream_size_;
    }
    const auto& ranges = plan_->causal_ranges();
    while (range_index_ < ranges.size() &&
        position >= ranges[range_index_].offset +
            ranges[range_index_].length) {
      ++range_index_;
    }
    if (range_index_ >= ranges.size() ||
        position < ranges[range_index_].offset) {
      return false;
    }
    *begin = ranges[range_index_].offset;
    *end = *begin + ranges[range_index_].length;
    if (entered_range_ != range_index_) {
      entered_range_ = range_index_;
      *entered = true;
    }
    return position < *end;
  }

  std::size_t CurrentRange() const {
    return plan_ && !plan_->causal_all() ? range_index_ : 0;
  }

 private:
  const VirtualReplayPlan* plan_;
  std::uint64_t stream_size_;
  std::size_t range_index_ = 0;
  std::size_t entered_range_ = std::numeric_limits<std::size_t>::max();
  bool all_entered_ = false;
};

std::array<std::uint8_t, 16> CausalScr2PatternMask(
    const VirtualReplayPlan* plan) {
  std::array<std::uint8_t, 16> mask{};
  mask.fill(0xffu);
  if (plan && plan->causal_scr2()) mask = plan->causal_pattern_mask();
  return mask;
}

struct CausalScr2PatternStats {
  std::uint64_t triggers = 0;
  std::uint64_t takes = 0;
  std::uint64_t replayed_bytes = 0;
  std::uint16_t prefix_length = 0;
  std::uint16_t suffix_length = 0;
  double estimated_gain_bits = 0.0;
};

bool Compress(unsigned long long input_bytes, std::ifstream* is,
    std::ofstream* os, unsigned long long* output_bytes, Predictor* p,
    DonorPlan* donor_plan, VirtualReplayPlan* replay_plan) {
  Encoder e(os, p);
#if FX4_RESIDUAL_ORACLE_TRACE
  oracle_trace_stopped = false;
  if (!e.StartOracleTrace(input_bytes)) {
    fprintf(stderr, "cannot create FX4 residual oracle trace\n");
    return false;
  }
  const char* stop_value = std::getenv("FX4_STOP_AFTER_ORACLE_TRACE");
  const bool stop_after_oracle_trace =
      stop_value && *stop_value && std::strcmp(stop_value, "0") != 0;

#endif
#if FX4_VIRTUAL_REPLAY
  const char* trace_path = std::getenv("FX4_VR_COST_TRACE");
  if (replay_plan && trace_path && *trace_path) {
    fprintf(stderr, "FX4_VR_COST_TRACE cannot be combined with FX4_VR_PLAN\n");
    return false;
  }
  if (!e.StartCostTrace(trace_path, input_bytes)) {
    fprintf(stderr, "cannot create FX4 virtual-replay cost trace\n");
    return false;
  }
#endif
#if FX4_PROGRESS_LOG
  FILE* progress = fopen("./progress.log", "w");
#endif
  const unsigned long long progress_step =
      1 + (input_bytes / FX4_PROGRESS_STEPS);
  unsigned long long next_progress = 0;
  ClearOutput();
  std::vector<char> buffer(FX4_IO_BUFFER_BYTES);
  unsigned long long pos = 0;
  std::size_t replay_index = 0;
  CausalScr2Matcher causal_scr2(
      replay_plan && replay_plan->causal_scr2()
          ? replay_plan->causal_prior_code() : 0,
      CausalScr2PatternMask(replay_plan), input_bytes);
  CausalScr2RangeCursor causal_ranges(replay_plan, input_bytes);
  std::uint64_t causal_triggers = 0;
  std::uint64_t causal_takes = 0;
  std::uint64_t causal_replayed = 0;
  const char* causal_stats_path = std::getenv("FX4_CAUSAL_SCR2_STATS");
  const bool collect_causal_stats =
      causal_stats_path && *causal_stats_path;
  const std::size_t causal_range_count =
      replay_plan && replay_plan->causal_scr2() &&
          !replay_plan->causal_all()
      ? replay_plan->causal_ranges().size() : 1;
  std::vector<std::array<CausalScr2PatternStats, 129>> causal_stats(
      collect_causal_stats ? causal_range_count : 0);
#if FX4_DONOR_PLAN
  const char* stats_path = std::getenv("FX4_REGION_STATS");
  std::ofstream stats;
  if (stats_path && *stats_path) {
    stats.open(stats_path, std::ios::out | std::ios::trunc);
    if (!stats.is_open()) return false;
    stats << "region,donor_seed_offset,cumulative_before,cumulative_after,"
             "payload_bytes\n";
  }
  size_t region_start = e.OutputSize();
  uint32_t current_region = 0;
  const char* stop_text = std::getenv("FX4_DONOR_STOP_AFTER_REGIONS");
  const uint32_t stop_after_regions = stop_text && *stop_text
      ? static_cast<uint32_t>(std::strtoul(stop_text, nullptr, 10))
      : 0;
#endif

  auto before_byte = [&](unsigned long long logical_pos) -> bool {
#if FX4_DONOR_PLAN
    if (logical_pos != 0 && logical_pos % DonorPlan::kChunkSize == 0) {
      const size_t region_end = e.OutputSize();
      if (stats.is_open()) {
        stats << current_region << ','
              << (donor_plan
                  ? donor_plan->DonorOffset(current_region)
                  : DonorPlan::kNoDonor)
              << ',' << region_start << ',' << region_end << ','
              << (region_end - region_start) << '\n';
      }
      ++current_region;
      region_start = region_end;
      if (stop_after_regions != 0 && current_region >= stop_after_regions) {
        if (stats.is_open()) stats.flush();
        fprintf(stderr, "FX4 donor staged run stopped after %u regions\n",
            current_region);
        return false;
      }
    }
    if (donor_plan && !donor_plan->ReplayAt(logical_pos, p)) return false;
#else
    (void)logical_pos;
#endif
    return true;
  };

  auto after_byte = [&](unsigned long long logical_pos,
      std::uint8_t value) {
#if FX4_DONOR_PLAN
    if (donor_plan) donor_plan->CaptureByte(logical_pos, value);
#else
    (void)logical_pos;
    (void)value;
#endif
  };

  auto report_progress = [&](unsigned long long logical_pos) {
    if (logical_pos < next_progress) return;
    const double frac = 100.0 * logical_pos / input_bytes;
#if FX4_STDERR_PROGRESS
    fprintf(stderr, "\rprogress: %.2f%%", frac);
    fflush(stderr);
#endif
#if FX4_PROGRESS_LOG
    if (progress) fprintf(progress, "%.2f %zu\n", frac, e.OutputSize());
#endif
    do {
      next_progress += progress_step;
    } while (logical_pos >= next_progress);
  };

  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer.size(), input_bytes - pos));
    is->read(buffer.data(), chunk);
    const size_t got = static_cast<size_t>(is->gcount());
    if (got == 0) break;
    size_t i = 0;
    while (i < got) {
      std::uint64_t causal_begin = 0;
      std::uint64_t causal_end = 0;
      bool causal_entered = false;
      const bool causal_active = causal_ranges.Active(
          pos, &causal_begin, &causal_end, &causal_entered);
      if (causal_entered) causal_scr2.ResetModel();
      if (causal_active) {
        CausalScr2Matcher::Candidate candidate;
        if (causal_scr2.FindCandidate(&candidate) &&
            pos >= causal_begin + candidate.prefix_length &&
            candidate.suffix_length <= causal_end - pos &&
            candidate.suffix_length <= input_bytes - pos &&
            pos / FX4_IO_BUFFER_BYTES ==
                (pos + candidate.suffix_length - 1) /
                    FX4_IO_BUFFER_BYTES &&
            candidate.suffix_length <= got - i) {
          const std::uint8_t* suffix = causal_scr2.PatternData(
              candidate.pattern) + candidate.prefix_length;
          const std::uint16_t ppmd_context = causal_scr2.PpmdContext(
              p->PpmdByteProbability(suffix[0]), p->PpmdEffectiveOrder(),
              p->PpmdEscapeDepth(), p->PpmdEscapeRate());
          const bool take = std::memcmp(
              buffer.data() + i, suffix, candidate.suffix_length) == 0;
          const unsigned int shortcut_probability =
              causal_scr2.Probability(candidate.pattern, candidate.context,
                  candidate.class_context, ppmd_context);
          e.EncodeRawBit(take, shortcut_probability);
          causal_scr2.Update(candidate.pattern, candidate.context,
              candidate.class_context, ppmd_context, take);
          ++causal_triggers;
          CausalScr2PatternStats* pattern_stats = nullptr;
          if (collect_causal_stats) {
            pattern_stats = &causal_stats[causal_ranges.CurrentRange()]
                [candidate.pattern];
            ++pattern_stats->triggers;
            pattern_stats->prefix_length = candidate.prefix_length;
            pattern_stats->suffix_length = candidate.suffix_length;
          }
          if (collect_causal_stats) {
            const double mass = take ? shortcut_probability :
                65536u - shortcut_probability;
            pattern_stats->estimated_gain_bits -=
                -std::log2(mass / 65536.0);
          }
          if (take) {
            ++causal_takes;
            causal_replayed += candidate.suffix_length;
            if (pattern_stats) {
              ++pattern_stats->takes;
              pattern_stats->replayed_bytes += candidate.suffix_length;
            }
            for (std::uint16_t k = 0; k < candidate.suffix_length; ++k) {
              const std::uint8_t value = suffix[k];
              if (!before_byte(pos)) return false;
              if (collect_causal_stats) {
                pattern_stats->estimated_gain_bits +=
                    e.ObserveKnownByteCost(value);
              } else {
                e.ObserveKnownByte(value);
              }
              causal_scr2.ObserveByte(value);
              after_byte(pos, value);
              report_progress(pos);
              ++pos;
              ++i;
            }
            continue;
          }
        }
      }

      if (replay_plan && !replay_plan->causal_scr2() &&
          replay_index < replay_plan->event_count()) {
        const VirtualReplayPlan::Event& event =
            replay_plan->event(replay_index);
        if (event.offset < pos) return false;
        if (event.offset == pos) {
          const auto& pattern = replay_plan->pattern(event.pattern);
          if (i + pattern.size() > got) return false;
          for (std::size_t k = 0; k < pattern.size(); ++k) {
            if (static_cast<std::uint8_t>(buffer[i + k]) != pattern[k]) {
              fprintf(stderr,
                  "FX4 virtual-replay input mismatch at byte %llu\n",
                  static_cast<unsigned long long>(pos + k));
              return false;
            }
          }
          for (std::uint8_t value : pattern) {
            if (!before_byte(pos)) return false;
            e.ObserveKnownByte(value);
            after_byte(pos, value);
            report_progress(pos);
            ++pos;
            ++i;
          }
          ++replay_index;
          continue;
        }
      }

      const std::uint8_t value = static_cast<std::uint8_t>(buffer[i]);
      if (!before_byte(pos)) return false;
      e.BeginTraceByte(pos, value, 0);
      for (int bit = 7; bit >= 0; --bit) e.Encode((value >> bit) & 1);
      e.EndTraceByte();
      if (replay_plan && replay_plan->causal_scr2()) {
        causal_scr2.ObserveByte(value);
      }
      after_byte(pos, value);
      report_progress(pos);
      ++pos;
      ++i;
#if FX4_RESIDUAL_ORACLE_TRACE
      if (stop_after_oracle_trace && e.OracleTraceComplete()) {
        oracle_trace_stopped = true;
        break;
      }
#endif

    }
#if FX4_RESIDUAL_ORACLE_TRACE
    if (oracle_trace_stopped) break;
#endif

  }
  if (replay_plan && !replay_plan->causal_scr2() &&
      replay_index != replay_plan->event_count()) return false;
  e.Flush();
  if (replay_plan && replay_plan->causal_scr2()) {
    fprintf(stderr,
        "causal SCR2: triggers=%llu takes=%llu replayed=%llu bytes\n",
        static_cast<unsigned long long>(causal_triggers),
        static_cast<unsigned long long>(causal_takes),
        static_cast<unsigned long long>(causal_replayed));
    if (collect_causal_stats) {
      std::ofstream stats(causal_stats_path,
          std::ios::out | std::ios::trunc);
      if (!stats.is_open()) return false;
      stats << "range,offset,length,pattern,prefix_length,suffix_length,"
               "triggers,takes,"
               "replayed_bytes,estimated_gain_bits,estimated_gain_bytes\n";
      for (std::size_t range = 0; range < causal_stats.size(); ++range) {
        const std::uint64_t range_offset = replay_plan->causal_all()
            ? 0 : replay_plan->causal_ranges()[range].offset;
        const std::uint64_t range_length = replay_plan->causal_all()
            ? input_bytes : replay_plan->causal_ranges()[range].length;
        for (std::uint16_t pattern = 1; pattern <= 128; ++pattern) {
          const CausalScr2PatternStats& row = causal_stats[range][pattern];
          if (row.triggers == 0) continue;
          stats << range << ',' << range_offset << ',' << range_length << ','
                << pattern << ',' << row.prefix_length << ','
                << row.suffix_length << ',' << row.triggers << ','
                << row.takes << ',' << row.replayed_bytes << ','
                << row.estimated_gain_bits << ','
                << (row.estimated_gain_bits / 8.0) << '\n';
        }
      }
      if (!stats.good()) return false;
    }
  }
#if FX4_DONOR_PLAN
  if (stats.is_open()) {
    const size_t region_end = e.OutputSize();
    stats << current_region << ','
          << (donor_plan
              ? donor_plan->DonorOffset(current_region)
              : DonorPlan::kNoDonor)
          << ',' << region_start << ',' << region_end << ','
          << (region_end - region_start) << '\n';
  }
#endif
  *output_bytes = os->tellp();
#if FX4_PROGRESS_LOG
  if (progress) fclose(progress);
#endif
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rprogress: 100.00%%");
  fflush(stderr);
#endif
#if FX4_RESIDUAL_ORACLE_TRACE
  if (oracle_trace_stopped) {
    fprintf(stderr, "\noracle trace complete after %llu logical bytes\n",
        static_cast<unsigned long long>(pos));
    return os->good();
  }
#endif

  return pos == input_bytes && os->good();
}
bool Decompress(unsigned long long output_length, std::ifstream* is,
    std::ofstream* os, Predictor* p, DonorPlan* donor_plan,
    VirtualReplayPlan* replay_plan) {
  Decoder d(is, p);
  const unsigned long long progress_step =
      1 + (output_length / FX4_PROGRESS_STEPS);
  unsigned long long next_progress = 0;
  std::vector<char> output;
  output.reserve(FX4_IO_BUFFER_BYTES);
  std::size_t replay_index = 0;
  CausalScr2Matcher causal_scr2(
      replay_plan && replay_plan->causal_scr2()
          ? replay_plan->causal_prior_code() : 0,
      CausalScr2PatternMask(replay_plan), output_length);
  CausalScr2RangeCursor causal_ranges(replay_plan, output_length);
  ClearOutput();

  auto before_byte = [&](unsigned long long logical_pos) -> bool {
#if FX4_DONOR_PLAN
    if (donor_plan && !donor_plan->ReplayAt(logical_pos, p)) return false;
#else
    (void)logical_pos;
#endif
    return true;
  };

  auto after_byte = [&](unsigned long long logical_pos,
      std::uint8_t value) {
#if FX4_DONOR_PLAN
    if (donor_plan) donor_plan->CaptureByte(logical_pos, value);
#else
    (void)logical_pos;
    (void)value;
#endif
  };

  auto emit_byte = [&](std::uint8_t value) {
    output.push_back(static_cast<char>(value));
    if (output.size() >= FX4_IO_BUFFER_BYTES) {
      os->write(output.data(), static_cast<std::streamsize>(output.size()));
      output.clear();
    }
  };

  auto report_progress = [&](unsigned long long logical_pos) {
    if (logical_pos < next_progress) return;
    const double frac = 100.0 * logical_pos / output_length;
#if FX4_STDERR_PROGRESS
    fprintf(stderr, "\rprogress: %.2f%%", frac);
    fflush(stderr);
#endif
    do {
      next_progress += progress_step;
    } while (logical_pos >= next_progress);
  };

  unsigned long long pos = 0;
  while (pos < output_length) {
    std::uint64_t causal_begin = 0;
    std::uint64_t causal_end = 0;
    bool causal_entered = false;
    const bool causal_active = causal_ranges.Active(
        pos, &causal_begin, &causal_end, &causal_entered);
    if (causal_entered) causal_scr2.ResetModel();
    if (causal_active) {
      CausalScr2Matcher::Candidate candidate;
      if (causal_scr2.FindCandidate(&candidate) &&
          pos >= causal_begin + candidate.prefix_length &&
          candidate.suffix_length <= causal_end - pos &&
          candidate.suffix_length <= output_length - pos &&
          pos / FX4_IO_BUFFER_BYTES ==
              (pos + candidate.suffix_length - 1) /
                  FX4_IO_BUFFER_BYTES) {
        const std::uint8_t* suffix = causal_scr2.PatternData(
            candidate.pattern) + candidate.prefix_length;
        const std::uint16_t ppmd_context = causal_scr2.PpmdContext(
            p->PpmdByteProbability(suffix[0]), p->PpmdEffectiveOrder(),
            p->PpmdEscapeDepth(), p->PpmdEscapeRate());
        const bool take = d.DecodeRawBit(
            causal_scr2.Probability(candidate.pattern, candidate.context,
                candidate.class_context, ppmd_context)) != 0;
        causal_scr2.Update(candidate.pattern, candidate.context,
            candidate.class_context, ppmd_context, take);
        if (take) {
          for (std::uint16_t k = 0; k < candidate.suffix_length; ++k) {
            const std::uint8_t value = suffix[k];
            if (!before_byte(pos)) return false;
            d.ObserveKnownByte(value);
            causal_scr2.ObserveByte(value);
            emit_byte(value);
            after_byte(pos, value);
            report_progress(pos);
            ++pos;
          }
          continue;
        }
      }
    }

    if (replay_plan && !replay_plan->causal_scr2() &&
        replay_index < replay_plan->event_count()) {
      const VirtualReplayPlan::Event& event = replay_plan->event(replay_index);
      if (event.offset < pos) return false;
      if (event.offset == pos) {
        const auto& pattern = replay_plan->pattern(event.pattern);
        for (std::uint8_t value : pattern) {
          if (!before_byte(pos)) return false;
          d.ObserveKnownByte(value);
          emit_byte(value);
          after_byte(pos, value);
          report_progress(pos);
          ++pos;
        }
        ++replay_index;
        continue;
      }
    }

    if (!before_byte(pos)) return false;
    int byte = 1;
    while (byte < 256) byte += byte + d.Decode();
    const std::uint8_t value = static_cast<std::uint8_t>(byte);
    if (replay_plan && replay_plan->causal_scr2()) {
      causal_scr2.ObserveByte(value);
    }
    emit_byte(value);
    after_byte(pos, value);
    report_progress(pos);
    ++pos;
  }
  if (replay_plan && !replay_plan->causal_scr2() &&
      replay_index != replay_plan->event_count()) return false;
  if (!output.empty()) {
    os->write(output.data(), static_cast<std::streamsize>(output.size()));
  }
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rprogress: 100.00%%");
  fflush(stderr);
#endif
  return os->good();
}
bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rpreprocessing...");
#endif
  fflush(stderr);
  preprocessor::Encode(data_in, data_out, *input_bytes, temp_path, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr,
    const altxs::ProductMeta* altxs_meta = nullptr,
    const char* altxs_m3_side_path = nullptr) {
  const bool raw_entropy_input =
#if FX4_DONOR_PLAN
      EnvironmentEnabled("FX4_RAW_ENTROPY_INPUT");
#else
      false;
#endif
#if FX4_DONOR_PLAN
  if (raw_entropy_input) {
    struct stat input_info {};
    if (stat(input_path.c_str(), &input_info) != 0 ||
        !CopyResearchStream(input_path, temp_path.c_str())) {
      return false;
    }
    *input_bytes = static_cast<unsigned long long>(input_info.st_size);
  } else {
#endif
    FILE* data_in = fopen(input_path.c_str(), "rb");
    if (!data_in) return false;
    FILE* temp_out = fopen(temp_path.c_str(), "wb");
    if (!temp_out) {
      fclose(data_in);
      return false;
    }

    fseek(data_in, 0L, SEEK_END);
    *input_bytes = ftell(data_in);
    fseek(data_in, 0L, SEEK_SET);

    if (enable_preprocess) {
  #if FX4_STDERR_PROGRESS
      fprintf(stderr, "\rpreprocessing...");
#endif
      fflush(stderr);
      preprocessor::Encode(data_in, temp_out, *input_bytes, temp_path,
          dictionary);
    } else {
      preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
    }
    fclose(data_in);
    fclose(temp_out);
#if FX4_DONOR_PLAN
  }
#endif

#if FX4_RESEARCH_STREAM_DUMP
  if (!CopyResearchStream(temp_path, std::getenv("FX4_DUMP_POST_WRT"))) {
    fprintf(stderr, "cannot dump exact post-WRT stream\n");
    return false;
  }
#endif

  bool altxs_used = false;
#if FX4_ALTXS_M3_M5
  if (altxs_meta) {
    if (post_wrt_side_path || !altxs_m3_side_path ||
        !*altxs_m3_side_path) {
      fprintf(stderr, "altxs M3+M5 and payload_lex/R1 are exclusive\n");
      return false;
    }
    std::uint64_t before_m5 = 0;
    std::uint64_t after_m5 = 0;
    if (!altxs::EncodeWrtProductFile(temp_path, altxs_m3_side_path,
        *altxs_meta, &before_m5, &after_m5)) {
      fprintf(stderr, "altxs M5 product transform failed\n");
      return false;
    }
    altxs_used = true;
    std::remove(altxs_m3_side_path);
    fprintf(stderr, "altxs M5 product: %llu -> %llu bytes\n",
        static_cast<unsigned long long>(before_m5),
        static_cast<unsigned long long>(after_m5));
    malloc_trim(0);
  }
#else
  if (altxs_meta) return false;
#endif

  if (post_wrt_side_path &&
      !r1_reorder::ReorderEncodedTailFile(temp_path, post_wrt_side_path)) {
    fprintf(stderr, "payload_lex encoded-tail reorder failed\n");
    return false;
  }

#if FX4_RESEARCH_STREAM_DUMP
  const char* post_r1_dump_path = std::getenv("FX4_DUMP_POST_R1");
  if (!CopyResearchStream(temp_path, post_r1_dump_path)) {
    fprintf(stderr, "cannot dump exact post-R1 predictor stream\n");
    return false;
  }

  if (EnvironmentEnabled("FX4_STOP_AFTER_POST_R1")) {
    if (!post_wrt_side_path) {
      fprintf(stderr,
          "FX4_STOP_AFTER_POST_R1 is valid only for the R1-enabled -e path\n");
      return false;
    }
    if (!post_r1_dump_path || !*post_r1_dump_path) {
      fprintf(stderr,
          "FX4_STOP_AFTER_POST_R1 requires FX4_DUMP_POST_R1=<output-file>\n");
      return false;
    }
    struct stat stream_info;
    if (stat(temp_path.c_str(), &stream_info) != 0) {
      fprintf(stderr, "cannot measure exact post-R1 predictor stream\n");
      return false;
    }
    *output_bytes = static_cast<unsigned long long>(stream_info.st_size);
    fprintf(stderr, "post-R1 predictor stream dumped: %s (%llu bytes)\n",
        post_r1_dump_path, *output_bytes);
    remove(temp_path.c_str());
    return true;
  }
#endif

  bool postr1_transform_used = false;
#if FX4_POSTR1_TRANSFORM
  const char* postr1_plan_path = std::getenv("FX4_POSTR1_TRANSFORM_PLAN");
  if (postr1_plan_path && *postr1_plan_path) {
    if (altxs_used) {
      fprintf(stderr,
          "post-R1 plans require the payload_lex/R1 coordinate space\n");
      return false;
    }
    if (!post_wrt_side_path && !raw_entropy_input) {
      fprintf(stderr,
          "FX4_POSTR1_TRANSFORM_PLAN requires -e or "
          "FX4_RAW_ENTROPY_INPUT=1\n");
      return false;
    }
    std::uint64_t before_transform = 0;
    std::uint64_t after_transform = 0;
    if (!postr1::EncodeFile(temp_path, postr1_plan_path,
        &before_transform, &after_transform)) {
      fprintf(stderr, "selective post-R1 transform failed\n");
      return false;
    }
    postr1_transform_used = true;
    fprintf(stderr, "post-R1 portfolio: %llu -> %llu bytes\n",
        static_cast<unsigned long long>(before_transform),
        static_cast<unsigned long long>(after_transform));
    malloc_trim(0);
  }
#endif

  bool scr2_used = false;
#if FX4_SCR2
  const bool scr2_requested = EnvironmentEnabled("FX4_ENABLE_SCR2") ||
      (FX4_SCR2_DEFAULT != 0 && post_wrt_side_path != nullptr);
  if (scr2_requested && altxs_used) {
    fprintf(stderr, "SCR2 plans are not valid on the altxs M5 stream\n");
    return false;
  }
  if (scr2_requested && !postr1_transform_used && !altxs_used) {
    std::uint64_t before_scr2 = 0;
    std::uint64_t after_scr2 = 0;
    if (!scr2::EncodeFile(temp_path, &before_scr2, &after_scr2)) {
      fprintf(stderr, "SCR2 post-R1 transform failed\n");
      return false;
    }
    if (after_scr2 < before_scr2) {
      scr2_used = true;
      fprintf(stderr, "SCR2: %llu -> %llu bytes\n",
          static_cast<unsigned long long>(before_scr2),
          static_cast<unsigned long long>(after_scr2));
    } else {
      std::uint64_t restored_size = 0;
      if (!scr2::DecodeFile(temp_path, &restored_size) ||
          restored_size != before_scr2) {
        fprintf(stderr, "SCR2 raw fallback restore failed\n");
        return false;
      }
    }
    malloc_trim(0);
  }
#endif

#if FX4_VIRTUAL_REPLAY
  const char* replay_dump_path = std::getenv("FX4_VR_DUMP_INPUT");
  if (!CopyResearchStream(temp_path, replay_dump_path)) {
    fprintf(stderr, "cannot dump exact FX4 virtual-replay input\n");
    return false;
  }
#endif

  std::ifstream temp_in(temp_path, std::ios::in | std::ios::binary);
  if (!temp_in.is_open()) return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  temp_in.seekg(0, std::ios::end);
  unsigned long long temp_bytes = temp_in.tellg();
  temp_in.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
  const bool force_full_vocab =
#if FX4_DONOR_PLAN
      EnvironmentEnabled("FX4_FORCE_FULL_VOCAB");
#else
      false;
#endif
  if (force_full_vocab || temp_bytes < kMinVocabFileSize) {
    std::fill(vocab.begin(), vocab.end(), true);
  } else {
    ExtractVocab(temp_bytes, &temp_in, &vocab);
    temp_in.seekg(0, std::ios::beg);
  }

#if FX4_RESEARCH_DONOR_BOOTSTRAP
  const std::vector<std::uint8_t> research_donor =
      LoadResearchDonorBootstrap();
  AddResearchDonorVocabulary(research_donor, &vocab);
#endif

  VirtualReplayPlan* active_replay_plan = nullptr;
#if FX4_VIRTUAL_REPLAY
  VirtualReplayPlan replay_plan;
  const char* replay_plan_path = std::getenv("FX4_VR_PLAN");
  const char* causal_scr2_spec = std::getenv("FX4_CAUSAL_SCR2");
  if (replay_plan_path && *replay_plan_path &&
      causal_scr2_spec && *causal_scr2_spec) {
    fprintf(stderr, "FX4_VR_PLAN and FX4_CAUSAL_SCR2 are exclusive\n");
    return false;
  }
  if (replay_plan_path && *replay_plan_path) {
    if (!replay_plan.LoadExternal(replay_plan_path, temp_bytes)) {
      fprintf(stderr, "invalid FX4_VR_PLAN: %s\n", replay_plan_path);
      return false;
    }
    if (!replay_plan.empty()) active_replay_plan = &replay_plan;
  } else if (causal_scr2_spec && *causal_scr2_spec) {
    if (scr2_used || postr1_transform_used) {
      fprintf(stderr,
          "FX4_CAUSAL_SCR2 requires the unmodified post-R1 stream\n");
      return false;
    }
    unsigned int prior_code = 0;
    const char* prior = std::getenv("FX4_CAUSAL_SCR2_PRIOR");
    if (prior && std::strcmp(prior, "75") == 0) prior_code = 1;
    else if (prior && (std::strcmp(prior, "88") == 0 ||
                      std::strcmp(prior, "87") == 0)) prior_code = 2;
    else if (prior && *prior && std::strcmp(prior, "50") != 0) {
      fprintf(stderr, "FX4_CAUSAL_SCR2_PRIOR must be 50, 75, or 88\n");
      return false;
    }
    const char* patterns = std::getenv("FX4_CAUSAL_SCR2_PATTERNS");
    if (!replay_plan.ConfigureCausalScr2(causal_scr2_spec, temp_bytes,
        prior_code, patterns ? patterns : "all")) {
      fprintf(stderr, "invalid FX4_CAUSAL_SCR2: %s\n", causal_scr2_spec);
      return false;
    }
    active_replay_plan = &replay_plan;
    fprintf(stderr, "causal SCR2 enabled: %s, prior=%s\n",
        causal_scr2_spec, prior_code == 0 ? "50" :
        (prior_code == 1 ? "75" : "88"));
  }
#endif

  DonorPlan* active_donor_plan = nullptr;
#if FX4_DONOR_PLAN
  DonorPlan donor_plan;
  const char* donor_plan_path = std::getenv("FX4_DONOR_PLAN");
  if (donor_plan_path && *donor_plan_path) {
    if (altxs_used) {
      fprintf(stderr,
          "donor plans must be regenerated in the altxs M5 coordinate space\n");
      return false;
    }
    if (!donor_plan.LoadExternal(donor_plan_path, temp_bytes)) {
      fprintf(stderr, "invalid FX4_DONOR_PLAN: %s\n", donor_plan_path);
      return false;
    }
    if (!donor_plan.empty()) active_donor_plan = &donor_plan;
  }
#endif

#if FX4_DONOR_FORK_DISCOVERY
  const char* discovery_results =
      std::getenv("FX4_DONOR_DISCOVERY_RESULTS");
  if (discovery_results && *discovery_results && active_donor_plan) {
    temp_in.close();
    data_out.close();
    remove(output_path.c_str());
    uint64_t discovery_output_bytes = 0;
    const bool discovery_ok = RunDonorForkDiscovery(
        temp_path, output_path, temp_bytes, vocab, dictionary,
        enable_preprocess || dictionary != nullptr, active_donor_plan,
        &discovery_output_bytes);
    *output_bytes = discovery_output_bytes;
    remove(output_path.c_str());
    remove(temp_path.c_str());
    return discovery_ok;
  }
#endif

  WriteHeader(temp_bytes, vocab, dictionary != NULL,
      active_donor_plan != nullptr, scr2_used,
      postr1_transform_used, active_replay_plan != nullptr, altxs_used,
      &data_out);
#if FX4_DONOR_PLAN
  if (active_donor_plan && !active_donor_plan->WriteArchive(&data_out)) {
    fprintf(stderr, "cannot write FX4 donor plan\n");
    return false;
  }
#endif
#if FX4_VIRTUAL_REPLAY
  if (active_replay_plan && !active_replay_plan->WriteArchive(&data_out)) {
    fprintf(stderr, "cannot write FX4 virtual-replay plan\n");
    return false;
  }
#endif
  Predictor p(vocab, scr2_used);
  if (enable_preprocess
#if FX4_DONOR_FORK_DISCOVERY || FX4_DONOR_PLAN
      || dictionary != nullptr
#endif
  ) {
    preprocessor::Pretrain(&p, dictionary);
  }
#if FX4_RESEARCH_DONOR_BOOTSTRAP
  else if (dictionary) {
    preprocessor::Pretrain(&p, dictionary);
  }
  if (!EnvironmentEnabled("FX4_RESEARCH_DONOR_PROFILE_ONLY")) {
    ReplayResearchDonor(research_donor, &p);
  }
  ConfigureResearchDonorProfile(research_donor, &p);
#endif
  if (!Compress(temp_bytes, &temp_in, &data_out, output_bytes, &p,
      active_donor_plan, active_replay_plan)) {
    fprintf(stderr, "FX4 entropy compression failed\n");
    return false;
  }
  temp_in.close();
  data_out.close();
#if FX4_RESIDUAL_ORACLE_TRACE
  if (oracle_trace_stopped) remove(output_path.c_str());
#endif
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  std::vector<bool> vocab(256, false);
  bool dictionary_used = false;
  bool donor_plan_used = false;
  bool scr2_used = false;
  bool postr1_transform_used = false;
  bool virtual_replay_used = false;
  bool altxs_used = false;
  if (!ReadHeader(&data_in, output_bytes, &dictionary_used,
      &donor_plan_used, &scr2_used, &postr1_transform_used,
      &virtual_replay_used, &altxs_used, &vocab)) {
    return false;
  }
  if (!dictionary_used && dictionary != NULL) return false;
  if (dictionary_used && dictionary == NULL) return false;

  if (*output_bytes == 0) {  // undo store
    if (scr2_used || postr1_transform_used || virtual_replay_used ||
        altxs_used) {
      return false;
    }
    data_in.close();
    FILE* in = fopen(input_path.c_str(), "rb");
    if (!in) return false;
    FILE* data_out = fopen(output_path.c_str(), "wb");
    if (!data_out) return false;
    fseek(in, 5L, SEEK_SET);
    fprintf(stderr, "\rdecoding...");
    fflush(stderr);
    preprocessor::Decode(in, data_out, dictionary);
    fseek(data_out, 0L, SEEK_END);
    *output_bytes = ftell(data_out);
    fclose(in);
    fclose(data_out);
    return true;
  }
  {
    DonorPlan* active_donor_plan = nullptr;
#if FX4_DONOR_PLAN
    DonorPlan donor_plan;
    if (donor_plan_used) {
      if (!donor_plan.ReadArchive(&data_in, *output_bytes)) {
        fprintf(stderr, "invalid FX4 donor plan in archive\n");
        return false;
      }
      active_donor_plan = &donor_plan;
    }
#else
    if (donor_plan_used) {
      fprintf(stderr, "archive requires donor-enabled FX4 build\n");
      return false;
    }
#endif

    VirtualReplayPlan* active_replay_plan = nullptr;
#if FX4_VIRTUAL_REPLAY
    VirtualReplayPlan replay_plan;
    if (virtual_replay_used) {
      if (!replay_plan.ReadArchive(&data_in, *output_bytes)) {
        fprintf(stderr, "invalid FX4 virtual-replay plan in archive\n");
        return false;
      }
      active_replay_plan = &replay_plan;
    }
#else
    if (virtual_replay_used) {
      fprintf(stderr, "archive requires virtual-replay-enabled FX4 build\n");
      return false;
    }
#endif

#if FX4_RESEARCH_DONOR_BOOTSTRAP
    const std::vector<std::uint8_t> research_donor =
        LoadResearchDonorBootstrap();
    AddResearchDonorVocabulary(research_donor, &vocab);
#endif
    Predictor p(vocab, scr2_used);
    if (dictionary_used) preprocessor::Pretrain(&p, dictionary);
#if FX4_RESEARCH_DONOR_BOOTSTRAP
    if (!EnvironmentEnabled("FX4_RESEARCH_DONOR_PROFILE_ONLY")) {
      ReplayResearchDonor(research_donor, &p);
    }
    ConfigureResearchDonorProfile(research_donor, &p);
#endif

    std::ofstream temp_out(temp_path, std::ios::out | std::ios::binary);
    if (!temp_out.is_open()) return false;
    if (!Decompress(*output_bytes, &data_in, &temp_out, &p,
        active_donor_plan, active_replay_plan)) {
      fprintf(stderr, "FX4 entropy decompression failed\n");
      return false;
    }
    temp_out.close();
    p.FreeFxcmMemory();
    data_in.close();
  }
  malloc_trim(0);

#if FX4_ALTXS_M3_M5
  if (altxs_used) {
    std::uint64_t restored_wrt_bytes = 0;
    if (!altxs::DecodeWrtProductFile(temp_path, kAltxsM3SideDecomp,
        &altxs_decode_meta, &restored_wrt_bytes)) {
      fprintf(stderr, "altxs M5 product restore failed\n");
      return false;
    }
    altxs_decode_used = true;
    fprintf(stderr, "altxs M5 restore: %llu WRT bytes\n",
        static_cast<unsigned long long>(restored_wrt_bytes));
    malloc_trim(0);
  }
#else
  if (altxs_used) return false;
#endif

#if FX4_POSTR1_TRANSFORM
  if (postr1_transform_used) {
    std::uint64_t restored_size = 0;
    if (!postr1::DecodeFile(temp_path, &restored_size)) {
      fprintf(stderr, "selective post-R1 inverse transform failed\n");
      return false;
    }
  }
#else
  if (postr1_transform_used) return false;
#endif

#if FX4_SCR2
  if (scr2_used) {
    std::uint64_t restored_size = 0;
    if (!scr2::DecodeFile(temp_path, &restored_size)) {
      fprintf(stderr, "SCR2 inverse transform failed\n");
      return false;
    }
  }
#else
  if (scr2_used) return false;
#endif

#if FX4_DONOR_PLAN
  if (EnvironmentEnabled("FX4_RAW_ENTROPY_OUTPUT")) {
    if (post_wrt_side_path || scr2_used || virtual_replay_used || altxs_used ||
        !CopyResearchStream(temp_path, output_path.c_str())) {
      return false;
    }
    struct stat output_info {};
    if (stat(output_path.c_str(), &output_info) != 0) return false;
    *output_bytes = static_cast<unsigned long long>(output_info.st_size);
    remove(temp_path.c_str());
    return true;
  }
#endif

  if (post_wrt_side_path && !altxs_used) {
    if (!r1_reorder::ExtractSideFromFile(temp_path, post_wrt_side_path) ||
        !r1_reorder::RestoreEncodedTailFile(temp_path, post_wrt_side_path)) {
      fprintf(stderr, "payload_lex encoded-tail restore failed\n");
      return false;
    }
  }


  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

int main(int argc, char** argv) {
if ((argc != 1) && (argv[1][1] != 'h') && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e' ))) {
    return Help();
  }
   srand(SEED);

  clock_t start = clock();

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;


  if ((argc > 1) && (argv[1][1] != 'h'))  {
    if (argv[1][1] == 'n') enable_preprocess = false;
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') {
#if !FX4_RESEARCH_DONOR_BOOTSTRAP && !FX4_DONOR_FORK_DISCOVERY && !FX4_DONOR_PLAN
        return Help();
#endif
      }
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
  }

  std::string temp_path = output_path + ".cmix.temp";

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    std::cout << "Running cmix decompression..." << std::endl;
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");//_decomp

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes, ".r1_payload_lex_side_decomp")) {
      return Help();
    }
    std::cout << "Cmix decompression finished" << std::endl;

#if FX4_ALTXS_M3_M5
    if (altxs_decode_used) {
      if (!altxs::SplitDenseReadyFile(output_path, altxs_decode_meta,
          ".main_decomp_dense", ".intro_decomp", ".coda_decomp") ||
          !altxs::RestorePhda9File(".main_decomp_dense",
              kAltxsM3SideDecomp, ".main_decomp")) {
        fprintf(stderr, "altxs M3 outer restore failed\n");
        return Help();
      }
      std::remove(kAltxsM3SideDecomp);
    } else {
      split4Decomp();
    }
#else
    split4Decomp();
#endif

    // apply phda9 preprocessor
    phda9_resto();

    // change the order of articles in the input
    sort();

    // merge all input parts after preprocessing
    cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d");
    cat("un1_d", ".coda_decomp", "enwik9_uncompressed");

    goto print_end_message;
  }

  if (argv[1][1] == 's') {
    if (!Store(input_path, temp_path, output_path, dictionary, &input_bytes,
        &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'c' || argv[1][1] == 'n') {
      remove(".dict");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
#if FX4_RESIDUAL_ORACLE_TRACE
    if (oracle_trace_stopped) return 0;
#endif
  } else if (argv[1][1] == 'e') {
    // Compress enwik9
    input_path = argv[2];
    output_path = argv[3]; //name of a compressor output

    if (selfextract_comp() != 0) {
      return Help();
    }

    // Preparing enwik9 for reordering
    split4Comp(input_path.c_str());

    // change the order of articles in the input
    reorder();

#if FX4_ALTXS_M3_M5
    altxs::ProductMeta altxs_meta;
    altxs_meta.intro_bytes = getFileSize(".intro");
    altxs_meta.coda_bytes = getFileSize(".coda");
    altxs_meta.main_bytes = getFileSize(".main");
#endif

    // apply phda9 preprocessor
    phda9_prepr();

#if FX4_ALTXS_M3_M5
    {
      std::uint64_t before_m3 = 0;
      std::uint64_t after_m3 = 0;
      std::uint64_t side_m3 = 0;
      if (!altxs::DensifyPhda9File(".main_phda9prepr", kAltxsM3Side,
          &before_m3, &after_m3, &side_m3)) {
        fprintf(stderr, "altxs M3 PHDA9 densify failed\n");
        return Help();
      }
      fprintf(stderr, "altxs M3: %llu -> %llu bytes + %llu side bytes\n",
          static_cast<unsigned long long>(before_m3),
          static_cast<unsigned long long>(after_m3),
          static_cast<unsigned long long>(side_m3));
      malloc_trim(0);
    }
#endif

    // merge all input parts after preprocessing
    cat(".main_phda9prepr", ".intro", "un1");
    cat("un1", ".coda", ".ready4cmix");

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
#if FX4_ALTXS_M3_M5
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes, nullptr, &altxs_meta,
        kAltxsM3Side)) {
#else
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes, ".r1_payload_lex_side")) {
#endif
      return Help();
    }
#if FX4_RESIDUAL_ORACLE_TRACE
    if (oracle_trace_stopped) return 0;
#endif
#if FX4_RESEARCH_STREAM_DUMP
    if (EnvironmentEnabled("FX4_STOP_AFTER_POST_R1")) return 0;
#endif
#if FX4_DONOR_FORK_DISCOVERY
    if (DonorForkDiscoveryCompleted()) return 0;
#endif

    // construct a selfextracting decompressor binary
    // archive9 = decomp_binary(upxed) + comp_dict + cmix_output + header.dat
    cat(".decomp_bin", ".dict.comp", "dec1");

    // get the size of the output file
    size_t output_size = getFileSize(output_path);

    HeaderInfo header;
    read("test.dat", header);
    header.decomp_input_size = output_size;
#ifdef KH_TRANSFORMER6M_ARCHIVE
    if (header.transformer6m_weights_size <= 0 ||
        static_cast<size_t>(header.transformer6m_weights_size) !=
            getFileSize(".tfweights")) {
      fprintf(stderr, "invalid or missing packaged transformer6m weights\n");
      return Help();
    }
#endif
#ifdef KH_BITLSTM32_ARCHIVE
    {
      const char* head_path = getenv("KH_BITLSTM32");
      std::ofstream head_out(".head_blob4archive",
          std::ios::binary | std::ios::trunc);
      if (!head_out) return Help();
      if (head_path && head_path[0]) {
        std::ifstream head_in(head_path, std::ios::binary);
        if (!head_in) return Help();
        head_out << head_in.rdbuf();
        if (!head_out) return Help();
      }
      head_out.close();
      header.head_blob_size =
          static_cast<int>(getFileSize(".head_blob4archive"));
      // This field describes S1 only; the copy in archive9 is head_blob_size.
      header.s1_head_blob_size = 0;
    }
#endif
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
    {
      const char* model_path = getenv("KH_RESIDUAL_LSTM96");
      std::ofstream model_out(".residual_lstm96_blob4archive",
          std::ios::binary | std::ios::trunc);
      if (!model_out || !model_path || !model_path[0]) return Help();
      std::ifstream model_in(model_path, std::ios::binary);
      if (!model_in) return Help();
      model_out << model_in.rdbuf();
      if (!model_out) return Help();
      model_out.close();
      header.residual_lstm96_blob_size = static_cast<int>(
          getFileSize(".residual_lstm96_blob4archive"));
      if (header.residual_lstm96_blob_size <= 0) return Help();
    }
#endif
    write("header4archive.dat", header);

#ifdef KH_TRANSFORMER6M_ARCHIVE
    cat("dec1", ".tfweights", "dec1t");
    cat("dec1t", output_path.c_str(), "dec2");
#else
    cat("dec1", output_path.c_str(), "dec2");
#endif
#ifdef KH_BITLSTM32_ARCHIVE
    cat("dec2", ".head_blob4archive", "dec2c");
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
    cat("dec2c", ".residual_lstm96_blob4archive", "dec2r");
    cat("dec2r", "header4archive.dat", "archive9");
#else
    cat("dec2c", "header4archive.dat", "archive9");
#endif
#else
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
    cat("dec2", ".residual_lstm96_blob4archive", "dec2r");
    cat("dec2r", "header4archive.dat", "archive9");
#else
    cat("dec2", "header4archive.dat", "archive9");
#endif
#endif

    // make the decompressor binary executable
    char mode[] = "0777";
    char buf[100] = "archive9";
    int i = strtol(mode, 0, 8);
    chmod(buf, i);

  } else if (argv[1][1] == 'h') {
    if (argc < 5) return Help();
    HeaderInfo header = {};
    header.dict_size = atoi(argv[2]);
    header.new_article_order_size = atoi(argv[3]);
    header.decomp_input_size = atoi(argv[4]);
#ifdef KH_BITLSTM32_ARCHIVE
    header.head_blob_size = 0;
    if (argc < 7) return Help();
    header.s1_head_blob_size = atoi(argv[6]);
    if (header.s1_head_blob_size <= 0) return Help();
#endif
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
    header.residual_lstm96_blob_size = 0;
#endif
#ifdef KH_TRANSFORMER6M_ARCHIVE
    if (argc < 6) return Help();
    header.transformer6m_weights_size = atoi(argv[5]);
    if (header.transformer6m_weights_size <= 0) return Help();
#endif
    write("header.dat", header);
    goto exit;
  }  else if (argv[1][1] == 'x') {
    // run compression
    input_path = argv[2];
    output_path = argv[3];
    dictionary = fopen(".dict", "rb");
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
    goto print_end_message;
  }
  else {
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

print_end_message:
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);

exit:
  return 0;
}
