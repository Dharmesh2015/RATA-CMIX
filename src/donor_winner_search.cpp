#include "donor_winner_search.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "coder/encoder.h"
#include "donor_plan.h"
#include "models/postr1_experts.h"
#include "predictor.h"
#include "preprocess/preprocessor.h"

namespace {

constexpr uint32_t kAlignment = 256;
constexpr uint32_t kInitialLength = 4096;
constexpr uint32_t kLossSpanBytes = 4096;
constexpr size_t kLossSpanCount =
    DonorPlan::kChunkSize / kLossSpanBytes;
constexpr size_t kPostingCapacity = 6;

constexpr const char* kTrialHeader =
    "recipient_region,recipient_offset,trial_id,stage,depth,donor_count,"
    "donors,baseline_payload_bytes,trial_payload_bytes,gain_bytes,"
    "marginal_gain_bytes,status\n";
constexpr const char* kSelectionHeader =
    "recipient_region,recipient_offset,baseline_payload_bytes,"
    "selected_payload_bytes,gain_bytes,net_gain_bytes,donor_count,donors,"
    "candidate_count,exact_trials\n";
constexpr const char* kSpanHeader =
    "recipient_region,recipient_offset,span_rank,span_offset,"
    "emitted_bytes\n";
constexpr const char* kCandidateHeader =
    "recipient_region,recipient_offset,candidate_rank,donor_offset,"
    "donor_length,phrase_score,source\n";
constexpr const char* kCostHeader =
    "region,baseline_payload,candidate_payload,assignment_cost,span_cost,"
    "copy_cost,candidate_total,saving_bytes,improvement_percent,status\n";
constexpr const char* kMarginalHeader =
    "recipient_region,recipient_offset,profile_donors,removed_donor,"
    "profile_donor_count,full_payload_bytes,without_payload_bytes,"
    "marginal_gain_bytes,status\n";

struct DonorWindow {
  uint32_t offset = 0;
  uint32_t length = 0;
  uint32_t score = 0;
};

struct RecipientSpan {
  uint32_t region = 0;
  uint64_t offset = 0;
  uint32_t length = 0;
  uint8_t stream_class =
      static_cast<uint8_t>(PostR1Experts::StreamClass::kMixed);
};

std::vector<uint64_t> recipient_offsets;
bool recipient_spans_from_csv = false;
std::vector<std::vector<DonorWindow>> ranked_page_candidates;
std::unordered_set<std::uint32_t> ranked_recipient_filter;
bool ranked_recipient_filter_enabled = false;
std::uint8_t current_recipient_stream_class =
    static_cast<std::uint8_t>(PostR1Experts::StreamClass::kMixed);

uint64_t RecipientOffset(uint32_t region) {
  return region < recipient_offsets.size()
      ? recipient_offsets[region]
      : static_cast<uint64_t>(region) * DonorPlan::kChunkSize;
}

using DonorSequence = std::vector<DonorWindow>;

struct ProbeMessage {
  uint64_t payload_bytes = 0;
  uint32_t ok = 0;
  uint16_t span_count = 0;
  uint16_t reserved = 0;
  std::array<uint32_t, kLossSpanCount> span_bytes{};
};

struct TrialRecord {
  bool ok = false;
  uint64_t payload_bytes = 0;
};

struct SelectionRecord {
  bool valid = false;
  uint64_t baseline_bytes = 0;
  uint64_t selected_bytes = 0;
  DonorSequence sequence;
};

struct ScoredSequence {
  DonorSequence sequence;
  uint64_t payload_bytes = 0;
};

struct SearchConfig {
  uint32_t top_spans = 8;
  uint32_t candidate_offsets = 12;
  uint32_t refine_offsets = 4;
  uint32_t local_seeds = 2;
  uint32_t local_radius = 1024;
  uint32_t combo_candidates = 8;
  uint32_t beam_width = 4;
  uint32_t max_depth = 8;
  uint32_t near_loss_bytes = 64;
  uint32_t planned_donors = 7;
  uint32_t max_regions = 0;
  uint32_t start_region = 1;
  uint64_t max_new_trials = 0;
  uint64_t stop_offset = std::numeric_limits<uint64_t>::max();
  bool planned_only = false;
  bool planned_prefixes = true;
  bool leave_one_out = true;
  bool quick_singles = false;
  uint32_t quick_donor_strength = 1;
  uint32_t quick_metadata_bytes = 29;
  bool quick_context_mixer = false;
  uint32_t quick_pair_candidates = 0;
  uint32_t quick_prefix_depth = 1;
  bool quick_reverse_prefixes = false;
};

uint64_t EnvironmentU64(const char* name, uint64_t fallback) {
  const char* text = getenv(name);
  if (!text || !*text) return fallback;
  char* end = nullptr;
  const unsigned long long value = strtoull(text, &end, 10);
  return *end == '\0' ? value : fallback;
}

uint32_t EnvironmentU32(const char* name, uint32_t fallback,
    uint32_t minimum, uint32_t maximum) {
  const uint64_t value = EnvironmentU64(name, fallback);
  if (value < minimum || value > maximum) return fallback;
  return static_cast<uint32_t>(value);
}

SearchConfig LoadConfig() {
  SearchConfig config;
  config.top_spans = EnvironmentU32("FX4_WINNER_TOP_SPANS", 8, 1, 32);
  config.candidate_offsets =
      EnvironmentU32("FX4_WINNER_CANDIDATES", 12, 1, 64);
  config.refine_offsets =
      EnvironmentU32("FX4_WINNER_REFINE_OFFSETS", 4, 1, 16);
  config.local_seeds =
      EnvironmentU32("FX4_WINNER_LOCAL_SEEDS", 2, 0, 8);
  config.local_radius =
      EnvironmentU32("FX4_WINNER_LOCAL_RADIUS", 1024, 0, 16384);
  config.combo_candidates =
      EnvironmentU32("FX4_WINNER_COMBO_CANDIDATES", 8, 2, 16);
  config.beam_width =
      EnvironmentU32("FX4_WINNER_BEAM_WIDTH", 4, 1, 16);
  config.max_depth =
      EnvironmentU32("FX4_WINNER_MAX_DEPTH", 8, 2, 8);
  config.near_loss_bytes =
      EnvironmentU32("FX4_WINNER_NEAR_BYTES", 64, 0, 1024);
  config.planned_donors =
      EnvironmentU32("FX4_WINNER_PLANNED_DONORS", 7, 1, 8);
  config.max_regions =
      EnvironmentU32("FX4_WINNER_MAX_REGIONS", 0, 0, 65535);
  config.start_region =
      EnvironmentU32("FX4_WINNER_START_REGION", 1, 0, 65535);
  config.max_new_trials =
      EnvironmentU64("FX4_DONOR_MAX_NEW_TRIALS", 0);
  config.stop_offset =
      EnvironmentU64("FX4_WINNER_STOP_OFFSET",
          std::numeric_limits<uint64_t>::max());
  config.planned_only =
      EnvironmentU32("FX4_WINNER_PLANNED_ONLY", 0, 0, 1) != 0;
  config.planned_prefixes =
      EnvironmentU32("FX4_WINNER_PLANNED_PREFIXES", 1, 0, 1) != 0;
  config.leave_one_out =
      EnvironmentU32("FX4_WINNER_LEAVE_ONE_OUT", 1, 0, 1) != 0;
  config.quick_singles =
      EnvironmentU32("FX4_WINNER_QUICK_SINGLES", 0, 0, 1) != 0;
  config.quick_donor_strength =
      EnvironmentU32("FX4_WINNER_DONOR_STRENGTH", 1, 0, 3);
  config.quick_metadata_bytes =
      EnvironmentU32("FX4_WINNER_METADATA_BYTES", 29, 0, 1024);
  config.quick_context_mixer =
      EnvironmentU32("FX4_WINNER_CONTEXT_MIXER", 0, 0, 1) != 0;
  config.quick_pair_candidates =
      EnvironmentU32("FX4_WINNER_QUICK_PAIR_CANDIDATES", 0, 0, 16);
  config.quick_prefix_depth =
      EnvironmentU32("FX4_WINNER_QUICK_PREFIX_DEPTH", 1, 1, 8);
  config.quick_reverse_prefixes =
      EnvironmentU32("FX4_WINNER_QUICK_REVERSE_PREFIXES", 0, 0, 1) != 0;
  return config;
}

void SetCpuFromEnvironment(const char* name) {
  const char* text = getenv(name);
  if (!text || !*text) return;
  char* end = nullptr;
  const long cpu = strtol(text, &end, 10);
  if (*end != '\0' || cpu < 0 || cpu >= CPU_SETSIZE) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(static_cast<int>(cpu), &set);
  sched_setaffinity(0, sizeof(set), &set);
}

bool WriteAll(int fd, const void* data, size_t size) {
  const uint8_t* input = static_cast<const uint8_t*>(data);
  while (size != 0) {
    const ssize_t written = write(fd, input, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    input += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t size) {
  uint8_t* output = static_cast<uint8_t*>(data);
  while (size != 0) {
    const ssize_t count = read(fd, output, size);
    if (count == 0) return false;
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    output += count;
    size -= static_cast<size_t>(count);
  }
  return true;
}

bool SyncFile(FILE* file) {
  return fflush(file) == 0 && fsync(fileno(file)) == 0;
}

std::string TrialPath(const std::string& base) {
  return base + ".winner_trials.csv";
}

std::string NearPath(const std::string& base) {
  return base + ".winner_near.csv";
}

std::string WinnersPath(const std::string& base) {
  return base + ".winner_positive.csv";
}

std::string SelectionPath(const std::string& base) {
  return base + ".winner_selected.csv";
}

std::string SpanPath(const std::string& base) {
  return base + ".winner_spans.csv";
}

std::string CandidatePath(const std::string& base) {
  return base + ".winner_candidates.csv";
}

std::string CostAccountingPath(const std::string& base) {
  return base + ".cost_accounting.csv";
}

std::string MarginalPath(const std::string& base) {
  return base + ".winner_marginals.csv";
}

std::string StatusPath(const std::string& base) {
  return base + ".winner.status";
}

std::string CompletePath(const std::string& base) {
  return base + ".winner.complete";
}

std::string PausedPath(const std::string& base) {
  return base + ".winner.paused";
}

bool EnsureCsv(const std::string& path, const char* header) {
  struct stat info {};
  if (stat(path.c_str(), &info) == 0 && info.st_size != 0) {
    std::ifstream input(path);
    std::string first;
    return input.is_open() && std::getline(input, first) &&
        first + "\n" == header;
  }
  FILE* output = fopen(path.c_str(), "wb");
  if (!output) return false;
  const bool ok = fputs(header, output) >= 0 && SyncFile(output);
  fclose(output);
  return ok;
}

bool EnsureLedgers(const std::string& base) {
  return EnsureCsv(TrialPath(base), kTrialHeader) &&
      EnsureCsv(NearPath(base), kTrialHeader) &&
      EnsureCsv(WinnersPath(base), kTrialHeader) &&
      EnsureCsv(SelectionPath(base), kSelectionHeader) &&
      EnsureCsv(SpanPath(base), kSpanHeader) &&
      EnsureCsv(CandidatePath(base), kCandidateHeader) &&
      EnsureCsv(CostAccountingPath(base), kCostHeader) &&
      EnsureCsv(MarginalPath(base), kMarginalHeader);
}

bool AppendRow(const std::string& path, const std::string& row) {
  FILE* output = fopen(path.c_str(), "ab");
  if (!output) return false;
  const bool ok = fwrite(row.data(), 1, row.size(), output) == row.size() &&
      SyncFile(output);
  fclose(output);
  return ok;
}

size_t CompactVarintSize(uint64_t value) {
  size_t bytes = 1;
  while (value >= 128u) {
    value >>= 7;
    ++bytes;
  }
  return bytes;
}

uint64_t CompactSignedDelta(int64_t value) {
  return value >= 0
      ? static_cast<uint64_t>(value) << 1
      : (static_cast<uint64_t>(-(value + 1)) << 1) | 1u;
}

// Exact v7 bytes for one recipient's donor sequence. Discovery and production
// now rank the same delta-coded offset representation.
uint64_t DonorPlanMetadataCost(
    uint32_t recipient, const DonorSequence& sequence) {
  if (sequence.empty()) return 0;
  uint64_t bytes =
      2 + CompactVarintSize(1) +
      CompactVarintSize(recipient) + CompactVarintSize(sequence.size());
  int64_t previous_shifted_offset = 0;
  for (size_t index = 0; index < sequence.size(); ++index) {
    const int64_t shifted_offset = sequence[index].offset >> 8;
    const uint64_t encoded_offset = index == 0
        ? static_cast<uint64_t>(shifted_offset)
        : CompactSignedDelta(shifted_offset - previous_shifted_offset);
    bytes += CompactVarintSize(encoded_offset) + 1;
    previous_shifted_offset = shifted_offset;
  }
  bytes += CompactVarintSize(0);  // no shared expert profiles
  bytes += CompactVarintSize(0);  // no expert spans
  return bytes;
}

// ceil(baseline * 0.01) via integer arithmetic: ceil(a/100) = (a+99)/100.
uint64_t StrongTargetBytes(uint64_t baseline) {
  return baseline - (baseline + 99) / 100;
}

const char* AcceptanceStatus(uint64_t candidate_total, uint64_t baseline,
    uint64_t strong_target) {
  if (candidate_total <= strong_target) return "STRONG_1PCT";
  if (candidate_total < baseline) return "POSITIVE_BELOW_1PCT";
  return "REJECTED_SAVED";
}

// Durable Phase-2 cost-accounting row: total archive cost (payload +
// donor-plan metadata + span/COPY metadata, the latter two 0 until Phases
// 11/13 land) compared against baseline, with the STRONG_1PCT /
// POSITIVE_BELOW_1PCT / REJECTED_SAVED acceptance status.
bool AppendCostAccounting(const std::string& base, uint32_t region,
    uint64_t baseline_payload, uint64_t candidate_payload,
    uint64_t assignment_cost, uint64_t span_cost, uint64_t copy_cost) {
  const uint64_t candidate_total =
      candidate_payload + assignment_cost + span_cost + copy_cost;
  const int64_t saving_bytes =
      static_cast<int64_t>(baseline_payload) -
      static_cast<int64_t>(candidate_total);
  const double improvement_percent = baseline_payload == 0 ? 0.0 :
      (100.0 * static_cast<double>(saving_bytes)) /
          static_cast<double>(baseline_payload);
  const uint64_t strong_target = StrongTargetBytes(baseline_payload);
  const char* status =
      AcceptanceStatus(candidate_total, baseline_payload, strong_target);
  char row[512] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
      ",%" PRIu64 ",%" PRId64 ",%.4f,%s\n",
      region, baseline_payload, candidate_payload, assignment_cost,
      span_cost, copy_cost, candidate_total, saving_bytes,
      improvement_percent, status);
  return length > 0 && static_cast<size_t>(length) < sizeof(row) &&
      AppendRow(CostAccountingPath(base),
          std::string(row, static_cast<size_t>(length)));
}

std::vector<std::string> Split(const std::string& text, char delimiter) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t end = text.find(delimiter, start);
    fields.push_back(text.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  if (!fields.empty() && !fields.back().empty() &&
      fields.back().back() == '\r') {
    fields.back().pop_back();
  }
  return fields;
}

bool ParseU64(const std::string& text, uint64_t* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const unsigned long long parsed = strtoull(text.c_str(), &end, 10);
  if (*end != '\0') return false;
  *value = parsed;
  return true;
}

int FindColumn(const std::vector<std::string>& header, const char* name) {
  const auto found = std::find(header.begin(), header.end(), name);
  return found == header.end() ? -1 :
      static_cast<int>(found - header.begin());
}

bool LoadRecipientSpans(uint64_t input_bytes,
    std::vector<RecipientSpan>* spans) {
  spans->clear();
  const uint64_t stop_offset =
      EnvironmentU64("FX4_WINNER_STOP_OFFSET", input_bytes);
  const char* path = getenv("FX4_WINNER_PACKS_CSV");
  recipient_spans_from_csv = path && *path;
  if (!recipient_spans_from_csv) {
    const uint64_t count =
        std::min(input_bytes, stop_offset) / DonorPlan::kChunkSize;
    for (uint64_t region = 0; region < count; ++region) {
      spans->push_back({
          static_cast<uint32_t>(region),
          region * DonorPlan::kChunkSize,
          DonorPlan::kChunkSize});
    }
  } else {
    std::ifstream input(path);
    std::string line;
    if (!input.is_open() || !std::getline(input, line)) return false;
    const auto header = Split(line, ',');
    const int id_column = FindColumn(header, "pack_id");
    const int offset_column = FindColumn(header, "post_r1_start");
    const int end_column = FindColumn(header, "post_r1_end");
    const int length_column = FindColumn(header, "post_r1_length");
    const int class_column = FindColumn(header, "stream_class");
    if (id_column < 0 || offset_column < 0 || end_column < 0 ||
        length_column < 0) {
      return false;
    }
    uint64_t previous_end = 0;
    while (std::getline(input, line)) {
      const std::vector<std::string> fields = Split(line, ',');
      if (fields.size() != header.size()) return false;
      uint64_t region = 0;
      uint64_t offset = 0;
      uint64_t end = 0;
      uint64_t length = 0;
      uint64_t stream_class =
          static_cast<uint8_t>(PostR1Experts::StreamClass::kMixed);
      if (!ParseU64(fields[id_column], &region) ||
          !ParseU64(fields[offset_column], &offset) ||
          !ParseU64(fields[end_column], &end) ||
          !ParseU64(fields[length_column], &length) ||
          (class_column >= 0 &&
           !ParseU64(fields[class_column], &stream_class)) ||
          region != spans->size() ||
          region > std::numeric_limits<uint32_t>::max() ||
          length == 0 || length > std::numeric_limits<uint32_t>::max() ||
          end < offset || end - offset != length ||
          offset < previous_end ||
          stream_class > static_cast<uint8_t>(
              PostR1Experts::StreamClass::kMixed)) {
        return false;
      }
      if (end > input_bytes || end > stop_offset) break;
      spans->push_back({
          static_cast<uint32_t>(region), offset,
          static_cast<uint32_t>(length),
          static_cast<uint8_t>(stream_class)});
      previous_end = end;
    }
  }

  recipient_offsets.assign(spans->size(), 0);
  for (const RecipientSpan& span : *spans) {
    recipient_offsets[span.region] = span.offset;
  }
  return !spans->empty();
}

bool LoadRankedPageCandidates(const std::vector<RecipientSpan>& spans) {
  ranked_page_candidates.assign(spans.size(), {});
  const char* path = getenv("FX4_WINNER_PAGE_CANDIDATES_CSV");
  if (!path || !*path) return true;
  std::ifstream input(path);
  std::string line;
  constexpr const char* expected =
      "pack_id,recipient_offset,recipient_length,candidate_rank,"
      "donor_offset,donor_length,phrase_score,signature_hits";
  constexpr const char* classified =
      "pack_id,recipient_offset,recipient_length,candidate_rank,"
      "donor_offset,donor_length,phrase_score,signature_hits,"
      "recipient_class,donor_class";
  if (!input.is_open() || !std::getline(input, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != expected && line != classified) {
    return false;
  }
  const size_t expected_fields = line == classified ? 10u : 8u;
  while (std::getline(input, line)) {
    const auto fields = Split(line, ',');
    if (fields.size() != expected_fields) return false;
    std::uint64_t region = 0;
    std::uint64_t recipient_offset = 0;
    std::uint64_t recipient_length = 0;
    std::uint64_t donor_offset = 0;
    std::uint64_t donor_length = 0;
    std::uint64_t score = 0;
    if (!ParseU64(fields[0], &region) ||
        !ParseU64(fields[1], &recipient_offset) ||
        !ParseU64(fields[2], &recipient_length) ||
        !ParseU64(fields[4], &donor_offset) ||
        !ParseU64(fields[5], &donor_length) ||
        !ParseU64(fields[6], &score) ||
        region >= spans.size() ||
        recipient_offset != spans[region].offset ||
        recipient_length != spans[region].length ||
        donor_offset > std::numeric_limits<std::uint32_t>::max() ||
        donor_length < 256 || donor_length > 65536 ||
        (donor_length & (donor_length - 1)) != 0 ||
        (donor_offset & (kAlignment - 1)) != 0 ||
        donor_offset + donor_length > recipient_offset ||
        score > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    ranked_page_candidates[region].push_back({
        static_cast<std::uint32_t>(donor_offset),
        static_cast<std::uint32_t>(donor_length),
        static_cast<std::uint32_t>(score)});
  }
  return true;
}

bool LoadRankedRecipientFilter(const std::vector<RecipientSpan>& spans) {
  ranked_recipient_filter.clear();
  ranked_recipient_filter_enabled = false;
  const char* path = getenv("FX4_WINNER_RECIPIENTS_CSV");
  if (!path || !*path) return true;
  std::ifstream input(path);
  std::string line;
  constexpr const char* expected =
      "rank,pack_id,recipient_offset,recipient_length,best_phrase_score,"
      "best_signature_hits,candidate_count";
  if (!input.is_open() || !std::getline(input, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != expected) {
    return false;
  }
  const std::uint64_t limit =
      EnvironmentU64("FX4_WINNER_RECIPIENT_LIMIT", 0);
  while (std::getline(input, line)) {
    const auto fields = Split(line, ',');
    if (fields.size() != 7) return false;
    std::uint64_t rank = 0;
    std::uint64_t region = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t score = 0;
    if (!ParseU64(fields[0], &rank) ||
        !ParseU64(fields[1], &region) ||
        !ParseU64(fields[2], &offset) ||
        !ParseU64(fields[3], &length) ||
        !ParseU64(fields[4], &score) ||
        region >= spans.size() || offset != spans[region].offset ||
        length != spans[region].length) {
      return false;
    }
    if (score == 0) continue;
    if (limit != 0 && ranked_recipient_filter.size() >= limit) break;
    ranked_recipient_filter.insert(static_cast<std::uint32_t>(region));
  }
  ranked_recipient_filter_enabled = true;
  return !ranked_recipient_filter.empty();
}

bool SearchRecipient(std::uint32_t region) {
  return !ranked_recipient_filter_enabled ||
      ranked_recipient_filter.count(region) != 0;
}

std::string SequenceText(const DonorSequence& sequence) {
  std::string text;
  for (size_t index = 0; index < sequence.size(); ++index) {
    if (index != 0) text.push_back(';');
    text += std::to_string(sequence[index].offset);
    text.push_back(':');
    text += std::to_string(sequence[index].length);
  }
  return text;
}

bool ParseSequence(const std::string& text, DonorSequence* sequence) {
  sequence->clear();
  if (text.empty() || text == "-") return true;
  for (const std::string& item : Split(text, ';')) {
    const size_t separator = item.find(':');
    if (separator == std::string::npos) return false;
    uint64_t offset = 0;
    uint64_t length = 0;
    if (!ParseU64(item.substr(0, separator), &offset) ||
        !ParseU64(item.substr(separator + 1), &length) ||
        offset > std::numeric_limits<uint32_t>::max() ||
        length > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    sequence->push_back({
        static_cast<uint32_t>(offset), static_cast<uint32_t>(length), 0});
  }
  return true;
}

std::string MarginalKey(uint32_t region, const DonorSequence& profile,
    const DonorWindow& removed) {
  return std::to_string(region) + "|" + SequenceText(profile) + "|" +
      std::to_string(removed.offset) + ":" +
      std::to_string(removed.length);
}

std::string TrialKey(uint32_t region, const DonorSequence& sequence) {
  return std::to_string(region) + "|" + SequenceText(sequence);
}

bool LoadTrials(const std::string& base,
    std::unordered_map<std::string, TrialRecord>* trials,
    uint64_t* next_trial_id) {
  std::ifstream input(TrialPath(base));
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line) || line + "\n" != kTrialHeader) {
    return false;
  }
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = Split(line, ',');
    if (fields.size() != 12) return false;
    uint64_t region = 0;
    uint64_t recipient_offset = 0;
    uint64_t trial_id = 0;
    uint64_t payload = 0;
    if (!ParseU64(fields[0], &region) ||
        !ParseU64(fields[1], &recipient_offset) ||
        !ParseU64(fields[2], &trial_id) ||
        !ParseU64(fields[8], &payload) || region > 0xffff ||
        recipient_offset != RecipientOffset(static_cast<uint32_t>(region))) {
      return false;
    }
    DonorSequence sequence;
    if (!ParseSequence(fields[6], &sequence)) return false;
    (*trials)[TrialKey(static_cast<uint32_t>(region), sequence)] = {
        fields[11] != "error", payload};
    *next_trial_id = std::max(*next_trial_id, trial_id + 1);
  }
  return true;
}

bool LoadMarginals(const std::string& base,
    std::unordered_set<std::string>* completed) {
  std::ifstream input(MarginalPath(base));
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line) || line + "\n" != kMarginalHeader) {
    return false;
  }
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = Split(line, ',');
    if (fields.size() != 9) return false;
    uint64_t region = 0;
    uint64_t recipient_offset = 0;
    if (!ParseU64(fields[0], &region) ||
        !ParseU64(fields[1], &recipient_offset) ||
        region > std::numeric_limits<uint32_t>::max() ||
        recipient_offset != RecipientOffset(static_cast<uint32_t>(region))) {
      return false;
    }
    DonorSequence profile;
    DonorSequence removed;
    if (!ParseSequence(fields[2], &profile) ||
        !ParseSequence(fields[3], &removed) || removed.size() != 1) {
      return false;
    }
    completed->insert(MarginalKey(
        static_cast<uint32_t>(region), profile, removed.front()));
  }
  return true;
}
bool LoadSelections(const std::string& base,
    std::vector<SelectionRecord>* selections) {
  std::ifstream input(SelectionPath(base));
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line) || line + "\n" != kSelectionHeader) {
    return false;
  }
  while (std::getline(input, line)) {
    const std::vector<std::string> fields = Split(line, ',');
    if (fields.size() != 10) return false;
    uint64_t region = 0;
    uint64_t recipient_offset = 0;
    uint64_t baseline = 0;
    uint64_t selected = 0;
    if (!ParseU64(fields[0], &region) ||
        !ParseU64(fields[1], &recipient_offset) ||
        !ParseU64(fields[2], &baseline) ||
        !ParseU64(fields[3], &selected) || region >= selections->size() ||
        recipient_offset != RecipientOffset(static_cast<uint32_t>(region))) {
      return false;
    }
    SelectionRecord record;
    record.valid = true;
    record.baseline_bytes = baseline;
    record.selected_bytes = selected;
    if (!ParseSequence(fields[7], &record.sequence)) return false;
    (*selections)[static_cast<size_t>(region)] = std::move(record);
  }
  return true;
}

bool WriteStatus(const std::string& base, const char* phase,
    uint32_t region, uint64_t new_trials, const DonorSequence& sequence,
    int64_t gain) {
  const std::string path = StatusPath(base);
  const std::string temporary = path + ".tmp";
  FILE* output = fopen(temporary.c_str(), "wb");
  if (!output) return false;
  fprintf(output,
      "phase=%s\nrecipient_region=%u\nrecipient_offset=%" PRIu64
      "\nnew_trials_this_run=%" PRIu64 "\ngain_bytes=%" PRId64
      "\ndonors=%s\npid=%ld\n",
      phase, region,
      RecipientOffset(region),
      new_trials, gain,
      sequence.empty() ? "-" : SequenceText(sequence).c_str(),
      static_cast<long>(getpid()));
  const bool ok = SyncFile(output);
  fclose(output);
  return ok && rename(temporary.c_str(), path.c_str()) == 0;
}

bool AppendTrial(const std::string& base, uint32_t region,
    uint64_t trial_id, const char* stage, const DonorSequence& sequence,
    uint64_t baseline_bytes, uint64_t payload_bytes,
    int64_t marginal_gain, bool ok, uint32_t near_loss_bytes,
    uint64_t new_trials) {
  const int64_t gain = ok
      ? static_cast<int64_t>(baseline_bytes) -
          static_cast<int64_t>(payload_bytes)
      : 0;
  const char* status = !ok ? "error" : gain > 0 ? "winner" :
      gain >= -static_cast<int64_t>(near_loss_bytes) ? "near" : "loss";
  char row[2048] = {};
  const std::string donors = sequence.empty() ? "-" : SequenceText(sequence);
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%" PRIu64 ",%s,%zu,%zu,%s,%" PRIu64
      ",%" PRIu64 ",%" PRId64 ",%" PRId64 ",%s\n",
      region, RecipientOffset(region),
      trial_id, stage, sequence.size(), sequence.size(), donors.c_str(),
      baseline_bytes, payload_bytes, gain, marginal_gain, status);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(row)) return false;
  const std::string text(row, static_cast<size_t>(length));
  if (!AppendRow(TrialPath(base), text)) return false;
  if (ok && gain >= -static_cast<int64_t>(near_loss_bytes) &&
      !AppendRow(NearPath(base), text)) {
    return false;
  }
  if (ok && gain > 0 && !AppendRow(WinnersPath(base), text)) return false;
  return WriteStatus(base, stage, region, new_trials, sequence, gain);
}

bool AppendMarginal(const std::string& base, uint32_t region,
    const ScoredSequence& full, const DonorWindow& removed,
    uint64_t without_payload) {
  const int64_t marginal =
      static_cast<int64_t>(without_payload) -
      static_cast<int64_t>(full.payload_bytes);
  const char* status = marginal > 0 ? "helps" :
      (marginal < 0 ? "hurts" : "neutral");
  const std::string profile = SequenceText(full.sequence);
  const std::string removed_text = SequenceText(DonorSequence{removed});
  char row[2048] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%s,%s,%zu,%" PRIu64 ",%" PRIu64
      ",%" PRId64 ",%s\n",
      region, RecipientOffset(region), profile.c_str(),
      removed_text.c_str(), full.sequence.size(), full.payload_bytes,
      without_payload, marginal, status);
  return length > 0 && static_cast<size_t>(length) < sizeof(row) &&
      AppendRow(MarginalPath(base),
          std::string(row, static_cast<size_t>(length)));
}
bool AppendSelection(const std::string& base, uint32_t region,
    const SelectionRecord& selection, uint32_t candidate_count,
    uint64_t exact_trials,
    uint64_t metadata_override = std::numeric_limits<uint64_t>::max()) {
  const int64_t gain = static_cast<int64_t>(selection.baseline_bytes) -
      static_cast<int64_t>(selection.selected_bytes);
  const uint64_t metadata_bytes = selection.sequence.empty() ? 0 :
      (metadata_override == std::numeric_limits<uint64_t>::max()
          ? DonorPlanMetadataCost(region, selection.sequence)
          : metadata_override);
  const int64_t metadata = static_cast<int64_t>(metadata_bytes);
  const std::string donors = selection.sequence.empty()
      ? "-" : SequenceText(selection.sequence);
  char row[2048] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRId64
      ",%" PRId64 ",%zu,%s,%u,%" PRIu64 "\n",
      region, RecipientOffset(region),
      selection.baseline_bytes, selection.selected_bytes, gain,
      gain - metadata, selection.sequence.size(), donors.c_str(),
      candidate_count, exact_trials);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(row) ||
      !AppendRow(SelectionPath(base),
          std::string(row, static_cast<size_t>(length)))) {
    return false;
  }
  // Phase 2 authoritative cost ledger: span_cost/copy_cost are 0 until the
  // causal span selector (Phase 11) and COPY events (Phase 13) exist.
  return AppendCostAccounting(base, region, selection.baseline_bytes,
      selection.selected_bytes, metadata_bytes, 0, 0);
}

bool AppendSpans(const std::string& base, uint32_t region,
    const ProbeMessage& baseline, uint32_t top_spans) {
  std::vector<uint32_t> order(baseline.span_count);
  for (uint32_t index = 0; index < baseline.span_count; ++index) {
    order[index] = index;
  }
  std::sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
    if (baseline.span_bytes[left] != baseline.span_bytes[right]) {
      return baseline.span_bytes[left] > baseline.span_bytes[right];
    }
    return left < right;
  });
  order.resize(std::min<size_t>(order.size(), top_spans));
  for (size_t rank = 0; rank < order.size(); ++rank) {
    const uint64_t span_offset =
        RecipientOffset(region) +
        static_cast<uint64_t>(order[rank]) * kLossSpanBytes;
    char row[256] = {};
    const int length = snprintf(row, sizeof(row),
        "%u,%" PRIu64 ",%zu,%" PRIu64 ",%u\n",
        region, RecipientOffset(region),
        rank, span_offset, baseline.span_bytes[order[rank]]);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(row) ||
        !AppendRow(SpanPath(base),
            std::string(row, static_cast<size_t>(length)))) {
      return false;
    }
  }
  return true;
}

bool WriteCandidateRows(const std::string& base, uint32_t region,
    const std::vector<DonorWindow>& candidates) {
  for (size_t rank = 0; rank < candidates.size(); ++rank) {
    char row[320] = {};
    const char* source = candidates[rank].score ==
            std::numeric_limits<uint32_t>::max()
        ? "plan"
        : (candidates[rank].score == 0 ? "recent_fallback" : "phrase_token");
    const int length = snprintf(row, sizeof(row),
        "%u,%" PRIu64 ",%zu,%u,%u,%u,%s\n",
        region, RecipientOffset(region),
        rank, candidates[rank].offset, candidates[rank].length,
        candidates[rank].score, source);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(row) ||
        !AppendRow(CandidatePath(base),
            std::string(row, static_cast<size_t>(length)))) {
      return false;
    }
  }
  return true;
}

bool ReadWindow(int input_fd, uint32_t offset, uint32_t length,
    std::vector<uint8_t>* bytes) {
  bytes->resize(length);
  size_t complete = 0;
  while (complete < length) {
    const ssize_t count = pread(input_fd, bytes->data() + complete,
        length - complete, static_cast<off_t>(offset + complete));
    if (count < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (count == 0) return false;
    complete += static_cast<size_t>(count);
  }
  return true;
}

bool LoadDonorProfile(int input_fd, const DonorSequence& sequence,
    std::vector<uint8_t>* profile, std::vector<uint32_t>* segment_lengths) {
  profile->clear();
  segment_lengths->clear();
  std::vector<uint8_t> window;
  for (const DonorWindow& donor : sequence) {
    if (!ReadWindow(input_fd, donor.offset, donor.length, &window)) {
      return false;
    }
    profile->insert(profile->end(), window.begin(), window.end());
    segment_lengths->push_back(donor.length);
  }
  return true;
}

bool EncodeRegion(const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, DonorPlan* donor_plan, ProbeMessage* profile) {
  uint64_t span_start = encoder->OutputSize();
  size_t span_index = 0;
  for (size_t index = 0; index < size; ++index) {
    const uint8_t byte = static_cast<uint8_t>(bytes[index]);
    for (int bit = 7; bit >= 0; --bit) {
      encoder->Encode((byte >> bit) & 1);
    }
    if (donor_plan) donor_plan->CaptureByte(position + index, byte);
    if (profile &&
        ((index + 1) % kLossSpanBytes == 0 || index + 1 == size)) {
      if (span_index < profile->span_bytes.size()) {
        const uint64_t now = encoder->OutputSize();
        profile->span_bytes[span_index++] =
            static_cast<uint32_t>(now - span_start);
        span_start = now;
      }
    }
  }
  if (profile) profile->span_count = static_cast<uint16_t>(span_index);
  return true;
}

bool ProbeSequence(const DonorSequence& sequence, const char* bytes,
    size_t size, uint64_t position, Encoder* encoder, Predictor* predictor,
    int input_fd, bool collect_spans, ProbeMessage* result) {
  int message_pipe[2] = {-1, -1};
  if (pipe(message_pipe) != 0) return false;
  const uint64_t start_size = encoder->ProjectedFinalOutputSize();
  const pid_t probe = fork();
  if (probe < 0) {
    close(message_pipe[0]);
    close(message_pipe[1]);
    return false;
  }
  if (probe == 0) {
    close(message_pipe[0]);
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    SetCpuFromEnvironment("FX4_DONOR_TRIAL_CPU");
    encoder->SetCountOnly(true);
    ProbeMessage message;
    bool profile_ok = true;
    if (!sequence.empty()) {
      std::vector<uint8_t> profile;
      std::vector<uint32_t> segment_lengths;
      profile_ok = LoadDonorProfile(
          input_fd, sequence, &profile, &segment_lengths);
      if (profile_ok) {
        const std::uint32_t mask = PostR1Experts::kDonorProfile;
        predictor->EnablePostR1Portfolio(mask);
        predictor->SetPostR1DonorProfile(profile, segment_lengths);
        predictor->SetPostR1Span(position, mask,
            current_recipient_stream_class, current_recipient_stream_class, 0);
      }
    }
    message.ok = profile_ok && EncodeRegion(bytes, size, position, encoder,
        nullptr, collect_spans ? &message : nullptr) ? 1u : 0u;
    message.payload_bytes =
        encoder->ProjectedFinalOutputSize() - start_size;
    const bool sent = WriteAll(message_pipe[1], &message, sizeof(message));
    close(message_pipe[1]);
    _exit(sent && message.ok ? 0 : 2);
  }

  close(message_pipe[1]);
  ProbeMessage message;
  const bool read_ok = ReadAll(message_pipe[0], &message, sizeof(message));
  close(message_pipe[0]);
  int status = 0;
  const bool waited = waitpid(probe, &status, 0) == probe;
  if (!read_ok || !waited || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0 || !message.ok) {
    result->ok = 0;
    result->payload_bytes = 0;
    return true;
  }
  *result = message;
  return true;
}

uint64_t MixHash(uint64_t value) {
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

uint64_t PhraseHash(const uint8_t* bytes) {
  uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
  for (size_t index = 0; index < 16; ++index) {
    hash = MixHash(hash ^
        (static_cast<uint64_t>(bytes[index]) + index * 257u));
  }
  return hash;
}

uint64_t TokenHash(const uint8_t* bytes) {
  uint64_t hash = UINT64_C(0xd6e8feb86659fd93);
  uint32_t useful = 0;
  for (size_t index = 0; index < kAlignment; ++index) {
    const uint8_t byte = bytes[index];
    uint8_t token = 0;
    if (byte >= 128) token = byte;
    else if (byte >= '0' && byte <= '9') token = '0';
    else if (byte == '<' || byte == '>' || byte == '[' || byte == ']' ||
        byte == '{' || byte == '}' || byte == '|' || byte == '=' ||
        byte == '&' || byte == ';') {
      token = byte;
    }
    if (token != 0) {
      hash = MixHash(hash ^
          (static_cast<uint64_t>(token) + useful * 65537u));
      ++useful;
    }
  }
  return useful >= 8 ? hash ^ UINT64_C(0xa5a5a5a5a5a5a5a5) : 0;
}

class CausalPhraseIndex {
 public:
  void AddRegion(const char* input, size_t size, uint64_t position) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(input);
    for (size_t local = 0; local + kAlignment <= size;
         local += kAlignment) {
      std::array<uint64_t, 2> minima{
          std::numeric_limits<uint64_t>::max(),
          std::numeric_limits<uint64_t>::max()};
      for (size_t phrase = 0; phrase < kAlignment; phrase += 16) {
        const uint64_t hash = PhraseHash(bytes + local + phrase);
        if (hash < minima[0]) {
          minima[1] = minima[0];
          minima[0] = hash;
        } else if (hash < minima[1] && hash != minima[0]) {
          minima[1] = hash;
        }
      }
      const uint32_t offset = static_cast<uint32_t>(position + local);
      Add(minima[0], offset);
      if (minima[1] != std::numeric_limits<uint64_t>::max()) {
        Add(minima[1], offset);
      }
      const uint64_t token = TokenHash(bytes + local);
      if (token != 0) Add(token, offset);
    }
  }

  std::vector<std::pair<uint32_t, uint32_t>> Find(
      const char* input, size_t size, const ProbeMessage& baseline,
      uint32_t top_spans, uint64_t recipient_offset, uint32_t limit) const {
    std::vector<uint32_t> spans(baseline.span_count);
    for (uint32_t index = 0; index < baseline.span_count; ++index) {
      spans[index] = index;
    }
    std::sort(spans.begin(), spans.end(), [&](uint32_t left, uint32_t right) {
      return baseline.span_bytes[left] != baseline.span_bytes[right]
          ? baseline.span_bytes[left] > baseline.span_bytes[right]
          : left < right;
    });
    spans.resize(std::min<size_t>(spans.size(), top_spans));

    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(input);
    std::unordered_map<uint32_t, uint32_t> scores;
    for (uint32_t span : spans) {
      const size_t start = static_cast<size_t>(span) * kLossSpanBytes;
      if (start >= size) continue;
      const size_t span_end = std::min(size, start + kLossSpanBytes);
      for (size_t local = start;
           local + kAlignment <= span_end;
           local += kAlignment) {
        for (size_t phrase = 0; phrase < kAlignment; phrase += 16) {
          Score(PhraseHash(bytes + local + phrase), recipient_offset,
              2, &scores);
        }
        const uint64_t token = TokenHash(bytes + local);
        if (token != 0) Score(token, recipient_offset, 3, &scores);
      }
    }
    std::vector<std::pair<uint32_t, uint32_t>> result(
        scores.begin(), scores.end());
    std::sort(result.begin(), result.end(),
        [](const auto& left, const auto& right) {
          return left.second != right.second
              ? left.second > right.second
              : left.first < right.first;
        });
    if (result.size() > limit) result.resize(limit);
    return result;
  }

 private:
  struct Posting {
    std::array<uint32_t, kPostingCapacity> offsets{};
    uint32_t seen = 0;
    uint8_t count = 0;
  };

  void Add(uint64_t hash, uint32_t offset) {
    Posting& posting = postings_[hash];
    for (uint8_t index = 0; index < posting.count; ++index) {
      if (posting.offsets[index] == offset) return;
    }
    if (posting.count < posting.offsets.size()) {
      posting.offsets[posting.count++] = offset;
    } else {
      posting.offsets[2 + (posting.seen % (kPostingCapacity - 2))] = offset;
    }
    ++posting.seen;
  }

  void Score(uint64_t hash, uint64_t recipient_offset, uint32_t amount,
      std::unordered_map<uint32_t, uint32_t>* scores) const {
    const auto found = postings_.find(hash);
    if (found == postings_.end()) return;
    for (uint8_t index = 0; index < found->second.count; ++index) {
      const uint32_t offset = found->second.offsets[index];
      if (static_cast<uint64_t>(offset) + kAlignment <= recipient_offset) {
        (*scores)[offset] += amount;
      }
    }
  }

  std::unordered_map<uint64_t, Posting> postings_;
};

bool SameWindow(const DonorWindow& left, const DonorWindow& right) {
  return left.offset == right.offset && left.length == right.length;
}

bool ContainsWindow(
    const DonorSequence& sequence, const DonorWindow& candidate) {
  return std::any_of(sequence.begin(), sequence.end(),
      [&](const DonorWindow& donor) { return SameWindow(donor, candidate); });
}

bool ValidWindow(const DonorWindow& donor, uint64_t recipient_offset) {
  return (donor.offset & (kAlignment - 1)) == 0 &&
      donor.length >= 256 && donor.length <= 65536 &&
      (donor.length & (donor.length - 1)) == 0 &&
      static_cast<uint64_t>(donor.offset) + donor.length <= recipient_offset;
}

void AddUniqueWindow(const DonorWindow& candidate,
    uint64_t recipient_offset, std::vector<DonorWindow>* windows) {
  if (!ValidWindow(candidate, recipient_offset)) return;
  const auto found = std::find_if(windows->begin(), windows->end(),
      [&](const DonorWindow& donor) { return SameWindow(donor, candidate); });
  if (found == windows->end()) windows->push_back(candidate);
}

uint64_t TotalBytes(uint32_t region, const ScoredSequence& scored) {
  return scored.payload_bytes +
      DonorPlanMetadataCost(region, scored.sequence);
}

uint64_t TotalBytes(uint32_t region, const SelectionRecord& selection) {
  return selection.selected_bytes +
      DonorPlanMetadataCost(region, selection.sequence);
}

bool BeatsSelection(uint32_t region,
    const ScoredSequence& scored, const SelectionRecord& selection) {
  return TotalBytes(region, scored) < TotalBytes(region, selection);
}

void SortScored(std::vector<ScoredSequence>* sequences) {
  std::sort(sequences->begin(), sequences->end(),
      [](const ScoredSequence& left, const ScoredSequence& right) {
        if (left.payload_bytes != right.payload_bytes) {
          return left.payload_bytes < right.payload_bytes;
        }
        return SequenceText(left.sequence) < SequenceText(right.sequence);
      });
}

bool FindBestRawSequence(uint32_t region, uint64_t baseline_bytes,
    const std::unordered_map<std::string, TrialRecord>& trials,
    ScoredSequence* best) {
  best->sequence.clear();
  best->payload_bytes = baseline_bytes;
  const std::string prefix = std::to_string(region) + "|";
  for (const auto& entry : trials) {
    if (!entry.second.ok || entry.second.payload_bytes >= baseline_bytes ||
        entry.first.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    DonorSequence sequence;
    if (!ParseSequence(entry.first.substr(prefix.size()), &sequence) ||
        sequence.empty()) {
      continue;
    }
    const bool better = best->sequence.empty() ||
        entry.second.payload_bytes < best->payload_bytes ||
        (entry.second.payload_bytes == best->payload_bytes &&
         SequenceText(sequence) < SequenceText(best->sequence));
    if (!better) continue;
    best->sequence = std::move(sequence);
    best->payload_bytes = entry.second.payload_bytes;
  }
  return !best->sequence.empty();
}
bool Evaluate(uint32_t region, const char* stage,
    const DonorSequence& sequence, uint64_t baseline_bytes,
    uint64_t parent_bytes, const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, Predictor* predictor, int input_fd,
    const SearchConfig& config, const std::string& ledger_base,
    std::unordered_map<std::string, TrialRecord>* trials,
    uint64_t* next_trial_id, uint64_t* new_trials, bool* paused,
    ScoredSequence* scored) {
  const std::string key = TrialKey(region, sequence);
  const auto found = trials->find(key);
  if (found != trials->end()) {
    if (!found->second.ok) return false;
    *scored = {sequence, found->second.payload_bytes};
    return true;
  }
  if (config.max_new_trials != 0 &&
      *new_trials >= config.max_new_trials) {
    *paused = true;
    return false;
  }

  ProbeMessage probe;
  if (!ProbeSequence(sequence, bytes, size, position, encoder, predictor,
      input_fd, false, &probe)) {
    return false;
  }
  ++*new_trials;
  const int64_t marginal = probe.ok
      ? static_cast<int64_t>(parent_bytes) -
          static_cast<int64_t>(probe.payload_bytes)
      : 0;
  if (!AppendTrial(ledger_base, region, (*next_trial_id)++, stage, sequence,
      baseline_bytes, probe.payload_bytes, marginal, probe.ok != 0,
      config.near_loss_bytes, *new_trials)) {
    return false;
  }
  (*trials)[key] = {probe.ok != 0, probe.payload_bytes};
  if (!probe.ok) return false;
  *scored = {sequence, probe.payload_bytes};
  return true;
}

enum class SearchResult {
  kFailed,
  kPaused,
  kComplete,
  kCompleteAndEncoded,
};

struct InPlaceTrial {
  DonorSequence sequence;
  std::unique_ptr<PostR1Experts> expert;
  std::unique_ptr<Encoder> encoder;
};

uint32_t QuickMetadataBytes(
    const SearchConfig& config, const DonorSequence& sequence) {
  // Version-7 plans delta-code donor offsets. The first donor is covered by
  // the configured single-span estimate; each additional donor normally adds
  // a one-byte length and a one-to-three-byte offset delta. Exact winners are
  // still rebuilt and measured with DonorPlan::SerializedArchiveSize().
  if (sequence.empty()) return 0;
  if (sequence.size() == 1) return config.quick_metadata_bytes;
  return config.quick_metadata_bytes +
      static_cast<uint32_t>((sequence.size() - 1) * 4u);
}

void AddUniqueSequence(const DonorSequence& sequence,
    std::unordered_set<std::string>* seen,
    std::vector<DonorSequence>* sequences) {
  if (sequence.empty() || sequence.size() > 8) return;
  const std::string key = SequenceText(sequence);
  if (seen->insert(key).second) sequences->push_back(sequence);
}

SearchResult SearchQuickSinglesInPlace(uint32_t region,
    const char* bytes, size_t size, uint64_t position, Encoder* encoder,
    Predictor* predictor, int input_fd, DonorPlan* donor_plan,
    const SearchConfig& config, const std::string& ledger_base,
    std::unordered_map<std::string, TrialRecord>* trials,
    uint64_t* next_trial_id, uint64_t* new_trials,
    SelectionRecord* selection) {
  const uint64_t region_trial_start = *new_trials;
  std::vector<DonorWindow> initial;
  if (region < ranked_page_candidates.size()) {
    for (const DonorWindow& candidate : ranked_page_candidates[region]) {
      AddUniqueWindow(candidate, position, &initial);
    }
  }
  if (!recipient_spans_from_csv) {
    for (const DonorPlan::CandidateSpec& candidate :
         donor_plan->CandidateSpecs(region)) {
      AddUniqueWindow({candidate.donor_offset, candidate.length,
          std::numeric_limits<uint32_t>::max()}, position, &initial);
    }
  }
  for (uint32_t rank = 1;
       initial.size() < config.planned_donors && rank <= 64; ++rank) {
    const uint64_t gap = static_cast<uint64_t>(rank) * 8192u;
    if (position < gap + kInitialLength) break;
    const uint32_t offset = static_cast<uint32_t>(
        (position - gap) & ~(static_cast<uint64_t>(kAlignment) - 1u));
    AddUniqueWindow({offset, kInitialLength, 0}, position, &initial);
  }
  std::stable_sort(initial.begin(), initial.end(),
      [](const DonorWindow& left, const DonorWindow& right) {
        return left.score != right.score
            ? left.score > right.score : left.offset < right.offset;
      });
  const size_t planned_count = region < ranked_page_candidates.size()
      ? ranked_page_candidates[region].size() : 0;
  const size_t initial_limit = std::max<size_t>(
      config.planned_donors, planned_count);
  if (initial.size() > initial_limit) initial.resize(initial_limit);
  if (initial.empty() || !WriteCandidateRows(ledger_base, region, initial)) {
    return SearchResult::kFailed;
  }

  std::vector<DonorSequence> sequences;
  std::unordered_set<std::string> seen_sequences;
  for (const DonorWindow& donor : initial) {
    AddUniqueSequence(DonorSequence{donor}, &seen_sequences, &sequences);
  }

  const size_t pair_count = std::min<size_t>(
      config.quick_pair_candidates, initial.size());
  for (size_t left = 0; left < pair_count; ++left) {
    for (size_t right = 0; right < pair_count; ++right) {
      if (left == right) continue;
      AddUniqueSequence(DonorSequence{initial[left], initial[right]},
          &seen_sequences, &sequences);
    }
  }

  const size_t prefix_depth = std::min<size_t>(
      config.quick_prefix_depth, initial.size());
  for (size_t depth = 2; depth <= prefix_depth; ++depth) {
    AddUniqueSequence(DonorSequence(initial.begin(), initial.begin() + depth),
        &seen_sequences, &sequences);
    if (config.quick_reverse_prefixes) {
      DonorSequence reverse(initial.begin(), initial.begin() + depth);
      std::reverse(reverse.begin(), reverse.end());
      AddUniqueSequence(reverse, &seen_sequences, &sequences);
    }
  }

  std::vector<InPlaceTrial> branches;
  branches.reserve(sequences.size());
  for (const DonorSequence& sequence : sequences) {
    std::vector<uint8_t> profile;
    std::vector<uint32_t> segment_lengths;
    if (!LoadDonorProfile(
        input_fd, sequence, &profile, &segment_lengths)) {
      return SearchResult::kFailed;
    }
    std::unique_ptr<PostR1Experts> expert(new PostR1Experts());
    const std::uint32_t expert_mask = PostR1Experts::kOracle |
        PostR1Experts::kDonorProfile |
        (config.quick_context_mixer ? PostR1Experts::kContextMixer : 0u);
    expert->EnablePortfolio(expert_mask);
    expert->SetDonorProfile(profile, segment_lengths);
    const std::uint8_t profile_id = static_cast<std::uint8_t>(
        (config.quick_donor_strength & 3u) << 6);
    expert->SetSpan(position, expert_mask,
        static_cast<PostR1Experts::StreamClass>(
            current_recipient_stream_class),
        profile_id, 0);
    std::unique_ptr<Encoder> branch(new Encoder(encoder->Clone()));
    branch->SetCountOnly(true);
    branches.push_back(
        {std::move(sequence), std::move(expert), std::move(branch)});
  }

  const uint64_t start_size = encoder->ProjectedFinalOutputSize();
  uint64_t span_start = encoder->OutputSize();
  ProbeMessage baseline;
  baseline.ok = 1;
  size_t span_index = 0;
  for (size_t index = 0; index < size; ++index) {
    const uint8_t byte = static_cast<uint8_t>(bytes[index]);
    unsigned int prefix = 1;
    for (unsigned int bit_position = 0; bit_position < 8; ++bit_position) {
      const int bit = (byte >> (7 - bit_position)) & 1;
      const float base_probability = predictor->Predict();
      for (InPlaceTrial& branch : branches) {
        predictor->SetPostR1BranchSignals(branch.expert.get());
        const float probability = branch.expert->Predict(
            base_probability, prefix, bit_position);
        branch.encoder->EncodeRawBit(
            bit, Encoder::DiscretizeProbability(probability));
      }
      encoder->EncodeRawBit(
          bit, Encoder::DiscretizeProbability(base_probability));
      for (InPlaceTrial& branch : branches) {
        branch.expert->Perceive(bit);
      }
      predictor->Perceive(bit);
      prefix = (prefix << 1) | static_cast<unsigned int>(bit);
    }
    for (InPlaceTrial& branch : branches) {
      branch.expert->ByteUpdate(byte);
    }
    if (donor_plan) donor_plan->CaptureByte(position + index, byte);
    if ((index + 1) % kLossSpanBytes == 0 || index + 1 == size) {
      if (span_index < baseline.span_bytes.size()) {
        const uint64_t now = encoder->OutputSize();
        baseline.span_bytes[span_index++] =
            static_cast<uint32_t>(now - span_start);
        span_start = now;
      }
    }
  }
  baseline.span_count = static_cast<uint16_t>(span_index);
  baseline.payload_bytes =
      encoder->ProjectedFinalOutputSize() - start_size;
  if (!AppendSpans(ledger_base, region, baseline, config.top_spans)) {
    return SearchResult::kFailed;
  }

  selection->valid = true;
  selection->baseline_bytes = baseline.payload_bytes;
  selection->selected_bytes = baseline.payload_bytes;
  selection->sequence.clear();
  for (InPlaceTrial& branch : branches) {
    const uint64_t payload_bytes =
        branch.encoder->ProjectedFinalOutputSize() - start_size;
    ++*new_trials;
    const int64_t marginal =
        static_cast<int64_t>(baseline.payload_bytes) -
        static_cast<int64_t>(payload_bytes);
    if (!AppendTrial(ledger_base, region, (*next_trial_id)++,
        "single_inplace", branch.sequence, baseline.payload_bytes,
        payload_bytes, marginal, true, config.near_loss_bytes,
        *new_trials)) {
      return SearchResult::kFailed;
    }
    (*trials)[TrialKey(region, branch.sequence)] = {true, payload_bytes};
    const uint64_t candidate_total = payload_bytes +
        QuickMetadataBytes(config, branch.sequence);
    const uint64_t selected_total = selection->selected_bytes +
        QuickMetadataBytes(config, selection->sequence);
    if (candidate_total < selected_total) {
      selection->selected_bytes = payload_bytes;
      selection->sequence = branch.sequence;
    }
  }
  if (!AppendSelection(ledger_base, region, *selection,
      static_cast<uint32_t>(sequences.size()),
      *new_trials - region_trial_start,
      QuickMetadataBytes(config, selection->sequence))) {
    return SearchResult::kFailed;
  }
  return SearchResult::kCompleteAndEncoded;
}

SearchResult AttributeDonorMarginals(uint32_t region,
    const ScoredSequence& full, uint64_t baseline_bytes,
    const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, Predictor* predictor, int input_fd,
    const SearchConfig& config, const std::string& ledger_base,
    std::unordered_map<std::string, TrialRecord>* trials,
    std::unordered_set<std::string>* completed_marginals,
    uint64_t* next_trial_id, uint64_t* new_trials) {
  for (size_t removed_index = 0;
       removed_index < full.sequence.size(); ++removed_index) {
    const DonorWindow removed = full.sequence[removed_index];
    const std::string marginal_key =
        MarginalKey(region, full.sequence, removed);
    if (completed_marginals->count(marginal_key) != 0) continue;

    DonorSequence reduced = full.sequence;
    reduced.erase(reduced.begin() + removed_index);
    uint64_t without_payload = baseline_bytes;
    if (!reduced.empty()) {
      const std::string trial_key = TrialKey(region, reduced);
      const auto found = trials->find(trial_key);
      if (found != trials->end()) {
        if (!found->second.ok) return SearchResult::kFailed;
        without_payload = found->second.payload_bytes;
      } else {
        if (config.max_new_trials != 0 &&
            *new_trials >= config.max_new_trials) {
          return SearchResult::kPaused;
        }
        ProbeMessage probe;
        if (!ProbeSequence(reduced, bytes, size, position, encoder,
            predictor, input_fd, false, &probe)) {
          return SearchResult::kFailed;
        }
        ++*new_trials;
        const int64_t marginal = probe.ok
            ? static_cast<int64_t>(probe.payload_bytes) -
                static_cast<int64_t>(full.payload_bytes)
            : 0;
        if (!AppendTrial(ledger_base, region, (*next_trial_id)++,
            "leave_one_out", reduced, baseline_bytes, probe.payload_bytes,
            marginal, probe.ok != 0, config.near_loss_bytes, *new_trials)) {
          return SearchResult::kFailed;
        }
        (*trials)[trial_key] = {probe.ok != 0, probe.payload_bytes};
        if (!probe.ok) return SearchResult::kFailed;
        without_payload = probe.payload_bytes;
      }
    }

    if (!AppendMarginal(
        ledger_base, region, full, removed, without_payload)) {
      return SearchResult::kFailed;
    }
    completed_marginals->insert(marginal_key);
  }
  return SearchResult::kComplete;
}
SearchResult SearchRegion(uint32_t region, const char* bytes, size_t size,
    uint64_t position, Encoder* encoder, Predictor* predictor, int input_fd,
    DonorPlan* donor_plan, const CausalPhraseIndex& phrase_index,
    const SearchConfig& config, const std::string& ledger_base,
    std::unordered_map<std::string, TrialRecord>* trials,
    uint64_t* next_trial_id, uint64_t* new_trials,
    SelectionRecord* selection) {
  if (config.quick_singles) {
    return SearchQuickSinglesInPlace(region, bytes, size, position, encoder,
        predictor, input_fd, donor_plan, config, ledger_base, trials,
        next_trial_id, new_trials, selection);
  }

  const uint64_t region_trial_start = *new_trials;
  ProbeMessage baseline;
  DonorSequence empty;
  if (!ProbeSequence(empty, bytes, size, position, encoder, predictor,
      input_fd, true, &baseline) || !baseline.ok ||
      !AppendSpans(ledger_base, region, baseline, config.top_spans)) {
    return SearchResult::kFailed;
  }

  selection->valid = true;
  selection->baseline_bytes = baseline.payload_bytes;
  selection->selected_bytes = baseline.payload_bytes;
  selection->sequence.clear();


  std::vector<DonorWindow> initial;
  if (region < ranked_page_candidates.size()) {
    for (const DonorWindow& candidate : ranked_page_candidates[region]) {
      AddUniqueWindow(candidate, position, &initial);
    }
  }
  if (!recipient_spans_from_csv) {
    for (const DonorPlan::CandidateSpec& candidate :
         donor_plan->CandidateSpecs(region)) {
      AddUniqueWindow({candidate.donor_offset, candidate.length,
          std::numeric_limits<uint32_t>::max()}, position, &initial);
    }
  }
  const auto matches = phrase_index.Find(bytes, size, baseline,
      config.top_spans, position, config.candidate_offsets);
  for (const auto& match : matches) {
    AddUniqueWindow({match.first, kInitialLength, match.second},
        position, &initial);
  }
  // Phrase overlap can be empty in a difficult region. Still evaluate a
  // bounded causal profile instead of silently treating that region as done.
  for (uint32_t rank = 1;
       initial.size() < config.planned_donors && rank <= 64; ++rank) {
    const uint64_t gap = static_cast<uint64_t>(rank) * 8192u;
    if (position < gap + kInitialLength) break;
    const uint32_t offset = static_cast<uint32_t>(
        (position - gap) & ~(static_cast<uint64_t>(kAlignment) - 1u));
    AddUniqueWindow({offset, kInitialLength, 0}, position, &initial);
  }
  std::stable_sort(initial.begin(), initial.end(),
      [](const DonorWindow& left, const DonorWindow& right) {
        return left.score != right.score
            ? left.score > right.score
            : left.offset < right.offset;
      });
  const size_t planned_count = region < ranked_page_candidates.size()
      ? ranked_page_candidates[region].size() : 0;
  const size_t initial_limit = config.candidate_offsets + planned_count +
      (recipient_spans_from_csv
          ? 0 : donor_plan->CandidateSpecs(region).size());
  if (initial.size() > initial_limit) initial.resize(initial_limit);
  if (!WriteCandidateRows(ledger_base, region, initial)) {
    return SearchResult::kFailed;
  }

  bool paused = false;
  if (config.planned_only) {
    const size_t count = std::min<size_t>(
        config.planned_donors, initial.size());
    std::vector<size_t> prefix_sizes;
    if (count != 0) {
      if (config.planned_prefixes) {
        for (size_t prefix = 1; prefix < count; prefix <<= 1) {
          prefix_sizes.push_back(prefix);
        }
      }
      prefix_sizes.push_back(count);
    }
    static const char* stages[] = {
        "planned0", "planned1", "planned2", "planned3", "planned4",
        "planned5", "planned6", "planned7", "planned8"};
    for (const size_t prefix : prefix_sizes) {
      DonorSequence planned(initial.begin(), initial.begin() + prefix);
      ScoredSequence scored;
      if (Evaluate(region, stages[std::min<size_t>(prefix, 8u)], planned,
          baseline.payload_bytes, baseline.payload_bytes, bytes, size,
          position, encoder, predictor, input_fd, config, ledger_base, trials,
          next_trial_id, new_trials, &paused, &scored) &&
          BeatsSelection(region, scored, *selection)) {
        selection->selected_bytes = scored.payload_bytes;
        selection->sequence = scored.sequence;
      }
      if (paused) return SearchResult::kPaused;
    }
    if (!AppendSelection(ledger_base, region, *selection,
        static_cast<uint32_t>(initial.size()),
        *new_trials - region_trial_start)) {
      return SearchResult::kFailed;
    }
    return SearchResult::kComplete;
  }
  std::vector<ScoredSequence> singles;
  auto consider = [&](const ScoredSequence& scored) {
    singles.push_back(scored);
    if (BeatsSelection(region, scored, *selection)) {
      selection->selected_bytes = scored.payload_bytes;
      selection->sequence = scored.sequence;
    }
  };

  for (const DonorWindow& donor : initial) {
    ScoredSequence scored;
    if (Evaluate(region, "single", DonorSequence{donor},
        baseline.payload_bytes, baseline.payload_bytes, bytes, size, position,
        encoder, predictor, input_fd, config, ledger_base, trials,
        next_trial_id, new_trials, &paused, &scored)) {
      consider(scored);
    }
    if (paused) return SearchResult::kPaused;
  }
  SortScored(&singles);

  // A broad first pass should measure each ranked donor exactly without
  // immediately multiplying the run into length, local-offset, and beam
  // searches. Positive and near-positive singles can be refined in a later
  // resumable pass using the durable trial ledger.
  if (config.quick_singles) {
    if (!AppendSelection(ledger_base, region, *selection,
        static_cast<uint32_t>(initial.size()),
        *new_trials - region_trial_start)) {
      return SearchResult::kFailed;
    }
    return SearchResult::kComplete;
  }

  const std::array<uint32_t, 9> lengths{
      256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  const size_t refine_count =
      std::min<size_t>(config.refine_offsets, singles.size());
  for (size_t index = 0; index < refine_count; ++index) {
    const DonorWindow base = singles[index].sequence.front();
    for (uint32_t length : lengths) {
      DonorWindow donor{base.offset, length, base.score};
      if (!ValidWindow(donor, position) || donor.length == base.length) {
        continue;
      }
      ScoredSequence scored;
      if (Evaluate(region, "length", DonorSequence{donor},
          baseline.payload_bytes, baseline.payload_bytes, bytes, size,
          position, encoder, predictor, input_fd, config, ledger_base,
          trials, next_trial_id, new_trials, &paused, &scored)) {
        consider(scored);
      }
      if (paused) return SearchResult::kPaused;
    }
  }
  SortScored(&singles);

  const size_t local_count =
      std::min<size_t>(config.local_seeds, singles.size());
  for (size_t index = 0; index < local_count; ++index) {
    const DonorWindow base = singles[index].sequence.front();
    for (int64_t delta = -static_cast<int64_t>(config.local_radius);
         delta <= static_cast<int64_t>(config.local_radius);
         delta += kAlignment) {
      if (delta == 0) continue;
      const int64_t shifted = static_cast<int64_t>(base.offset) + delta;
      if (shifted < 0 ||
          shifted > std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      DonorWindow donor{
          static_cast<uint32_t>(shifted), base.length, base.score};
      if (!ValidWindow(donor, position)) continue;
      ScoredSequence scored;
      if (Evaluate(region, "local", DonorSequence{donor},
          baseline.payload_bytes, baseline.payload_bytes, bytes, size,
          position, encoder, predictor, input_fd, config, ledger_base,
          trials, next_trial_id, new_trials, &paused, &scored)) {
        consider(scored);
      }
      if (paused) return SearchResult::kPaused;
    }
  }
  SortScored(&singles);

  std::vector<DonorWindow> combo_pool;
  for (const ScoredSequence& scored : singles) {
    if (scored.payload_bytes > baseline.payload_bytes +
        config.near_loss_bytes && combo_pool.size() >= 2) {
      continue;
    }
    AddUniqueWindow(scored.sequence.front(), position, &combo_pool);
    if (combo_pool.size() >= config.combo_candidates) break;
  }

  std::vector<ScoredSequence> beam;
  for (size_t left = 0; left < combo_pool.size(); ++left) {
    for (size_t right = 0; right < combo_pool.size(); ++right) {
      if (left == right) continue;
      DonorSequence sequence{combo_pool[left], combo_pool[right]};
      ScoredSequence scored;
      uint64_t parent = baseline.payload_bytes;
      const auto first = trials->find(
          TrialKey(region, DonorSequence{combo_pool[left]}));
      if (first != trials->end() && first->second.ok) {
        parent = first->second.payload_bytes;
      }
      if (Evaluate(region, "pair", sequence, baseline.payload_bytes,
          parent, bytes, size, position, encoder, predictor, input_fd,
          config, ledger_base, trials, next_trial_id, new_trials, &paused,
          &scored)) {
        beam.push_back(scored);
        if (BeatsSelection(region, scored, *selection)) {
          selection->selected_bytes = scored.payload_bytes;
          selection->sequence = scored.sequence;
        }
      }
      if (paused) return SearchResult::kPaused;
    }
  }
  SortScored(&beam);
  if (beam.size() > config.beam_width) beam.resize(config.beam_width);

  for (uint32_t depth = 3; depth <= config.max_depth && !beam.empty();
       ++depth) {
    std::vector<ScoredSequence> expanded;
    std::unordered_set<std::string> seen;
    for (const ScoredSequence& parent : beam) {
      for (const DonorWindow& donor : combo_pool) {
        if (ContainsWindow(parent.sequence, donor)) continue;
        DonorSequence sequence = parent.sequence;
        sequence.push_back(donor);
        if (!seen.insert(SequenceText(sequence)).second) continue;
        ScoredSequence scored;
        if (Evaluate(region, "beam", sequence, baseline.payload_bytes,
            parent.payload_bytes, bytes, size, position, encoder, predictor,
            input_fd, config, ledger_base, trials, next_trial_id, new_trials,
            &paused, &scored)) {
          if (scored.payload_bytes <= baseline.payload_bytes +
              config.near_loss_bytes) {
            expanded.push_back(scored);
          }
          if (BeatsSelection(region, scored, *selection)) {
            selection->selected_bytes = scored.payload_bytes;
            selection->sequence = scored.sequence;
          }
        }
        if (paused) return SearchResult::kPaused;
      }
    }
    SortScored(&expanded);
    if (expanded.size() > config.beam_width) {
      expanded.resize(config.beam_width);
    }
    beam.swap(expanded);
  }

  if (!AppendSelection(ledger_base, region, *selection,
      static_cast<uint32_t>(initial.size()),
      *new_trials - region_trial_start)) {
    return SearchResult::kFailed;
  }
  return SearchResult::kComplete;
}

}  // namespace

bool RunDonorWinnerSearch(const std::string& input_path,
    const std::string& scratch_output_path, uint64_t input_bytes,
    const std::vector<bool>& vocab, FILE* dictionary,
    bool pretrain_dictionary, DonorPlan* donor_plan,
    const std::string& ledger_path, uint64_t* output_bytes) {
  if (!donor_plan || !donor_plan->discovery_candidates() ||
      !EnsureLedgers(ledger_path)) {
    return false;
  }

  const SearchConfig config = LoadConfig();
  std::ifstream input(input_path, std::ios::in | std::ios::binary);
  std::ofstream output(
      scratch_output_path, std::ios::out | std::ios::binary);
  const int input_fd = open(input_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (!input.is_open() || !output.is_open() || input_fd < 0) {
    if (input_fd >= 0) close(input_fd);
    return false;
  }

  Predictor predictor(vocab);
  if (pretrain_dictionary) preprocessor::Pretrain(&predictor, dictionary);
  Encoder encoder(&output, &predictor);

  std::vector<RecipientSpan> recipient_spans;
  if (!LoadRecipientSpans(input_bytes, &recipient_spans)) {
    close(input_fd);
    return false;
  }
  if (!LoadRankedPageCandidates(recipient_spans) ||
      !LoadRankedRecipientFilter(recipient_spans)) {
    close(input_fd);
    return false;
  }
  const size_t complete_regions = recipient_spans.size();
  std::unordered_map<std::string, TrialRecord> trials;
  std::unordered_set<std::string> completed_marginals;
  std::vector<SelectionRecord> selections(complete_regions);
  uint64_t next_trial_id = 1;
  if (!LoadTrials(ledger_path, &trials, &next_trial_id) ||
      !LoadSelections(ledger_path, &selections) ||
      !LoadMarginals(ledger_path, &completed_marginals)) {
    close(input_fd);
    return false;
  }

  unlink(CompletePath(ledger_path).c_str());
  unlink(PausedPath(ledger_path).c_str());
  CausalPhraseIndex phrase_index;
  std::vector<char> buffer(DonorPlan::kChunkSize);
  uint64_t position = 0;
  uint64_t new_trials = 0;
  uint32_t searched_regions = 0;

  auto encode_baseline = [&](uint64_t byte_count) {
    while (byte_count != 0) {
      const size_t count = static_cast<size_t>(
          std::min<uint64_t>(buffer.size(), byte_count));
      input.read(buffer.data(), static_cast<std::streamsize>(count));
      if (static_cast<size_t>(input.gcount()) != count ||
          !EncodeRegion(buffer.data(), count, position, &encoder,
              donor_plan, nullptr)) {
        return false;
      }
      phrase_index.AddRegion(buffer.data(), count, position);
      position += count;
      byte_count -= count;
    }
    return true;
  };

  for (const RecipientSpan& span : recipient_spans) {
    if (span.offset < position ||
        span.offset + span.length > input_bytes ||
        !encode_baseline(span.offset - position)) {
      close(input_fd);
      return false;
    }
    if (buffer.size() < span.length) buffer.resize(span.length);
    input.read(buffer.data(), static_cast<std::streamsize>(span.length));
    const size_t count = static_cast<size_t>(input.gcount());
    if (count != span.length) {
      close(input_fd);
      return false;
    }

    const uint32_t region = span.region;
    current_recipient_stream_class = span.stream_class;
    const uint64_t start_size = encoder.ProjectedFinalOutputSize();
    SelectionRecord selection;
    bool region_encoded = false;
    if (region < selections.size() && selections[region].valid) {
      selection = selections[region];
    } else if (SearchRecipient(region) &&
        region >= config.start_region &&
        (config.max_regions == 0 || searched_regions < config.max_regions)) {
      const SearchResult result = SearchRegion(region, buffer.data(), count,
          position, &encoder, &predictor, input_fd, donor_plan,
          phrase_index, config, ledger_path, &trials, &next_trial_id,
          &new_trials, &selection);
      region_encoded = result == SearchResult::kCompleteAndEncoded;
      if (result == SearchResult::kFailed) {
        close(input_fd);
        return false;
      }
      if (result == SearchResult::kPaused) {
        FILE* paused = fopen(PausedPath(ledger_path).c_str(), "wb");
        if (!paused) {
          close(input_fd);
          return false;
        }
        fprintf(paused,
            "recipient_region=%u\nrecipient_offset=%" PRIu64
            "\nnew_trials=%" PRIu64 "\n",
            region, position, new_trials);
        const bool ok = SyncFile(paused) && WriteStatus(ledger_path,
            "paused", region, new_trials, {}, 0);
        fclose(paused);
        close(input_fd);
        *output_bytes = 0;
        return ok;
      }
      selections[region] = selection;
      ++searched_regions;
    } else {
      selection.valid = true;
      selection.baseline_bytes = 0;
      selection.selected_bytes = 0;
    }

    if (!region_encoded && config.leave_one_out && selection.valid &&
        selection.baseline_bytes != 0 && region >= config.start_region &&
        SearchRecipient(region)) {
      ScoredSequence best_raw;
      if (FindBestRawSequence(
          region, selection.baseline_bytes, trials, &best_raw)) {
        const SearchResult marginal_result = AttributeDonorMarginals(
            region, best_raw, selection.baseline_bytes, buffer.data(), count,
            position, &encoder, &predictor, input_fd, config, ledger_path,
            &trials, &completed_marginals, &next_trial_id, &new_trials);
        if (marginal_result == SearchResult::kFailed) {
          close(input_fd);
          return false;
        }
        if (marginal_result == SearchResult::kPaused) {
          FILE* paused = fopen(PausedPath(ledger_path).c_str(), "wb");
          if (!paused) {
            close(input_fd);
            return false;
          }
          fprintf(paused,
              "recipient_region=%u\nrecipient_offset=%" PRIu64
              "\nnew_trials=%" PRIu64 "\nphase=leave_one_out\n",
              region, position, new_trials);
          const bool ok = SyncFile(paused) && WriteStatus(ledger_path,
              "leave_one_out_paused", region, new_trials,
              best_raw.sequence, 0);
          fclose(paused);
          close(input_fd);
          *output_bytes = 0;
          return ok;
        }
      }
    }

    if (!region_encoded && !EncodeRegion(buffer.data(), count, position, &encoder,
        donor_plan, nullptr)) {
      close(input_fd);
      return false;
    }
    const uint64_t actual_bytes =
        encoder.ProjectedFinalOutputSize() - start_size;
    if (selection.valid && selection.baseline_bytes != 0 &&
        actual_bytes != selection.baseline_bytes) {
      fprintf(stderr,
          "baseline replay mismatch at region %u: expected %" PRIu64
          ", got %" PRIu64 "\n",
          region, selection.baseline_bytes, actual_bytes);
      close(input_fd);
      return false;
    }
    phrase_index.AddRegion(buffer.data(), count, position);
    position += count;

    if (config.max_regions != 0 &&
        searched_regions >= config.max_regions && position < input_bytes) {
      FILE* paused = fopen(PausedPath(ledger_path).c_str(), "wb");
      if (!paused) {
        close(input_fd);
        return false;
      }
      const uint32_t next_region = region + 1;
      fprintf(paused,
          "recipient_region=%u\nrecipient_offset=%" PRIu64
          "\nnew_trials=%" PRIu64 "\n",
          next_region, RecipientOffset(next_region), new_trials);
      const bool ok = SyncFile(paused) && WriteStatus(ledger_path,
          "region_limit", next_region, new_trials, {}, 0);
      fclose(paused);
      close(input_fd);
      *output_bytes = 0;
      return ok;
    }
  }

  const uint64_t target_bytes = std::min(input_bytes, config.stop_offset);
  if (position > target_bytes ||
      !encode_baseline(target_bytes - position)) {
    close(input_fd);
    return false;
  }

  encoder.Flush();
  output.close();
  close(input_fd);
  if (!output.good()) return false;
  struct stat output_info {};
  *output_bytes = stat(scratch_output_path.c_str(), &output_info) == 0
      ? static_cast<uint64_t>(output_info.st_size) : 0;

  FILE* complete = fopen(CompletePath(ledger_path).c_str(), "wb");
  if (!complete) return false;
  fprintf(complete,
      "regions=%zu\ninput_bytes=%" PRIu64 "\nnew_trials=%" PRIu64
      "\npid=%ld\n",
      complete_regions, target_bytes, new_trials,
      static_cast<long>(getpid()));
  const bool ok = SyncFile(complete);
  fclose(complete);
  return ok;
}
