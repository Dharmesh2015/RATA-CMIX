#include "donor_winner_search.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cmath>
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
#include <malloc.h>
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
#include "models/scr2_tokens.h"
#include "predictor.h"
#include "preprocess/preprocessor.h"

namespace {

constexpr uint32_t kAlignment = 256;
constexpr uint32_t kInitialLength = 4096;
constexpr uint32_t kLossSpanBytes = 4096;
constexpr size_t kLossSpanCount =
    DonorPlan::kChunkSize / kLossSpanBytes;
constexpr size_t kPostingCapacity = 6;
constexpr std::uint16_t kCostPositiveScr2 = 0xffffu;
constexpr double kScr2MinimumGrossBytes = 2.25;
constexpr double kScr2EventTaxBytes = 3.0;

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

constexpr const char* kPortfolioTrialHeader =
    "recipient_region,recipient_offset,recipient_length,trial_id,mode,"
    "expert_mask,mini_model_mask,profile_id,stream_class,donor_count,donors,"
    "vr_min_length,vr_event_count,"
    "baseline_payload_bytes,candidate_payload_bytes,gross_gain_bytes,"
    "standalone_plan_bytes,net_gain_bytes,status\n";
constexpr const char* kPortfolioSelectionHeader =
    "recipient_region,recipient_offset,recipient_length,mode,expert_mask,"
    "mini_model_mask,profile_id,stream_class,donor_count,donors,"
    "vr_min_length,vr_event_count,baseline_payload_bytes,"
    "candidate_payload_bytes,gross_gain_bytes,standalone_plan_bytes,"
    "net_gain_bytes,candidate_count,exact_trials,status\n";
constexpr const char* kPortfolioReplayEventHeader =
    "recipient_region,mode,event_offset,pattern\n";

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

struct ReplayEvent {
  std::uint32_t local_offset = 0;
  std::uint16_t pattern = 0;
};

struct ScoredReplayEvent {
  ReplayEvent event;
  std::uint32_t end = 0;
  double gross_bytes = 0.0;
};

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

struct PortfolioChoice {
  bool valid = false;
  std::string mode = "baseline";
  DonorSequence sequence;
  uint32_t expert_mask = 0;
  uint16_t mini_model_mask = 0;
  uint8_t profile_id = 0;
  uint8_t stream_class = static_cast<uint8_t>(
      PostR1Experts::StreamClass::kMixed);
  uint16_t vr_min_length = 0;
  uint32_t vr_event_count = 0;
  uint64_t payload_bytes = 0;
  uint64_t side_bytes = 0;
  int64_t source_net_bytes = 0;
};

struct ScoredSequence {
  DonorSequence sequence;
  uint64_t payload_bytes = 0;
};

std::vector<std::vector<PortfolioChoice>> phase_individual_trials;
std::vector<std::vector<PortfolioChoice>> phase_donor_trials;

struct SearchConfig {
  enum class PortfolioPhase {
    kIndividual,
    kDonorBeam,
    kCombine,
  };

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
  bool quick_context_mixer = false;
  uint32_t quick_pair_candidates = 0;
  uint32_t quick_prefix_depth = 1;
  bool quick_reverse_prefixes = false;
  uint32_t portfolio_level = 2;
  PortfolioPhase portfolio_phase = PortfolioPhase::kIndividual;
  uint32_t portfolio_beam_width = 8;
  uint32_t portfolio_donor_atoms = 24;
  uint32_t portfolio_max_candidates = 24;
  uint32_t portfolio_max_depth = 8;
  bool topology_search = false;
  bool topology_only = false;
  bool causal_cnn_search = false;
  bool causal_cnn_only = false;
  bool scr2_cost_search = false;
  bool scr2_cost_only = false;
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
  config.quick_context_mixer =
      EnvironmentU32("FX4_WINNER_CONTEXT_MIXER", 0, 0, 1) != 0;
  config.quick_pair_candidates =
      EnvironmentU32("FX4_WINNER_QUICK_PAIR_CANDIDATES", 0, 0, 16);
  config.quick_prefix_depth =
      EnvironmentU32("FX4_WINNER_QUICK_PREFIX_DEPTH", 1, 1, 8);
  config.quick_reverse_prefixes =
      EnvironmentU32("FX4_WINNER_QUICK_REVERSE_PREFIXES", 0, 0, 1) != 0;
  config.portfolio_level =
      EnvironmentU32("FX4_WINNER_PORTFOLIO_LEVEL", 2, 0, 2);
  const char* phase = getenv("FX4_WINNER_PORTFOLIO_PHASE");
  if (!phase || std::strcmp(phase, "individual") == 0) {
    config.portfolio_phase = SearchConfig::PortfolioPhase::kIndividual;
  } else if (std::strcmp(phase, "donor_beam") == 0) {
    config.portfolio_phase = SearchConfig::PortfolioPhase::kDonorBeam;
  } else if (std::strcmp(phase, "combine") == 0) {
    config.portfolio_phase = SearchConfig::PortfolioPhase::kCombine;
  } else {
    std::fprintf(stderr, "invalid FX4_WINNER_PORTFOLIO_PHASE: %s\n", phase);
    std::exit(2);
  }
  config.portfolio_beam_width =
      EnvironmentU32("FX4_WINNER_PORTFOLIO_BEAM_WIDTH", 8, 1, 64);
  config.portfolio_donor_atoms =
      EnvironmentU32("FX4_WINNER_PORTFOLIO_DONOR_ATOMS", 24, 2, 64);
  config.portfolio_max_candidates =
      EnvironmentU32("FX4_WINNER_PORTFOLIO_MAX_CANDIDATES", 48, 1, 128);
  config.portfolio_max_depth =
      EnvironmentU32("FX4_WINNER_PORTFOLIO_MAX_DEPTH", 8, 2, 16);
  config.topology_search =
      EnvironmentU32("FX4_WINNER_TOPOLOGY", 0, 0, 1) != 0;
  config.topology_only =
      EnvironmentU32("FX4_WINNER_TOPOLOGY_ONLY", 0, 0, 1) != 0;
  if (config.topology_only) config.topology_search = true;
  config.causal_cnn_search =
      EnvironmentU32("FX4_WINNER_CAUSAL_CNN", 0, 0, 1) != 0;
  config.causal_cnn_only =
      EnvironmentU32("FX4_WINNER_CAUSAL_CNN_ONLY", 0, 0, 1) != 0;
  if (config.causal_cnn_only) config.causal_cnn_search = true;
  config.scr2_cost_search =
      EnvironmentU32("FX4_WINNER_SCR2_COST", 0, 0, 1) != 0;
  config.scr2_cost_only =
      EnvironmentU32("FX4_WINNER_SCR2_COST_ONLY", 0, 0, 1) != 0;
  if (config.scr2_cost_only) config.scr2_cost_search = true;

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

std::string PortfolioTrialPath(const std::string& base) {
  return base + ".portfolio_trials.csv";
}

std::string PortfolioSelectionPath(const std::string& base) {
  return base + ".portfolio_selected.csv";
}

std::string PortfolioReplayEventPath(const std::string& base) {
  return base + ".portfolio_vr_events.csv";
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
      EnsureCsv(MarginalPath(base), kMarginalHeader) &&
      EnsureCsv(PortfolioTrialPath(base), kPortfolioTrialHeader) &&
      EnsureCsv(PortfolioSelectionPath(base), kPortfolioSelectionHeader) &&
      EnsureCsv(PortfolioReplayEventPath(base),
          kPortfolioReplayEventHeader);
}

bool AppendRow(const std::string& path, const std::string& row) {
  FILE* output = fopen(path.c_str(), "ab");
  if (!output) return false;
  const bool ok = fwrite(row.data(), 1, row.size(), output) == row.size() &&
      SyncFile(output);
  fclose(output);
  return ok;
}

bool AppendPortfolioReplayEvents(const std::string& base,
    std::uint32_t region, const std::string& mode, std::uint64_t position,
    const std::vector<ReplayEvent>& events) {
  if (events.empty()) return true;
  std::string rows;
  rows.reserve(events.size() * 48u);
  char row[192];
  for (const ReplayEvent& event : events) {
    const int length = std::snprintf(row, sizeof(row),
        "%u,%s,%" PRIu64 ",%u\n", region, mode.c_str(),
        position + event.local_offset, event.pattern);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(row)) {
      return false;
    }
    rows.append(row, static_cast<std::size_t>(length));
  }
  return AppendRow(PortfolioReplayEventPath(base), rows);
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

// Exact standalone v7 plan cost for one selective action. Charging the full
// header, donor profile, expert profile, and span to every candidate is
// conservative; the final combined plan can only be smaller through sharing.
uint64_t StandalonePortfolioPlanCost(uint64_t recipient_offset,
    uint32_t recipient_length, const DonorSequence& sequence,
    uint32_t expert_mask, uint8_t stream_class, uint8_t profile_id,
    uint16_t mini_model_mask) {
  (void)stream_class;
  (void)profile_id;
  if (expert_mask == 0) return sequence.empty() ? 0 : UINT64_MAX;
  if (((expert_mask & PostR1Experts::kDonorProfile) != 0) !=
      !sequence.empty()) {
    return UINT64_MAX;
  }
  if (((expert_mask & PostR1Experts::kMiniCmix) != 0) !=
      (mini_model_mask != 0)) {
    return UINT64_MAX;
  }

  uint64_t bytes = 2;  // version and flags
  bytes += CompactVarintSize(sequence.empty() ? 0 : 1);
  if (!sequence.empty()) {
    bytes += CompactVarintSize(0);  // standalone donor-profile id
    bytes += CompactVarintSize(sequence.size());
    int64_t previous_shifted_offset = 0;
    for (size_t index = 0; index < sequence.size(); ++index) {
      if ((sequence[index].offset & 255u) != 0 ||
          sequence[index].length < 256u ||
          sequence[index].length > 65536u ||
          (sequence[index].length & (sequence[index].length - 1u)) != 0) {
        return UINT64_MAX;
      }
      const int64_t shifted_offset = sequence[index].offset >> 8;
      const uint64_t encoded_offset = index == 0
          ? static_cast<uint64_t>(shifted_offset)
          : CompactSignedDelta(shifted_offset - previous_shifted_offset);
      bytes += CompactVarintSize(encoded_offset) + 1;
      previous_shifted_offset = shifted_offset;
    }
  }

  bytes += CompactVarintSize(1);  // one expert profile
  bytes += CompactVarintSize(expert_mask) + 2;
  bytes += CompactVarintSize(mini_model_mask);
  bytes += CompactVarintSize(1);  // one span
  bytes += CompactVarintSize(recipient_offset);
  bytes += CompactVarintSize(recipient_length);
  bytes += CompactVarintSize(0);  // profile index
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

std::vector<ReplayEvent> FindScr2Events(const char* bytes, std::size_t size,
    std::uint64_t position, std::uint16_t minimum_length) {
  std::vector<ReplayEvent> events;
  std::size_t local = 0;
  while (local < size) {
    std::uint16_t best_code = 0;
    std::uint16_t best_length = 0;
    const std::uint8_t first = static_cast<std::uint8_t>(bytes[local]);
    for (std::uint16_t code = 1; code <= scr2::kTokenCount; ++code) {
      const scr2::TokenInfo& token = scr2::kTokens[code];
      if (token.first != first || token.length < minimum_length ||
          local + token.length > size || token.length < best_length) {
        continue;
      }
      const std::uint64_t absolute = position + local;
      if (absolute / FX4_IO_BUFFER_BYTES !=
          (absolute + token.length - 1) / FX4_IO_BUFFER_BYTES) {
        continue;
      }
      if (std::memcmp(bytes + local, scr2::PatternData(code),
          token.length) == 0 &&
          (token.length > best_length ||
           (token.length == best_length && code < best_code))) {
        best_code = code;
        best_length = token.length;
      }
    }
    if (best_code == 0) {
      ++local;
      continue;
    }
    events.push_back({
        static_cast<std::uint32_t>(local), best_code});
    local += best_length;
  }
  return events;
}

std::uint64_t StandaloneVirtualReplayPlanCost(std::uint64_t position,
    const std::vector<ReplayEvent>& events) {
  if (events.empty()) return UINT64_MAX;
  std::array<std::uint16_t, scr2::kTokenCount + 1> compact{};
  compact.fill(std::numeric_limits<std::uint16_t>::max());
  std::vector<std::uint16_t> used;
  for (const ReplayEvent& event : events) {
    if (compact[event.pattern] ==
        std::numeric_limits<std::uint16_t>::max()) {
      compact[event.pattern] = static_cast<std::uint16_t>(used.size());
      used.push_back(event.pattern);
    }
  }

  std::uint64_t bytes = 1 + 2;
  for (const std::uint16_t code : used) {
    bytes += 2 + scr2::kTokens[code].length;
  }
  bytes += 4;
  std::uint64_t previous_end = 0;
  for (const ReplayEvent& event : events) {
    const std::uint64_t absolute = position + event.local_offset;
    if (absolute < previous_end) return UINT64_MAX;
    bytes += CompactVarintSize(absolute - previous_end);
    bytes += CompactVarintSize(compact[event.pattern]);
    previous_end = absolute + scr2::kTokens[event.pattern].length;
  }
  return bytes;
}

double ReplayBitCost(int bit, std::uint16_t probability) {
  const unsigned int mass = bit ? probability : 65536u - probability;
  return -std::log2(static_cast<double>(mass) / 65536.0);
}

std::vector<ScoredReplayEvent> FindScoredScr2Events(const char* bytes,
    std::size_t size, std::uint64_t position,
    const std::vector<std::uint16_t>& probabilities) {
  std::vector<ScoredReplayEvent> candidates;
  if (probabilities.size() != size * 8u) return candidates;

  std::vector<double> prefix_cost(size + 1, 0.0);
  for (std::size_t local = 0; local < size; ++local) {
    const std::uint8_t byte = static_cast<std::uint8_t>(bytes[local]);
    double cost = 0.0;
    for (unsigned int bit_position = 0; bit_position < 8; ++bit_position) {
      const int bit = (byte >> (7 - bit_position)) & 1;
      cost += ReplayBitCost(
          bit, probabilities[local * 8u + bit_position]);
    }
    prefix_cost[local + 1] = prefix_cost[local] + cost;
  }

  std::array<std::vector<std::uint16_t>, 256> by_first;
  for (std::uint16_t code = 1; code <= scr2::kTokenCount; ++code) {
    const scr2::TokenInfo& token = scr2::kTokens[code];
    if (token.length >= 6) by_first[token.first].push_back(code);
  }

  for (std::size_t local = 0; local < size; ++local) {
    const std::uint8_t first = static_cast<std::uint8_t>(bytes[local]);
    for (const std::uint16_t code : by_first[first]) {
      const scr2::TokenInfo& token = scr2::kTokens[code];
      const std::size_t end = local + token.length;
      if (end > size) continue;
      const std::uint64_t absolute = position + local;
      if (absolute / FX4_IO_BUFFER_BYTES !=
          (absolute + token.length - 1) / FX4_IO_BUFFER_BYTES) {
        continue;
      }
      if (std::memcmp(bytes + local, scr2::PatternData(code),
          token.length) != 0) {
        continue;
      }
      const double gross =
          (prefix_cost[end] - prefix_cost[local]) / 8.0;
      if (gross >= kScr2MinimumGrossBytes) {
        candidates.push_back({
            {static_cast<std::uint32_t>(local), code},
            static_cast<std::uint32_t>(end), gross});
      }
    }
  }
  return candidates;
}

std::vector<ScoredReplayEvent> WeightedReplaySelection(
    const std::vector<ScoredReplayEvent>& candidates,
    const std::array<bool, scr2::kTokenCount + 1>& allowed) {
  std::vector<ScoredReplayEvent> filtered;
  for (const ScoredReplayEvent& candidate : candidates) {
    if (allowed[candidate.event.pattern] &&
        candidate.gross_bytes > kScr2EventTaxBytes) {
      filtered.push_back(candidate);
    }
  }
  std::sort(filtered.begin(), filtered.end(),
      [](const ScoredReplayEvent& left,
          const ScoredReplayEvent& right) {
        if (left.end != right.end) return left.end < right.end;
        if (left.event.local_offset != right.event.local_offset) {
          return left.event.local_offset < right.event.local_offset;
        }
        return left.event.pattern < right.event.pattern;
      });
  if (filtered.empty()) return {};

  std::vector<std::uint32_t> ends;
  std::vector<int> previous(filtered.size(), -1);
  std::vector<double> best(filtered.size() + 1, 0.0);
  std::vector<bool> take(filtered.size(), false);
  ends.reserve(filtered.size());
  for (const ScoredReplayEvent& event : filtered) ends.push_back(event.end);

  for (std::size_t index = 0; index < filtered.size(); ++index) {
    const auto found = std::upper_bound(
        ends.begin(), ends.begin() + index,
        filtered[index].event.local_offset);
    const int predecessor =
        static_cast<int>(found - ends.begin()) - 1;
    previous[index] = predecessor;
    const double with_event = best[predecessor + 1] +
        filtered[index].gross_bytes - kScr2EventTaxBytes;
    if (with_event > best[index] + 1.0e-9) {
      best[index + 1] = with_event;
      take[index] = true;
    } else {
      best[index + 1] = best[index];
    }
  }

  std::vector<ScoredReplayEvent> selected;
  std::size_t cursor = filtered.size();
  while (cursor != 0) {
    const std::size_t index = cursor - 1;
    if (take[index] &&
        best[cursor] > best[index] + 1.0e-9) {
      selected.push_back(filtered[index]);
      cursor = static_cast<std::size_t>(previous[index] + 1);
    } else {
      --cursor;
    }
  }
  std::reverse(selected.begin(), selected.end());
  return selected;
}

std::vector<ReplayEvent> SelectCostPositiveScr2Events(const char* bytes,
    std::size_t size, std::uint64_t position,
    const std::vector<std::uint16_t>& probabilities) {
  const std::vector<ScoredReplayEvent> candidates =
      FindScoredScr2Events(bytes, size, position, probabilities);
  std::array<bool, scr2::kTokenCount + 1> allowed{};
  allowed.fill(true);
  allowed[0] = false;
  std::vector<ScoredReplayEvent> selected;

  for (unsigned int iteration = 0; iteration < 12; ++iteration) {
    selected = WeightedReplaySelection(candidates, allowed);
    if (selected.empty()) return {};

    // Remove events that cannot pay even their compact gap and pattern ID.
    bool removed_event = false;
    for (unsigned int prune = 0; prune < 4; ++prune) {
      std::array<std::uint16_t, scr2::kTokenCount + 1> compact{};
      compact.fill(std::numeric_limits<std::uint16_t>::max());
      std::uint16_t next_id = 0;
      for (const ScoredReplayEvent& event : selected) {
        if (compact[event.event.pattern] ==
            std::numeric_limits<std::uint16_t>::max()) {
          compact[event.event.pattern] = next_id++;
        }
      }

      std::vector<ScoredReplayEvent> kept;
      std::uint64_t previous_end = 0;
      for (const ScoredReplayEvent& event : selected) {
        const std::uint64_t absolute =
            position + event.event.local_offset;
        const std::uint64_t event_bytes =
            CompactVarintSize(absolute - previous_end) +
            CompactVarintSize(compact[event.event.pattern]);
        if (event.gross_bytes > static_cast<double>(event_bytes)) {
          kept.push_back(event);
          previous_end = position + event.end;
        } else {
          removed_event = true;
        }
      }
      if (kept.size() == selected.size()) break;
      selected.swap(kept);
      if (selected.empty()) return {};
    }

    std::array<double, scr2::kTokenCount + 1> gross{};
    std::array<double, scr2::kTokenCount + 1> event_cost{};
    std::array<std::uint16_t, scr2::kTokenCount + 1> compact{};
    compact.fill(std::numeric_limits<std::uint16_t>::max());
    std::uint16_t next_id = 0;
    for (const ScoredReplayEvent& event : selected) {
      if (compact[event.event.pattern] ==
          std::numeric_limits<std::uint16_t>::max()) {
        compact[event.event.pattern] = next_id++;
      }
    }
    std::uint64_t previous_end = 0;
    for (const ScoredReplayEvent& event : selected) {
      const std::uint16_t code = event.event.pattern;
      const std::uint64_t absolute = position + event.event.local_offset;
      gross[code] += event.gross_bytes;
      event_cost[code] += CompactVarintSize(absolute - previous_end) +
          CompactVarintSize(compact[code]);
      previous_end = position + event.end;
    }

    std::array<bool, scr2::kTokenCount + 1> next_allowed{};
    bool changed = removed_event;
    for (std::uint16_t code = 1; code <= scr2::kTokenCount; ++code) {
      next_allowed[code] = allowed[code] &&
          gross[code] - event_cost[code] >
              static_cast<double>(2 + scr2::kTokens[code].length);
      if (next_allowed[code] != allowed[code]) changed = true;
    }
    if (!changed) break;
    allowed = next_allowed;
  }

  std::vector<ReplayEvent> result;
  double gross = 0.0;
  result.reserve(selected.size());
  for (const ScoredReplayEvent& event : selected) {
    result.push_back(event.event);
    gross += event.gross_bytes;
  }
  const std::uint64_t side =
      StandaloneVirtualReplayPlanCost(position, result);
  if (side == UINT64_MAX || gross <= static_cast<double>(side)) return {};
  return result;
}

bool LoadPortfolioTrials(const char* path,
    const std::vector<RecipientSpan>& spans,
    std::vector<std::vector<PortfolioChoice>>* winners) {
  winners->assign(spans.size(), {});
  if (!path || !*path) return false;
  std::ifstream input(path);
  std::string line;
  if (!input.is_open() || !std::getline(input, line)) return false;
  const std::vector<std::string> header = Split(line, ',');
  const int region_col = FindColumn(header, "recipient_region");
  const int offset_col = FindColumn(header, "recipient_offset");
  const int length_col = FindColumn(header, "recipient_length");
  const int mode_col = FindColumn(header, "mode");
  const int expert_col = FindColumn(header, "expert_mask");
  const int mini_col = FindColumn(header, "mini_model_mask");
  const int profile_col = FindColumn(header, "profile_id");
  const int class_col = FindColumn(header, "stream_class");
  const int donors_col = FindColumn(header, "donors");
  const int vr_length_col = FindColumn(header, "vr_min_length");
  const int vr_events_col = FindColumn(header, "vr_event_count");
  const int baseline_col = FindColumn(header, "baseline_payload_bytes");
  const int payload_col = FindColumn(header, "candidate_payload_bytes");
  const int side_col = FindColumn(header, "standalone_plan_bytes");
  const int net_col = FindColumn(header, "net_gain_bytes");
  const int status_col = FindColumn(header, "status");
  if (region_col < 0 || offset_col < 0 || length_col < 0 ||
      mode_col < 0 || expert_col < 0 || mini_col < 0 ||
      profile_col < 0 || class_col < 0 || donors_col < 0 ||
      vr_length_col < 0 || vr_events_col < 0 || baseline_col < 0 ||
      payload_col < 0 || side_col < 0 || net_col < 0 ||
      status_col < 0) {
    return false;
  }

  while (std::getline(input, line)) {
    const std::vector<std::string> fields = Split(line, ',');
    if (fields.size() != header.size()) return false;
    std::uint64_t region = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t expert_mask = 0;
    std::uint64_t mini_mask = 0;
    std::uint64_t profile_id = 0;
    std::uint64_t stream_class = 0;
    std::uint64_t vr_min_length = 0;
    std::uint64_t vr_event_count = 0;
    std::uint64_t baseline = 0;
    std::uint64_t payload = 0;
    std::uint64_t side = 0;
    const bool valid_status =
        fields[status_col] == "winner_after_side" ||
        fields[status_col] == "near_after_side" ||
        fields[status_col] == "loss_after_side";
    if (!valid_status ||
        !ParseU64(fields[region_col], &region) ||
        !ParseU64(fields[offset_col], &offset) ||
        !ParseU64(fields[length_col], &length) ||
        !ParseU64(fields[expert_col], &expert_mask) ||
        !ParseU64(fields[mini_col], &mini_mask) ||
        !ParseU64(fields[profile_col], &profile_id) ||
        !ParseU64(fields[class_col], &stream_class) ||
        !ParseU64(fields[vr_length_col], &vr_min_length) ||
        !ParseU64(fields[vr_events_col], &vr_event_count) ||
        !ParseU64(fields[baseline_col], &baseline) ||
        !ParseU64(fields[payload_col], &payload) ||
        !ParseU64(fields[side_col], &side)) {
      continue;
    }
    char* net_end = nullptr;
    const long long net = strtoll(fields[net_col].c_str(), &net_end, 10);
    const int64_t expected_net =
        static_cast<int64_t>(baseline) -
        static_cast<int64_t>(payload + side);
    if (*net_end != '\0' || net != expected_net ||
        region >= spans.size() ||
        offset != spans[region].offset || length != spans[region].length ||
        expert_mask > UINT32_MAX || mini_mask > 0x07ffu ||
        profile_id > UINT8_MAX || stream_class > static_cast<std::uint8_t>(
            PostR1Experts::StreamClass::kMixed) ||
        vr_min_length > UINT16_MAX || vr_event_count > UINT32_MAX) {
      continue;
    }

    PortfolioChoice choice;
    choice.valid = true;
    choice.mode = fields[mode_col];
    if (!ParseSequence(fields[donors_col], &choice.sequence)) return false;
    choice.expert_mask = static_cast<std::uint32_t>(expert_mask);
    choice.mini_model_mask = static_cast<std::uint16_t>(mini_mask);
    choice.profile_id = static_cast<std::uint8_t>(profile_id);
    choice.stream_class = static_cast<std::uint8_t>(stream_class);
    choice.vr_min_length = static_cast<std::uint16_t>(vr_min_length);
    choice.vr_event_count = static_cast<std::uint32_t>(vr_event_count);
    choice.payload_bytes = payload;
    choice.side_bytes = side;
    choice.source_net_bytes = net;
    (*winners)[region].push_back(std::move(choice));
  }
  return true;
}

bool TopologyTrialsComplete(std::uint32_t region) {
  if (region >= phase_individual_trials.size()) return false;
  std::array<bool, 4> strengths{};
  for (const PortfolioChoice& choice : phase_individual_trials[region]) {
    if (choice.expert_mask != PostR1Experts::kTopologyRecurrence ||
        !choice.sequence.empty() || choice.vr_min_length != 0 ||
        (choice.profile_id & 63u) != 0) {
      continue;
    }
    strengths[(choice.profile_id >> 6) & 3u] = true;
  }
  return std::all_of(strengths.begin(), strengths.end(),
      [](bool present) { return present; });
}

bool CausalCnnTrialsComplete(std::uint32_t region) {
  if (region >= phase_individual_trials.size()) return false;
  std::array<bool, 4> strengths{};
  for (const PortfolioChoice& choice : phase_individual_trials[region]) {
    if (choice.expert_mask != PostR1Experts::kCausalCnn ||
        !choice.sequence.empty() || choice.vr_min_length != 0 ||
        (choice.profile_id & 63u) != 0) {
      continue;
    }
    strengths[(choice.profile_id >> 6) & 3u] = true;
  }
  return std::all_of(strengths.begin(), strengths.end(),
      [](bool present) { return present; });
}

bool Scr2CostTrialComplete(std::uint32_t region) {
  if (region >= phase_individual_trials.size()) return false;
  return std::any_of(phase_individual_trials[region].begin(),
      phase_individual_trials[region].end(),
      [](const PortfolioChoice& choice) {
        return choice.expert_mask == 0 && choice.sequence.empty() &&
            choice.vr_min_length == kCostPositiveScr2 &&
            choice.mode == "scr2_vr_cost";
      });
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

  // A portfolio row is the authoritative completion marker. Reading it as
  // well makes a recipient resumable even if interruption happened between
  // the portfolio fsync and the legacy compatibility row.
  std::ifstream portfolio(PortfolioSelectionPath(base));
  if (!portfolio.is_open()) return false;
  if (!std::getline(portfolio, line) ||
      line + "\n" != kPortfolioSelectionHeader) {
    return false;
  }
  while (std::getline(portfolio, line)) {
    const std::vector<std::string> fields = Split(line, ',');
    if (fields.size() != 20) return false;
    uint64_t region = 0;
    uint64_t recipient_offset = 0;
    uint64_t baseline = 0;
    if (!ParseU64(fields[0], &region) ||
        !ParseU64(fields[1], &recipient_offset) ||
        !ParseU64(fields[12], &baseline) ||
        region >= selections->size() ||
        recipient_offset != RecipientOffset(static_cast<uint32_t>(region))) {
      return false;
    }
    SelectionRecord record;
    record.valid = true;
    record.baseline_bytes = baseline;
    record.selected_bytes = baseline;
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

bool AppendPortfolioTrial(const std::string& base, uint32_t region,
    uint32_t recipient_length, uint64_t trial_id,
    const PortfolioChoice& choice, uint64_t baseline_bytes,
    uint32_t near_loss_bytes) {
  const int64_t gross = static_cast<int64_t>(baseline_bytes) -
      static_cast<int64_t>(choice.payload_bytes);
  const int64_t net = gross - static_cast<int64_t>(choice.side_bytes);
  const char* status = net > 0 ? "winner_after_side" :
      net >= -static_cast<int64_t>(near_loss_bytes)
          ? "near_after_side" : "loss_after_side";
  const std::string donors = choice.sequence.empty()
      ? "-" : SequenceText(choice.sequence);
  char row[4096] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%u,%" PRIu64 ",%s,%u,%u,%u,%u,%zu,%s,"
      "%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRId64 ",%" PRIu64
      ",%" PRId64 ",%s\n",
      region, RecipientOffset(region), recipient_length, trial_id,
      choice.mode.c_str(), choice.expert_mask, choice.mini_model_mask,
      choice.profile_id, choice.stream_class, choice.sequence.size(),
      donors.c_str(), choice.vr_min_length, choice.vr_event_count,
      baseline_bytes, choice.payload_bytes, gross, choice.side_bytes,
      net, status);
  return length > 0 && static_cast<size_t>(length) < sizeof(row) &&
      AppendRow(PortfolioTrialPath(base),
          std::string(row, static_cast<size_t>(length)));
}

bool AppendPortfolioSelection(const std::string& base, uint32_t region,
    uint32_t recipient_length, const PortfolioChoice& choice,
    uint64_t baseline_bytes, uint32_t candidate_count,
    uint64_t exact_trials) {
  const int64_t gross = static_cast<int64_t>(baseline_bytes) -
      static_cast<int64_t>(choice.payload_bytes);
  const int64_t net = gross - static_cast<int64_t>(choice.side_bytes);
  const char* status = net > 0 ? "accepted_after_side" : "baseline";
  const std::string donors = choice.sequence.empty()
      ? "-" : SequenceText(choice.sequence);
  char row[4096] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%u,%s,%u,%u,%u,%u,%zu,%s,%u,%u,%" PRIu64
      ",%" PRIu64 ",%" PRId64 ",%" PRIu64 ",%" PRId64
      ",%u,%" PRIu64 ",%s\n",
      region, RecipientOffset(region), recipient_length,
      choice.mode.c_str(), choice.expert_mask, choice.mini_model_mask,
      choice.profile_id, choice.stream_class, choice.sequence.size(),
      donors.c_str(), choice.vr_min_length, choice.vr_event_count,
      baseline_bytes, choice.payload_bytes, gross, choice.side_bytes, net,
      candidate_count, exact_trials, status);
  return length > 0 && static_cast<size_t>(length) < sizeof(row) &&
      AppendRow(PortfolioSelectionPath(base),
          std::string(row, static_cast<size_t>(length)));
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

void EncodeBaselineRangeBit(int bit, float probability, Encoder* encoder) {
  encoder->EncodeRawBit(
      bit, Encoder::DiscretizeProbability(probability));
}

bool EncodeRegion(const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, Predictor* predictor, DonorPlan* donor_plan,
    ProbeMessage* profile) {
  uint64_t span_start = encoder->OutputSize();
  size_t span_index = 0;
  for (size_t index = 0; index < size; ++index) {
    const uint8_t byte = static_cast<uint8_t>(bytes[index]);
    for (int bit = 7; bit >= 0; --bit) {
      const int value = (byte >> bit) & 1;
      const float probability = predictor->Predict();
      EncodeBaselineRangeBit(value, probability, encoder);
      predictor->Perceive(value);
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
        predictor, nullptr, collect_spans ? &message : nullptr) ? 1u : 0u;
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
  PortfolioChoice choice;
  std::unique_ptr<PostR1Experts> expert;
  std::unique_ptr<Encoder> encoder;
  std::vector<ReplayEvent> replay_events;
  std::size_t replay_index = 0;
};

void AddUniqueSequence(const DonorSequence& sequence,
    std::unordered_set<std::string>* seen,
    std::vector<DonorSequence>* sequences) {
  if (sequence.empty() || sequence.size() > 8) return;
  const std::string key = SequenceText(sequence);
  if (seen->insert(key).second) sequences->push_back(sequence);
}

std::string PortfolioIdentity(const PortfolioChoice& choice) {
  return SequenceText(choice.sequence) + "|" +
      std::to_string(choice.expert_mask) + "|" +
      std::to_string(choice.mini_model_mask) + "|" +
      std::to_string(choice.vr_min_length);
}

bool SamePortfolioAction(
    const PortfolioChoice& left, const PortfolioChoice& right) {
  return PortfolioIdentity(left) == PortfolioIdentity(right);
}

void AddBestSourceAction(const PortfolioChoice& source,
    std::vector<PortfolioChoice>* actions) {
  const auto found = std::find_if(actions->begin(), actions->end(),
      [&](const PortfolioChoice& action) {
        return SamePortfolioAction(action, source);
      });
  if (found == actions->end()) {
    actions->push_back(source);
  } else if (source.source_net_bytes > found->source_net_bytes) {
    *found = source;
  }
}

std::vector<DonorSequence> BuildWinningDonorBundles(uint32_t region,
    const SearchConfig& config) {
  struct DonorAtom {
    DonorWindow donor;
    int64_t net = 0;
  };
  std::vector<DonorAtom> atoms;
  if (region >= phase_individual_trials.size()) return {};
  for (const PortfolioChoice& choice : phase_individual_trials[region]) {
    if (choice.expert_mask != PostR1Experts::kDonorProfile ||
        choice.sequence.size() != 1 || choice.vr_min_length != 0) {
      continue;
    }
    const DonorWindow donor = choice.sequence.front();
    const auto found = std::find_if(atoms.begin(), atoms.end(),
        [&](const DonorAtom& atom) { return SameWindow(atom.donor, donor); });
    if (found == atoms.end()) {
      atoms.push_back({donor, choice.source_net_bytes});
    } else if (choice.source_net_bytes > found->net) {
      found->net = choice.source_net_bytes;
    }
  }
  std::sort(atoms.begin(), atoms.end(),
      [](const DonorAtom& left, const DonorAtom& right) {
        return left.net != right.net ? left.net > right.net :
            left.donor.offset < right.donor.offset;
      });
  if (atoms.size() > config.portfolio_donor_atoms) {
    atoms.resize(config.portfolio_donor_atoms);
  }
  if (atoms.size() < 2) return {};

  struct Node {
    DonorSequence sequence;
    std::vector<std::size_t> members;
    int64_t score = 0;
  };
  std::vector<Node> beam;
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    beam.push_back({DonorSequence{atoms[index].donor}, {index},
        atoms[index].net});
  }

  std::vector<DonorSequence> result;
  std::unordered_set<std::string> seen;
  const std::size_t maximum_depth = std::min<std::size_t>(
      config.planned_donors, atoms.size());
  for (std::size_t depth = 2; depth <= maximum_depth; ++depth) {
    std::vector<Node> expanded;
    std::unordered_set<std::string> expanded_seen;
    for (const Node& parent : beam) {
      for (std::size_t index = 0; index < atoms.size(); ++index) {
        if (std::find(parent.members.begin(), parent.members.end(), index) !=
            parent.members.end()) {
          continue;
        }
        Node child = parent;
        child.sequence.push_back(atoms[index].donor);
        child.members.push_back(index);
        child.score += atoms[index].net;
        const std::string key = SequenceText(child.sequence);
        if (expanded_seen.insert(key).second) {
          expanded.push_back(std::move(child));
        }
      }
    }
    std::sort(expanded.begin(), expanded.end(),
        [](const Node& left, const Node& right) {
          return left.score != right.score ? left.score > right.score :
              SequenceText(left.sequence) < SequenceText(right.sequence);
        });
    if (expanded.size() > config.portfolio_beam_width) {
      expanded.resize(config.portfolio_beam_width);
    }
    for (const Node& node : expanded) {
      AddUniqueSequence(node.sequence, &seen, &result);
    }
    beam.swap(expanded);
    if (beam.empty()) break;
  }
  if (result.size() > config.portfolio_max_candidates) {
    result.resize(config.portfolio_max_candidates);
  }
  return result;
}

bool MergePortfolioActions(const PortfolioChoice& left,
    const PortfolioChoice& right, PortfolioChoice* merged) {
  if (!left.sequence.empty() && !right.sequence.empty() &&
      SequenceText(left.sequence) != SequenceText(right.sequence)) {
    return false;
  }
  if (left.vr_min_length != 0 && right.vr_min_length != 0 &&
      left.vr_min_length != right.vr_min_length) {
    return false;
  }
  *merged = left;
  if (merged->sequence.empty()) merged->sequence = right.sequence;
  merged->expert_mask |= right.expert_mask;
  merged->mini_model_mask |= right.mini_model_mask;
  if (merged->vr_min_length == 0) {
    merged->vr_min_length = right.vr_min_length;
  }
  merged->profile_id = 0;
  merged->source_net_bytes += right.source_net_bytes;
  merged->mode = "beam_d" + std::to_string(merged->sequence.size()) +
      "_m" + std::to_string(merged->mini_model_mask) +
      "_x" + std::to_string(merged->expert_mask) +
      "_v" + std::to_string(merged->vr_min_length);
  return !SamePortfolioAction(left, *merged);
}

std::vector<PortfolioChoice> BuildWinningPortfolioBeam(uint32_t region,
    const SearchConfig& config) {
  std::vector<PortfolioChoice> atoms;
  if (region < phase_individual_trials.size()) {
    for (const PortfolioChoice& source : phase_individual_trials[region]) {
      const bool allowed =
          source.expert_mask == PostR1Experts::kDonorProfile ||
          source.expert_mask == PostR1Experts::kMiniCmix ||
          source.expert_mask == PostR1Experts::kShadowLstm200 ||
          source.expert_mask == PostR1Experts::kUrlStructure ||
          source.expert_mask == PostR1Experts::kTopologyRecurrence ||
          source.expert_mask == PostR1Experts::kCausalCnn ||
          (source.expert_mask == 0 && source.vr_min_length != 0 &&
           source.vr_min_length != kCostPositiveScr2);
      const int64_t gross_gain = source.source_net_bytes +
          static_cast<int64_t>(source.side_bytes);
      if (allowed && gross_gain > 0) {
        AddBestSourceAction(source, &atoms);
      }
    }
  }
  if (region < phase_donor_trials.size()) {
    for (const PortfolioChoice& source : phase_donor_trials[region]) {
      if (source.expert_mask == PostR1Experts::kDonorProfile &&
          source.sequence.size() >= 2 && source.sequence.size() <= 7 &&
          source.source_net_bytes +
              static_cast<int64_t>(source.side_bytes) > 0) {
        AddBestSourceAction(source, &atoms);
      }
    }
  }
  std::sort(atoms.begin(), atoms.end(),
      [](const PortfolioChoice& left, const PortfolioChoice& right) {
        return left.source_net_bytes != right.source_net_bytes
            ? left.source_net_bytes > right.source_net_bytes
            : PortfolioIdentity(left) < PortfolioIdentity(right);
      });
  if (atoms.size() > 32) atoms.resize(32);

  struct Node {
    PortfolioChoice choice;
    std::vector<std::size_t> members;
  };
  std::vector<Node> beam;
  std::vector<PortfolioChoice> candidates;
  std::unordered_set<std::string> seen;
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    beam.push_back({atoms[index], {index}});
    if (seen.insert(PortfolioIdentity(atoms[index])).second) {
      candidates.push_back(atoms[index]);
    }
  }
  if (beam.size() > config.portfolio_beam_width) {
    beam.resize(config.portfolio_beam_width);
  }

  const std::size_t maximum_depth = std::min<std::size_t>(
      config.portfolio_max_depth, atoms.size());
  for (std::size_t depth = 2; depth <= maximum_depth; ++depth) {
    std::vector<Node> expanded;
    std::unordered_set<std::string> expanded_seen;
    for (const Node& parent : beam) {
      const std::size_t first = parent.members.back() + 1;
      for (std::size_t index = first; index < atoms.size(); ++index) {
        PortfolioChoice merged;
        if (!MergePortfolioActions(
            parent.choice, atoms[index], &merged)) {
          continue;
        }
        Node child{merged, parent.members};
        child.members.push_back(index);
        const std::string key = PortfolioIdentity(child.choice);
        if (expanded_seen.insert(key).second) {
          expanded.push_back(std::move(child));
        }
      }
    }
    std::sort(expanded.begin(), expanded.end(),
        [](const Node& left, const Node& right) {
          return left.choice.source_net_bytes !=
                  right.choice.source_net_bytes
              ? left.choice.source_net_bytes > right.choice.source_net_bytes
              : PortfolioIdentity(left.choice) <
                  PortfolioIdentity(right.choice);
        });
    if (expanded.size() > config.portfolio_beam_width) {
      expanded.resize(config.portfolio_beam_width);
    }
    for (const Node& node : expanded) {
      if (seen.insert(PortfolioIdentity(node.choice)).second) {
        candidates.push_back(node.choice);
      }
    }
    beam.swap(expanded);
    if (beam.empty()) break;
  }

  if (candidates.size() > config.portfolio_max_candidates) {
    std::unordered_set<std::string> atom_keys;
    for (const PortfolioChoice& atom : atoms) {
      atom_keys.insert(PortfolioIdentity(atom));
    }
    std::vector<PortfolioChoice> retained;
    for (const PortfolioChoice& atom : atoms) {
      if (retained.size() >= config.portfolio_max_candidates) break;
      retained.push_back(atom);
    }
    std::vector<PortfolioChoice> combinations;
    for (const PortfolioChoice& candidate : candidates) {
      if (atom_keys.count(PortfolioIdentity(candidate)) == 0) {
        combinations.push_back(candidate);
      }
    }
    std::sort(combinations.begin(), combinations.end(),
        [](const PortfolioChoice& left, const PortfolioChoice& right) {
          return left.source_net_bytes != right.source_net_bytes
              ? left.source_net_bytes > right.source_net_bytes
              : PortfolioIdentity(left) < PortfolioIdentity(right);
        });
    for (const PortfolioChoice& candidate : combinations) {
      if (retained.size() >= config.portfolio_max_candidates) break;
      retained.push_back(candidate);
    }
    candidates.swap(retained);
  }
  return candidates;
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
  if (!initial.empty() &&
      !WriteCandidateRows(ledger_base, region, initial)) {
    return SearchResult::kFailed;
  }

  std::vector<DonorSequence> sequences;
  if (config.portfolio_phase ==
      SearchConfig::PortfolioPhase::kIndividual) {
    std::unordered_set<std::string> seen_sequences;
    for (const DonorWindow& donor : initial) {
      AddUniqueSequence(DonorSequence{donor}, &seen_sequences, &sequences);
    }
  } else if (config.portfolio_phase ==
      SearchConfig::PortfolioPhase::kDonorBeam) {
    sequences = BuildWinningDonorBundles(region, config);
  }

  std::vector<InPlaceTrial> branches;
  branches.reserve(config.portfolio_max_candidates * 4u + 64u);
  std::unordered_set<std::string> seen_actions;
  auto add_branch = [&](const PortfolioChoice& requested,
      std::uint8_t profile_id, const std::string& mode) {
    PortfolioChoice choice = requested;
    choice.valid = true;
    choice.mode = mode;
    choice.profile_id = profile_id;
    choice.stream_class = current_recipient_stream_class;
    choice.expert_mask |= config.quick_context_mixer
        ? PostR1Experts::kContextMixer : 0u;

    std::vector<ReplayEvent> replay_events;
    std::uint64_t replay_side = 0;
    const bool cost_positive_replay =
        choice.vr_min_length == kCostPositiveScr2;
    if (cost_positive_replay &&
        (choice.expert_mask != 0 || !choice.sequence.empty())) {
      return true;
    }
    if (choice.vr_min_length != 0 && !cost_positive_replay) {
      replay_events = FindScr2Events(
          bytes, size, position, choice.vr_min_length);
      if (replay_events.empty()) return true;
      replay_side =
          StandaloneVirtualReplayPlanCost(position, replay_events);
      if (replay_side == UINT64_MAX) return false;
      choice.vr_event_count =
          static_cast<std::uint32_t>(replay_events.size());
    }

    const std::string key = PortfolioIdentity(choice) + "|" +
        std::to_string(profile_id);
    if (!seen_actions.insert(key).second) return true;

    std::vector<uint8_t> profile;
    std::vector<uint32_t> segment_lengths;
    if (!choice.sequence.empty() && !LoadDonorProfile(
        input_fd, choice.sequence, &profile, &segment_lengths)) {
      return false;
    }

    const std::uint64_t portfolio_side = StandalonePortfolioPlanCost(
        position, static_cast<uint32_t>(size), choice.sequence,
        choice.expert_mask, current_recipient_stream_class,
        profile_id, choice.mini_model_mask);
    if (portfolio_side == UINT64_MAX) return false;
    choice.side_bytes = portfolio_side + replay_side;

    std::unique_ptr<PostR1Experts> expert;
    if (choice.expert_mask != 0) {
      expert.reset(new PostR1Experts());
      expert->EnablePortfolio(choice.expert_mask);
      if (!choice.sequence.empty()) {
        expert->SetDonorProfile(profile, segment_lengths);
      }
      expert->SetSpan(position, choice.expert_mask,
          static_cast<PostR1Experts::StreamClass>(
              current_recipient_stream_class),
          profile_id, choice.mini_model_mask);
    }
    std::unique_ptr<Encoder> branch(
        new Encoder(encoder->CloneCountOnly()));
    branches.push_back({std::move(choice), std::move(expert),
        std::move(branch), std::move(replay_events), 0});
    return true;
  };

  auto add_strength_sweep = [&](const PortfolioChoice& action) {
    if (action.expert_mask == 0 ||
        action.expert_mask == PostR1Experts::kUrlStructure) {
      return add_branch(action, 0, action.mode);
    }
    for (unsigned int gain = 0; gain < 4; ++gain) {
      const std::uint8_t profile_id =
          static_cast<std::uint8_t>(gain << 6);
      if (!add_branch(action, profile_id,
          action.mode + "_g" + std::to_string((gain + 1) * 25))) {
        return false;
      }
    }
    return true;
  };

  if (config.portfolio_phase ==
      SearchConfig::PortfolioPhase::kIndividual) {
    const bool specialist_only =
        config.topology_only || config.causal_cnn_only ||
        config.scr2_cost_only;
    if (!specialist_only) for (const DonorSequence& sequence : sequences) {
      PortfolioChoice action;
      action.mode = "donor";
      action.sequence = sequence;
      action.expert_mask = PostR1Experts::kDonorProfile;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#if FX4_MINI_CMIX
    if (!specialist_only) for (unsigned int model = 0; model < 11; ++model) {
      PortfolioChoice action;
      action.mode = "mini_m" + std::to_string(model);
      action.expert_mask = PostR1Experts::kMiniCmix;
      action.mini_model_mask = static_cast<std::uint16_t>(1u << model);
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#endif
#if FX4_SHADOW_LSTM200
    if (!specialist_only) {
      PortfolioChoice action;
      action.mode = "shadow_lstm200";
      action.expert_mask = PostR1Experts::kShadowLstm200;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#endif
    if (!specialist_only) {
      PortfolioChoice action;
      action.mode = "url_structure";
      action.expert_mask = PostR1Experts::kUrlStructure;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#if FX4_TOPOLOGY_RECURRENCE
    if (config.topology_search) {
      PortfolioChoice action;
      action.mode = "topology_recurrence";
      action.expert_mask = PostR1Experts::kTopologyRecurrence;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#endif
#if FX4_CAUSAL_CNN
    if (config.causal_cnn_search) {
      PortfolioChoice action;
      action.mode = "causal_cnn";
      action.expert_mask = PostR1Experts::kCausalCnn;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#endif
#if FX4_VIRTUAL_REPLAY
    if (config.scr2_cost_search) {
      PortfolioChoice action;
      action.mode = "scr2_vr_cost";
      action.vr_min_length = kCostPositiveScr2;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
#endif
  } else if (config.portfolio_phase ==
      SearchConfig::PortfolioPhase::kDonorBeam) {
    for (const DonorSequence& sequence : sequences) {
      PortfolioChoice action;
      action.mode = "donor_bundle_" + std::to_string(sequence.size());
      action.sequence = sequence;
      action.expert_mask = PostR1Experts::kDonorProfile;
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
  } else {
    const std::vector<PortfolioChoice> candidates =
        BuildWinningPortfolioBeam(region, config);
    for (const PortfolioChoice& action : candidates) {
      if (!add_strength_sweep(action)) return SearchResult::kFailed;
    }
  }
  const uint64_t start_size = encoder->ProjectedFinalOutputSize();
  uint64_t span_start = encoder->OutputSize();
  ProbeMessage baseline;
  baseline.ok = 1;
  size_t span_index = 0;
  const bool cost_positive_replay = std::any_of(
      branches.begin(), branches.end(), [](const InPlaceTrial& branch) {
        return branch.choice.vr_min_length == kCostPositiveScr2;
      });
  std::vector<std::uint16_t> replay_probabilities;
  if (cost_positive_replay) replay_probabilities.reserve(size * 8u);
  for (size_t index = 0; index < size; ++index) {
    const uint8_t byte = static_cast<uint8_t>(bytes[index]);
    unsigned int prefix = 1;
    for (unsigned int bit_position = 0; bit_position < 8; ++bit_position) {
      const int bit = (byte >> (7 - bit_position)) & 1;
      const float base_probability = predictor->Predict();
      if (cost_positive_replay) {
        replay_probabilities.push_back(
            static_cast<std::uint16_t>(
                Encoder::DiscretizeProbability(base_probability)));
      }
      for (InPlaceTrial& branch : branches) {
        if (branch.choice.vr_min_length == kCostPositiveScr2) continue;
        float probability = base_probability;
        if (branch.expert) {
          predictor->SetPostR1BranchSignals(branch.expert.get());
          probability = branch.expert->Predict(
              base_probability, prefix, bit_position);
        }
        bool replay_known = false;
        if (branch.replay_index < branch.replay_events.size()) {
          const ReplayEvent& event =
              branch.replay_events[branch.replay_index];
          const std::size_t event_end = event.local_offset +
              scr2::kTokens[event.pattern].length;
          replay_known = index >= event.local_offset && index < event_end;
        }
        if (!replay_known) {
          branch.encoder->EncodeRawBit(
              bit, Encoder::DiscretizeProbability(probability));
        }
      }
      EncodeBaselineRangeBit(bit, base_probability, encoder);
      for (InPlaceTrial& branch : branches) {
        if (branch.expert) branch.expert->Perceive(bit);
      }
      predictor->Perceive(bit);
      prefix = (prefix << 1) | static_cast<unsigned int>(bit);
    }
    for (InPlaceTrial& branch : branches) {
      if (branch.expert) branch.expert->ByteUpdate(byte);
      if (branch.replay_index < branch.replay_events.size()) {
        const ReplayEvent& event =
            branch.replay_events[branch.replay_index];
        const std::size_t event_end = event.local_offset +
            scr2::kTokens[event.pattern].length;
        if (index + 1 == event_end) ++branch.replay_index;
      }
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
  for (InPlaceTrial& branch : branches) {
    if (branch.choice.vr_min_length != kCostPositiveScr2) continue;
    branch.replay_events = SelectCostPositiveScr2Events(
        bytes, size, position, replay_probabilities);
    branch.choice.vr_event_count = static_cast<std::uint32_t>(
        branch.replay_events.size());
    if (!branch.replay_events.empty()) {
      const std::uint64_t side = StandaloneVirtualReplayPlanCost(
          position, branch.replay_events);
      if (side == UINT64_MAX) return SearchResult::kFailed;
      branch.choice.side_bytes += side;
    }

    std::size_t replay_index = 0;
    for (std::size_t index = 0; index < size; ++index) {
      bool replay_known = false;
      if (replay_index < branch.replay_events.size()) {
        const ReplayEvent& event = branch.replay_events[replay_index];
        const std::size_t event_end = event.local_offset +
            scr2::kTokens[event.pattern].length;
        replay_known =
            index >= event.local_offset && index < event_end;
        if (index + 1 == event_end) ++replay_index;
      }
      if (replay_known) continue;
      const std::uint8_t byte =
          static_cast<std::uint8_t>(bytes[index]);
      for (unsigned int bit_position = 0;
           bit_position < 8; ++bit_position) {
        const int bit = (byte >> (7 - bit_position)) & 1;
        branch.encoder->EncodeRawBit(
            bit, replay_probabilities[index * 8u + bit_position]);
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

  PortfolioChoice best;
  best.valid = true;
  best.mode = "baseline";
  best.payload_bytes = baseline.payload_bytes;
  best.side_bytes = 0;
  for (InPlaceTrial& branch : branches) {
    branch.choice.payload_bytes =
        branch.encoder->ProjectedFinalOutputSize() - start_size;
    const uint64_t trial_id = (*next_trial_id)++;
    ++*new_trials;
    if (branch.choice.vr_min_length == kCostPositiveScr2 &&
        branch.choice.payload_bytes < baseline.payload_bytes &&
        !AppendPortfolioReplayEvents(ledger_base, region,
            branch.choice.mode, position, branch.replay_events)) {
      return SearchResult::kFailed;
    }
    if (!AppendPortfolioTrial(ledger_base, region,
        static_cast<uint32_t>(size), trial_id, branch.choice,
        baseline.payload_bytes, config.near_loss_bytes)) {
      return SearchResult::kFailed;
    }
    if (branch.choice.expert_mask == PostR1Experts::kDonorProfile) {
      (*trials)[TrialKey(region, branch.choice.sequence)] =
          {true, branch.choice.payload_bytes};
    }
    const uint64_t candidate_total =
        branch.choice.payload_bytes + branch.choice.side_bytes;
    const uint64_t best_total = best.payload_bytes + best.side_bytes;
    if (candidate_total < best_total ||
        (candidate_total == best_total &&
         branch.choice.side_bytes < best.side_bytes)) {
      best = branch.choice;
    }
  }

  // The legacy selection ledger remains a baseline-resume marker. The
  // portfolio ledger below is authoritative for selective winners.
  if (!AppendPortfolioSelection(ledger_base, region,
      static_cast<uint32_t>(size), best, baseline.payload_bytes,
      static_cast<uint32_t>(branches.size()),
      *new_trials - region_trial_start) ||
      !AppendSelection(ledger_base, region, *selection,
      static_cast<uint32_t>(branches.size()),
      *new_trials - region_trial_start, 0)) {
    return SearchResult::kFailed;
  }
  const int64_t best_net =
      static_cast<int64_t>(baseline.payload_bytes) -
      static_cast<int64_t>(best.payload_bytes + best.side_bytes);
  if (!WriteStatus(ledger_base,
      best_net > 0 ? "portfolio_winner" : "portfolio_baseline",
      region, *new_trials, best.sequence, best_net)) {
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
    bool pretrain_dictionary, bool enable_transformer6m,
    DonorPlan* donor_plan,
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

  Predictor predictor(vocab, false, enable_transformer6m);
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
  phase_individual_trials.assign(recipient_spans.size(), {});
  phase_donor_trials.assign(recipient_spans.size(), {});
  if (config.portfolio_phase ==
          SearchConfig::PortfolioPhase::kIndividual &&
      (config.topology_search || config.causal_cnn_search ||
       config.scr2_cost_search)) {
    const std::string current_trials = PortfolioTrialPath(ledger_path);
    if (access(current_trials.c_str(), F_OK) == 0 &&
        !LoadPortfolioTrials(current_trials.c_str(), recipient_spans,
            &phase_individual_trials)) {
      std::fprintf(stderr,
          "cannot reload individual specialist trial ledger: %s\n",
          current_trials.c_str());
      close(input_fd);
      return false;
    }
  }
  if (config.portfolio_phase !=
      SearchConfig::PortfolioPhase::kIndividual) {
    const char* individual_path =
        getenv("FX4_WINNER_INDIVIDUAL_TRIALS");
    if (!LoadPortfolioTrials(
        individual_path, recipient_spans, &phase_individual_trials)) {
      std::fprintf(stderr,
          "cannot load individual trial ledger: %s\n",
          individual_path ? individual_path : "(unset)");
      close(input_fd);
      return false;
    }
  }
  if (config.portfolio_phase ==
      SearchConfig::PortfolioPhase::kCombine) {
    const char* donor_path = getenv("FX4_WINNER_DONOR_TRIALS");
    if (!LoadPortfolioTrials(
        donor_path, recipient_spans, &phase_donor_trials)) {
      std::fprintf(stderr,
          "cannot load donor-bundle trial ledger: %s\n",
          donor_path ? donor_path : "(unset)");
      close(input_fd);
      return false;
    }
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
              &predictor, donor_plan, nullptr)) {
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
    const bool completed_selection =
        region < selections.size() && selections[region].valid;
    const bool topology_catchup = completed_selection &&
        config.portfolio_phase == SearchConfig::PortfolioPhase::kIndividual &&
        config.topology_search && !TopologyTrialsComplete(region);
    const bool causal_cnn_catchup = completed_selection &&
        config.portfolio_phase == SearchConfig::PortfolioPhase::kIndividual &&
        config.causal_cnn_search && !CausalCnnTrialsComplete(region);
    const bool scr2_cost_catchup = completed_selection &&
        config.portfolio_phase == SearchConfig::PortfolioPhase::kIndividual &&
        config.scr2_cost_search && !Scr2CostTrialComplete(region);
    const bool specialist_catchup =
        topology_catchup || causal_cnn_catchup || scr2_cost_catchup;
    if (completed_selection && !specialist_catchup) {
      selection = selections[region];
    } else if (SearchRecipient(region) &&
        region >= config.start_region &&
        (config.max_regions == 0 || searched_regions < config.max_regions)) {
      SearchConfig region_config = config;
      if (specialist_catchup) {
        region_config.topology_only = true;
        region_config.causal_cnn_only = true;
        region_config.scr2_cost_only = true;
        region_config.topology_search = topology_catchup;
        region_config.causal_cnn_search = causal_cnn_catchup;
        region_config.scr2_cost_search = scr2_cost_catchup;
      }
      const SearchResult result = SearchRegion(region, buffer.data(), count,
          position, &encoder, &predictor, input_fd, donor_plan,
          phrase_index, region_config, ledger_path, &trials, &next_trial_id,
          &new_trials, &selection);
      // Trial objects contain large, short-lived profile tables. Return their
      // freed pages instead of retaining them for this multi-day process.
      malloc_trim(0);
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

    if (!region_encoded && !EncodeRegion(buffer.data(), count, position,
        &encoder, &predictor, donor_plan, nullptr)) {
      close(input_fd);
      return false;
    }
    const uint64_t actual_bytes =
        encoder.ProjectedFinalOutputSize() - start_size;
    const uint64_t replay_difference = selection.baseline_bytes > actual_bytes
        ? selection.baseline_bytes - actual_bytes
        : actual_bytes - selection.baseline_bytes;
    if (selection.valid && selection.baseline_bytes != 0 &&
        replay_difference > 1) {
      fprintf(stderr,
          "baseline replay mismatch at region %u: expected %" PRIu64
          ", got %" PRIu64 "\n",
          region, selection.baseline_bytes, actual_bytes);
      close(input_fd);
      return false;
    }
    if (selection.valid && selection.baseline_bytes != 0 &&
        replay_difference == 1) {
      fprintf(stderr,
          "baseline replay boundary adjustment at region %u: expected %" PRIu64
          ", got %" PRIu64 " (accepted: 1-byte coder attribution)\n",
          region, selection.baseline_bytes, actual_bytes);
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
