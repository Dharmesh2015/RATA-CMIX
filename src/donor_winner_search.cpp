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

struct DonorWindow {
  uint32_t offset = 0;
  uint32_t length = 0;
  uint32_t score = 0;
};

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
  bool planned_only = false;
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
  config.planned_only =
      EnvironmentU32("FX4_WINNER_PLANNED_ONLY", 0, 0, 1) != 0;
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
      EnsureCsv(CandidatePath(base), kCandidateHeader);
}

bool AppendRow(const std::string& path, const std::string& row) {
  FILE* output = fopen(path.c_str(), "ab");
  if (!output) return false;
  const bool ok = fwrite(row.data(), 1, row.size(), output) == row.size() &&
      SyncFile(output);
  fclose(output);
  return ok;
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
    uint64_t trial_id = 0;
    uint64_t payload = 0;
    if (!ParseU64(fields[0], &region) || !ParseU64(fields[2], &trial_id) ||
        !ParseU64(fields[8], &payload) || region > 0xffff) {
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
    uint64_t baseline = 0;
    uint64_t selected = 0;
    if (!ParseU64(fields[0], &region) || !ParseU64(fields[2], &baseline) ||
        !ParseU64(fields[3], &selected) || region >= selections->size()) {
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
      static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
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
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
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

bool AppendSelection(const std::string& base, uint32_t region,
    const SelectionRecord& selection, uint32_t candidate_count,
    uint64_t exact_trials) {
  const int64_t gain = static_cast<int64_t>(selection.baseline_bytes) -
      static_cast<int64_t>(selection.selected_bytes);
  const int64_t metadata = selection.sequence.empty()
      ? 0 : 3 + static_cast<int64_t>(7 * selection.sequence.size());
  const std::string donors = selection.sequence.empty()
      ? "-" : SequenceText(selection.sequence);
  char row[2048] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRId64
      ",%" PRId64 ",%zu,%s,%u,%" PRIu64 "\n",
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      selection.baseline_bytes, selection.selected_bytes, gain,
      gain - metadata, selection.sequence.size(), donors.c_str(),
      candidate_count, exact_trials);
  return length > 0 && static_cast<size_t>(length) < sizeof(row) &&
      AppendRow(SelectionPath(base),
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
        static_cast<uint64_t>(region) * DonorPlan::kChunkSize +
        static_cast<uint64_t>(order[rank]) * kLossSpanBytes;
    char row[256] = {};
    const int length = snprintf(row, sizeof(row),
        "%u,%" PRIu64 ",%zu,%" PRIu64 ",%u\n",
        region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
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
        region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
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

bool ReplaySequence(int input_fd, const DonorSequence& sequence,
    Predictor* predictor) {
  std::vector<uint8_t> bytes;
  for (const DonorWindow& donor : sequence) {
    if (!ReadWindow(input_fd, donor.offset, donor.length, &bytes)) {
      return false;
    }
    for (uint8_t byte : bytes) {
      for (int bit = 7; bit >= 0; --bit) {
        predictor->Predict();
        predictor->Perceive((byte >> bit) & 1);
      }
    }
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
  const uint64_t start_size = encoder->OutputSize();
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
    const bool replay_ok = ReplaySequence(input_fd, sequence, predictor);
    message.ok = replay_ok && EncodeRegion(bytes, size, position, encoder,
        nullptr, collect_spans ? &message : nullptr) ? 1u : 0u;
    message.payload_bytes = encoder->OutputSize() - start_size;
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
      const char* input, const ProbeMessage& baseline, uint32_t top_spans,
      uint64_t recipient_offset, uint32_t limit) const {
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
      for (size_t local = start;
           local + kAlignment <= start + kLossSpanBytes;
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

uint64_t TotalBytes(const ScoredSequence& scored) {
  return scored.payload_bytes + (scored.sequence.empty()
      ? 0u : 3u + 7u * scored.sequence.size());
}

uint64_t TotalBytes(const SelectionRecord& selection) {
  return selection.selected_bytes + (selection.sequence.empty()
      ? 0u : 3u + 7u * selection.sequence.size());
}

bool BeatsSelection(
    const ScoredSequence& scored, const SelectionRecord& selection) {
  return TotalBytes(scored) < TotalBytes(selection);
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
};

SearchResult SearchRegion(uint32_t region, const char* bytes, size_t size,
    uint64_t position, Encoder* encoder, Predictor* predictor, int input_fd,
    DonorPlan* donor_plan, const CausalPhraseIndex& phrase_index,
    const SearchConfig& config, const std::string& ledger_base,
    std::unordered_map<std::string, TrialRecord>* trials,
    uint64_t* next_trial_id, uint64_t* new_trials,
    SelectionRecord* selection) {
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
  for (const DonorPlan::CandidateSpec& candidate :
       donor_plan->CandidateSpecs(region)) {
    AddUniqueWindow({candidate.donor_offset, candidate.length,
        std::numeric_limits<uint32_t>::max()}, position, &initial);
  }
  const auto matches = phrase_index.Find(bytes, baseline, config.top_spans,
      position, config.candidate_offsets);
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
  const size_t initial_limit = config.candidate_offsets +
      donor_plan->CandidateSpecs(region).size();
  if (initial.size() > initial_limit) initial.resize(initial_limit);
  if (!WriteCandidateRows(ledger_base, region, initial)) {
    return SearchResult::kFailed;
  }

  bool paused = false;
  if (config.planned_only) {
    DonorSequence planned;
    const size_t count = std::min<size_t>(
        config.planned_donors, initial.size());
    if (count != 0) {
      planned.assign(initial.begin(), initial.begin() + count);
      ScoredSequence scored;
      if (Evaluate(region, "planned", planned, baseline.payload_bytes,
          baseline.payload_bytes, bytes, size, position, encoder, predictor,
          input_fd, config, ledger_base, trials, next_trial_id, new_trials,
          &paused, &scored) && BeatsSelection(scored, *selection)) {
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
    if (BeatsSelection(scored, *selection)) {
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
        if (BeatsSelection(scored, *selection)) {
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
          if (BeatsSelection(scored, *selection)) {
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

  const size_t complete_regions =
      static_cast<size_t>(input_bytes / DonorPlan::kChunkSize);
  std::unordered_map<std::string, TrialRecord> trials;
  std::vector<SelectionRecord> selections(complete_regions);
  uint64_t next_trial_id = 1;
  if (!LoadTrials(ledger_path, &trials, &next_trial_id) ||
      !LoadSelections(ledger_path, &selections)) {
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

  while (position < input_bytes) {
    const size_t wanted = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), input_bytes - position));
    input.read(buffer.data(), static_cast<std::streamsize>(wanted));
    const size_t count = static_cast<size_t>(input.gcount());
    if (count != wanted) {
      close(input_fd);
      return false;
    }

    const uint32_t region =
        static_cast<uint32_t>(position / DonorPlan::kChunkSize);
    const uint64_t start_size = encoder.OutputSize();
    SelectionRecord selection;
    if (count == DonorPlan::kChunkSize && region < selections.size() &&
        selections[region].valid) {
      selection = selections[region];
    } else if (count == DonorPlan::kChunkSize &&
        region >= config.start_region &&
        (config.max_regions == 0 || searched_regions < config.max_regions)) {
      const SearchResult result = SearchRegion(region, buffer.data(), count,
          position, &encoder, &predictor, input_fd, donor_plan,
          phrase_index, config, ledger_path, &trials, &next_trial_id,
          &new_trials, &selection);
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

    if (!selection.sequence.empty() &&
        !ReplaySequence(input_fd, selection.sequence, &predictor)) {
      close(input_fd);
      return false;
    }
    if (!EncodeRegion(buffer.data(), count, position, &encoder,
        donor_plan, nullptr)) {
      close(input_fd);
      return false;
    }
    const uint64_t actual_bytes = encoder.OutputSize() - start_size;
    if (selection.valid && selection.selected_bytes != 0 &&
        actual_bytes != selection.selected_bytes) {
      fprintf(stderr,
          "winner replay mismatch at region %u: expected %" PRIu64
          ", got %" PRIu64 "\n",
          region, selection.selected_bytes, actual_bytes);
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
      fprintf(paused,
          "recipient_region=%u\nrecipient_offset=%" PRIu64
          "\nnew_trials=%" PRIu64 "\n",
          region + 1, position, new_trials);
      const bool ok = SyncFile(paused) && WriteStatus(ledger_path,
          "region_limit", region + 1, new_trials, {}, 0);
      fclose(paused);
      close(input_fd);
      *output_bytes = 0;
      return ok;
    }
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
      complete_regions, input_bytes, new_trials,
      static_cast<long>(getpid()));
  const bool ok = SyncFile(complete);
  fclose(complete);
  return ok;
}
