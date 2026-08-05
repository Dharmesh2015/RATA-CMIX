#include "donor_fork_discovery.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "coder/encoder.h"
#include "donor_plan.h"
#include "donor_winner_search.h"
#include "fx4_config.h"
#include "predictor.h"
#include "preprocess/preprocessor.h"

namespace {

constexpr const char* kEdgeHeader =
    "recipient_region,recipient_offset,candidate_rank,donor_offset,"
    "baseline_payload_bytes,donor_payload_bytes,gain_bytes,status\n";
constexpr const char* kSelectionHeader =
    "recipient_region,recipient_offset,selected_donor_offset,"
    "baseline_payload_bytes,selected_payload_bytes,gain_bytes,"
    "candidate_count,successful_candidates\n";

bool discovery_completed = false;

enum class WorkerResult {
  kFailed,
  kPaused,
  kComplete,
};

struct RecordedEdge {
  bool valid = false;
  bool ok = false;
  uint64_t baseline_bytes = 0;
  uint64_t donor_bytes = 0;
};

struct RecordedSelection {
  bool valid = false;
  uint32_t donor_offset = DonorPlan::kNoDonor;
  uint64_t baseline_bytes = 0;
  uint64_t selected_bytes = 0;
  uint32_t candidate_count = 0;
  uint32_t successful_candidates = 0;
};

struct ProbeMessage {
  uint64_t payload_bytes = 0;
  uint32_t ok = 0;
};

uint64_t EdgeKey(uint32_t region, uint32_t donor_offset) {
  return (static_cast<uint64_t>(region) << 32) | donor_offset;
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

std::string WinnersPath(const std::string& ledger_path) {
  return ledger_path + ".winners.csv";
}

std::string SelectionsPath(const std::string& ledger_path) {
  return ledger_path + ".selected.csv";
}

std::string StatusPath(const std::string& ledger_path) {
  return ledger_path + ".status";
}

std::string CompletePath(const std::string& ledger_path) {
  return ledger_path + ".complete";
}

std::string PausedPath(const std::string& ledger_path) {
  return ledger_path + ".paused";
}

bool EnsureCsv(const std::string& path, const char* header) {
  struct stat info {};
  if (stat(path.c_str(), &info) == 0 && info.st_size != 0) {
    std::ifstream input(path);
    std::string first;
    return input.is_open() && std::getline(input, first) &&
        first + "\n" == header;
  }
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) return false;
  const bool ok = fputs(header, file) >= 0 && SyncFile(file);
  fclose(file);
  return ok;
}

bool EnsureLedgers(const std::string& ledger_path) {
  return EnsureCsv(ledger_path, kEdgeHeader) &&
      EnsureCsv(WinnersPath(ledger_path), kEdgeHeader) &&
      EnsureCsv(SelectionsPath(ledger_path), kSelectionHeader);
}

bool AppendRow(const std::string& path, const char* row, size_t length) {
  FILE* file = fopen(path.c_str(), "ab");
  if (!file) return false;
  const bool ok = fwrite(row, 1, length, file) == length && SyncFile(file);
  fclose(file);
  return ok;
}

bool WriteStatus(const std::string& ledger_path, const char* phase,
    uint32_t region, uint32_t donor_offset, int64_t gain,
    uint64_t new_trials) {
  const std::string path = StatusPath(ledger_path);
  const std::string temporary = path + ".tmp";
  FILE* file = fopen(temporary.c_str(), "wb");
  if (!file) return false;
  fprintf(file,
      "phase=%s\nrecipient_region=%u\nrecipient_offset=%" PRIu64
      "\ndonor_offset=%u\ngain_bytes=%" PRId64
      "\nnew_trials_this_run=%" PRIu64 "\npid=%ld\n",
      phase, region,
      static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      donor_offset, gain, new_trials, static_cast<long>(getpid()));
  const bool synced = SyncFile(file);
  fclose(file);
  if (!synced) return false;
  return rename(temporary.c_str(), path.c_str()) == 0;
}

bool LoadEdges(const std::string& ledger_path,
    std::unordered_map<uint64_t, RecordedEdge>* edges) {
  std::ifstream input(ledger_path);
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line) || line + "\n" != kEdgeHeader) return false;
  while (std::getline(input, line)) {
    unsigned region = 0;
    unsigned long long recipient_offset = 0;
    unsigned rank = 0;
    unsigned donor_offset = 0;
    unsigned long long baseline_bytes = 0;
    unsigned long long donor_bytes = 0;
    long long gain_bytes = 0;
    char status[16] = {};
    if (sscanf(line.c_str(), "%u,%llu,%u,%u,%llu,%llu,%lld,%15s",
        &region, &recipient_offset, &rank, &donor_offset, &baseline_bytes,
        &donor_bytes, &gain_bytes, status) != 8) {
      return false;
    }
    RecordedEdge edge;
    edge.valid = true;
    edge.ok = strcmp(status, "error") != 0;
    edge.baseline_bytes = baseline_bytes;
    edge.donor_bytes = donor_bytes;
    (*edges)[EdgeKey(region, donor_offset)] = edge;
  }
  return true;
}

bool LoadSelections(const std::string& ledger_path,
    std::vector<RecordedSelection>* selections) {
  std::ifstream input(SelectionsPath(ledger_path));
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line) || line + "\n" != kSelectionHeader) {
    return false;
  }
  while (std::getline(input, line)) {
    unsigned region = 0;
    unsigned long long recipient_offset = 0;
    unsigned donor_offset = 0;
    unsigned long long baseline_bytes = 0;
    unsigned long long selected_bytes = 0;
    long long gain_bytes = 0;
    unsigned candidate_count = 0;
    unsigned successful_candidates = 0;
    if (sscanf(line.c_str(), "%u,%llu,%u,%llu,%llu,%lld,%u,%u",
        &region, &recipient_offset, &donor_offset, &baseline_bytes,
        &selected_bytes, &gain_bytes, &candidate_count,
        &successful_candidates) != 8 ||
        region >= selections->size()) {
      return false;
    }
    RecordedSelection selection;
    selection.valid = true;
    selection.donor_offset = donor_offset;
    selection.baseline_bytes = baseline_bytes;
    selection.selected_bytes = selected_bytes;
    selection.candidate_count = candidate_count;
    selection.successful_candidates = successful_candidates;
    (*selections)[region] = selection;
  }
  return true;
}

bool AppendEdge(const std::string& ledger_path, uint32_t region,
    uint32_t rank, uint32_t donor_offset, uint64_t baseline_bytes,
    const ProbeMessage& donor, uint64_t new_trials) {
  const int64_t gain = donor.ok
      ? static_cast<int64_t>(baseline_bytes) -
          static_cast<int64_t>(donor.payload_bytes)
      : 0;
  const char* status =
      !donor.ok ? "error" : gain > 0 ? "donor" : "baseline";
  char row[320] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%u,%u,%" PRIu64 ",%" PRIu64
      ",%" PRId64 ",%s\n",
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      rank, donor_offset, baseline_bytes, donor.payload_bytes, gain, status);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(row)) return false;
  if (!AppendRow(ledger_path, row, static_cast<size_t>(length))) return false;
  if (gain > 0 &&
      !AppendRow(WinnersPath(ledger_path), row,
          static_cast<size_t>(length))) {
    return false;
  }
  return WriteStatus(
      ledger_path, "edge", region, donor_offset, gain, new_trials);
}

bool AppendSelection(const std::string& ledger_path, uint32_t region,
    const RecordedSelection& selection, uint64_t new_trials) {
  const int64_t gain = static_cast<int64_t>(selection.baseline_bytes) -
      static_cast<int64_t>(selection.selected_bytes);
  char row[320] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64
      ",%" PRId64 ",%u,%u\n",
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      selection.donor_offset, selection.baseline_bytes,
      selection.selected_bytes, gain, selection.candidate_count,
      selection.successful_candidates);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(row)) return false;
  if (!AppendRow(
      SelectionsPath(ledger_path), row, static_cast<size_t>(length))) {
    return false;
  }
  return WriteStatus(ledger_path, "selected", region,
      selection.donor_offset, gain, new_trials);
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

uint64_t EnvironmentU64(const char* name) {
  const char* text = getenv(name);
  if (!text || !*text) return 0;
  char* end = nullptr;
  const unsigned long long value = strtoull(text, &end, 10);
  return *end == '\0' ? value : 0;
}

bool EncodeRegion(const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, DonorPlan* donor_plan) {
  for (size_t index = 0; index < size; ++index) {
    const uint8_t byte = static_cast<uint8_t>(bytes[index]);
    for (int bit = 7; bit >= 0; --bit) {
      encoder->Encode((byte >> bit) & 1);
    }
    donor_plan->CaptureByte(position + index, byte);
  }
  return true;
}

bool ProbeRegion(uint32_t donor_offset, bool replay_donor,
    const char* bytes, size_t size, uint64_t position, Encoder* encoder,
    Predictor* predictor, DonorPlan* donor_plan, ProbeMessage* result) {
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
    const uint32_t region =
        static_cast<uint32_t>(position / DonorPlan::kChunkSize);
    const bool replay_ok = !replay_donor ||
        donor_plan->ReplayCandidate(region, donor_offset, predictor);
    message.ok = replay_ok &&
        EncodeRegion(bytes, size, position, encoder, donor_plan) ? 1u : 0u;
    message.payload_bytes = encoder->OutputSize() - start_size;
    const bool sent =
        WriteAll(message_pipe[1], &message, sizeof(message));
    close(message_pipe[1]);
    _exit(sent && message.ok ? 0 : 2);
  }

  close(message_pipe[1]);
  ProbeMessage message;
  const bool read_ok =
      ReadAll(message_pipe[0], &message, sizeof(message));
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

bool ContainsDonor(
    const std::vector<uint32_t>& candidates, uint32_t donor_offset) {
  return donor_offset == DonorPlan::kNoDonor ||
      std::find(candidates.begin(), candidates.end(), donor_offset) !=
          candidates.end();
}

WorkerResult RunWorker(const std::string& input_path,
    const std::string& scratch_output_path, uint64_t input_bytes,
    const std::vector<bool>& vocab, FILE* dictionary,
    bool pretrain_dictionary, DonorPlan* donor_plan,
    const std::string& ledger_path) {
  std::ifstream input(input_path, std::ios::in | std::ios::binary);
  std::ofstream output(
      scratch_output_path, std::ios::out | std::ios::binary);
  if (!input.is_open() || !output.is_open()) return WorkerResult::kFailed;

  Predictor predictor(vocab);
  if (pretrain_dictionary) preprocessor::Pretrain(&predictor, dictionary);
  Encoder encoder(&output, &predictor);

  const size_t complete_regions =
      static_cast<size_t>(input_bytes / DonorPlan::kChunkSize);
  std::unordered_map<uint64_t, RecordedEdge> edges;
  std::vector<RecordedSelection> selections(complete_regions);
  if (!LoadEdges(ledger_path, &edges) ||
      !LoadSelections(ledger_path, &selections)) {
    return WorkerResult::kFailed;
  }

  const uint64_t max_new_trials =
      EnvironmentU64("FX4_DONOR_MAX_NEW_TRIALS");
  uint64_t new_trials = 0;
  std::vector<char> buffer(DonorPlan::kChunkSize);
  uint64_t position = 0;
  while (position < input_bytes) {
    const size_t wanted = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), input_bytes - position));
    input.read(buffer.data(), static_cast<std::streamsize>(wanted));
    const size_t count = static_cast<size_t>(input.gcount());
    if (count != wanted) return WorkerResult::kFailed;

    const uint32_t region =
        static_cast<uint32_t>(position / DonorPlan::kChunkSize);
    const std::vector<uint32_t>& candidates =
        donor_plan->CandidateOffsets(region);
    const uint64_t start_size = encoder.OutputSize();

    if (count == DonorPlan::kChunkSize && !candidates.empty()) {
      if (region < selections.size() && selections[region].valid) {
        const RecordedSelection& selection = selections[region];
        if (selection.candidate_count != candidates.size() ||
            !ContainsDonor(candidates, selection.donor_offset)) {
          return WorkerResult::kFailed;
        }
        if (selection.donor_offset != DonorPlan::kNoDonor &&
            !donor_plan->ReplayCandidate(
                region, selection.donor_offset, &predictor)) {
          return WorkerResult::kFailed;
        }
        if (!EncodeRegion(buffer.data(), count, position, &encoder,
            donor_plan) ||
            encoder.OutputSize() - start_size != selection.selected_bytes) {
          return WorkerResult::kFailed;
        }
      } else {
        uint64_t baseline_bytes = 0;
        for (uint32_t donor_offset : candidates) {
          const auto found = edges.find(EdgeKey(region, donor_offset));
          if (found != edges.end() && found->second.valid) {
            baseline_bytes = found->second.baseline_bytes;
            break;
          }
        }
        if (baseline_bytes == 0) {
          ProbeMessage baseline;
          if (!ProbeRegion(DonorPlan::kNoDonor, false, buffer.data(), count,
              position, &encoder, &predictor, donor_plan, &baseline) ||
              !baseline.ok) {
            return WorkerResult::kFailed;
          }
          baseline_bytes = baseline.payload_bytes;
        }

        RecordedSelection selection;
        selection.valid = true;
        selection.baseline_bytes = baseline_bytes;
        selection.selected_bytes = baseline_bytes;
        selection.candidate_count =
            static_cast<uint32_t>(candidates.size());

        for (size_t rank = 0; rank < candidates.size(); ++rank) {
          const uint32_t donor_offset = candidates[rank];
          const uint64_t key = EdgeKey(region, donor_offset);
          auto found = edges.find(key);
          if (found == edges.end()) {
            if (max_new_trials != 0 && new_trials >= max_new_trials) {
              unlink(CompletePath(ledger_path).c_str());
              FILE* paused = fopen(PausedPath(ledger_path).c_str(), "wb");
              if (!paused) return WorkerResult::kFailed;
              fprintf(paused,
                  "recipient_region=%u\ncandidate_rank=%zu\n"
                  "donor_offset=%u\nnew_trials=%" PRIu64 "\n",
                  region, rank, donor_offset, new_trials);
              const bool pause_ok = SyncFile(paused);
              fclose(paused);
              if (!pause_ok ||
                  !WriteStatus(ledger_path, "paused", region,
                      donor_offset, 0, new_trials)) {
                return WorkerResult::kFailed;
              }
              return WorkerResult::kPaused;
            }

            ProbeMessage donor;
            if (!ProbeRegion(donor_offset, true, buffer.data(), count,
                position, &encoder, &predictor, donor_plan, &donor)) {
              return WorkerResult::kFailed;
            }
            ++new_trials;
            if (!AppendEdge(ledger_path, region,
                static_cast<uint32_t>(rank), donor_offset, baseline_bytes,
                donor, new_trials)) {
              return WorkerResult::kFailed;
            }
            RecordedEdge edge;
            edge.valid = true;
            edge.ok = donor.ok != 0;
            edge.baseline_bytes = baseline_bytes;
            edge.donor_bytes = donor.payload_bytes;
            found = edges.emplace(key, edge).first;
          }

          const RecordedEdge& edge = found->second;
          if (edge.baseline_bytes != baseline_bytes) {
            return WorkerResult::kFailed;
          }
          if (!edge.ok) continue;
          ++selection.successful_candidates;
          if (edge.donor_bytes < selection.selected_bytes) {
            selection.selected_bytes = edge.donor_bytes;
            selection.donor_offset = donor_offset;
          }
        }

        if (!AppendSelection(
            ledger_path, region, selection, new_trials)) {
          return WorkerResult::kFailed;
        }
        selections[region] = selection;
        if (selection.donor_offset != DonorPlan::kNoDonor &&
            !donor_plan->ReplayCandidate(
                region, selection.donor_offset, &predictor)) {
          return WorkerResult::kFailed;
        }
        if (!EncodeRegion(buffer.data(), count, position, &encoder,
            donor_plan) ||
            encoder.OutputSize() - start_size != selection.selected_bytes) {
          return WorkerResult::kFailed;
        }
      }
    } else if (!EncodeRegion(
        buffer.data(), count, position, &encoder, donor_plan)) {
      return WorkerResult::kFailed;
    }
    position += count;
  }

  encoder.Flush();
  output.close();
  if (!output.good()) return WorkerResult::kFailed;

  unlink(PausedPath(ledger_path).c_str());
  FILE* complete = fopen(CompletePath(ledger_path).c_str(), "wb");
  if (!complete) return WorkerResult::kFailed;
  fprintf(complete, "regions=%zu\ninput_bytes=%" PRIu64
      "\nnew_trials=%" PRIu64 "\npid=%ld\n",
      complete_regions, input_bytes, new_trials,
      static_cast<long>(getpid()));
  const bool ok = SyncFile(complete);
  fclose(complete);
  return ok ? WorkerResult::kComplete : WorkerResult::kFailed;
}

}  // namespace

bool RunDonorForkDiscovery(const std::string& input_path,
    const std::string& scratch_output_path, uint64_t input_bytes,
    const std::vector<bool>& vocab, FILE* dictionary,
    bool pretrain_dictionary, DonorPlan* donor_plan,
    uint64_t* output_bytes) {
#if !FX4_DONOR_FORK_DISCOVERY
  (void)input_path;
  (void)scratch_output_path;
  (void)input_bytes;
  (void)vocab;
  (void)dictionary;
  (void)pretrain_dictionary;
  (void)donor_plan;
  (void)output_bytes;
  return false;
#else
  const char* path = getenv("FX4_DONOR_DISCOVERY_RESULTS");
  if (!path || !*path || !donor_plan || donor_plan->empty() ||
      !donor_plan->discovery_candidates()) {
    return false;
  }
  const std::string ledger_path(path);
  const char* winner_mode = getenv("FX4_DONOR_WINNER_SEARCH");
  if (winner_mode && *winner_mode && strcmp(winner_mode, "0") != 0) {
    const bool ok = RunDonorWinnerSearch(input_path, scratch_output_path,
        input_bytes, vocab, dictionary, pretrain_dictionary, donor_plan,
        ledger_path, output_bytes);
    discovery_completed = ok;
    return ok;
  }
  if (!EnsureLedgers(ledger_path)) return false;
  unlink(CompletePath(ledger_path).c_str());
  unlink(PausedPath(ledger_path).c_str());

  const WorkerResult result = RunWorker(input_path, scratch_output_path,
      input_bytes, vocab, dictionary, pretrain_dictionary, donor_plan,
      ledger_path);
  if (result == WorkerResult::kFailed) return false;

  struct stat info {};
  *output_bytes =
      result == WorkerResult::kComplete &&
      stat(scratch_output_path.c_str(), &info) == 0
          ? info.st_size
          : 0;
  discovery_completed = true;
  return true;
#endif
}

bool DonorForkDiscoveryCompleted() {
  return discovery_completed;
}
