#include "donor_fork_discovery.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "coder/encoder.h"
#include "donor_plan.h"
#include "fx4_config.h"
#include "predictor.h"
#include "preprocess/preprocessor.h"

namespace {

constexpr int kBaselineWins = 0;
constexpr int kDonorWins = 1;
constexpr int kTrialLoserExit = 80;
constexpr int kHandoffExit = 81;

bool discovery_completed = false;

struct RecordedDecision {
  bool valid = false;
  bool donor_won = false;
  uint32_t donor_offset = DonorPlan::kNoDonor;
  uint64_t baseline_bytes = 0;
  uint64_t donor_bytes = 0;
};

struct BaselineMessage {
  uint64_t payload_bytes;
  uint32_t ok;
};

struct DecisionMessage {
  uint64_t baseline_bytes;
  uint64_t donor_bytes;
  uint32_t winner;
  uint32_t ok;
};

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

std::string StatusPath(const std::string& ledger_path) {
  return ledger_path + ".status";
}

std::string CompletePath(const std::string& ledger_path) {
  return ledger_path + ".complete";
}

bool EnsureCsv(const std::string& path, const char* header) {
  struct stat info {};
  if (stat(path.c_str(), &info) == 0 && info.st_size != 0) return true;
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) return false;
  const bool ok = fputs(header, file) >= 0 && SyncFile(file);
  fclose(file);
  return ok;
}

bool EnsureLedgers(const std::string& ledger_path) {
  const char* header =
      "recipient_region,recipient_offset,donor_offset,"
      "baseline_payload_bytes,donor_payload_bytes,gain_bytes,winner\n";
  return EnsureCsv(ledger_path, header) &&
      EnsureCsv(WinnersPath(ledger_path), header);
}

bool LoadDecisions(const std::string& ledger_path,
    std::vector<RecordedDecision>* decisions) {
  std::ifstream input(ledger_path);
  if (!input.is_open()) return false;
  std::string line;
  std::getline(input, line);
  while (std::getline(input, line)) {
    unsigned region = 0;
    unsigned long long recipient_offset = 0;
    unsigned donor_offset = 0;
    unsigned long long baseline_bytes = 0;
    unsigned long long donor_bytes = 0;
    long long gain_bytes = 0;
    char winner[16] = {};
    if (sscanf(line.c_str(), "%u,%llu,%u,%llu,%llu,%lld,%15s",
        &region, &recipient_offset, &donor_offset, &baseline_bytes,
        &donor_bytes, &gain_bytes, winner) != 7) {
      continue;
    }
    if (region >= decisions->size()) return false;
    RecordedDecision& decision = (*decisions)[region];
    decision.valid = true;
    decision.donor_won = strcmp(winner, "donor") == 0;
    decision.donor_offset = donor_offset;
    decision.baseline_bytes = baseline_bytes;
    decision.donor_bytes = donor_bytes;
  }
  return true;
}

bool AppendDecision(const std::string& ledger_path, uint32_t region,
    uint32_t donor_offset, const DecisionMessage& decision) {
  const int64_t gain = static_cast<int64_t>(decision.baseline_bytes) -
      static_cast<int64_t>(decision.donor_bytes);
  const char* winner =
      decision.winner == kDonorWins ? "donor" : "baseline";
  char row[256] = {};
  const int length = snprintf(row, sizeof(row),
      "%u,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRId64 ",%s\n",
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      donor_offset, decision.baseline_bytes, decision.donor_bytes, gain,
      winner);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(row)) return false;

  FILE* ledger = fopen(ledger_path.c_str(), "ab");
  if (!ledger) return false;
  const bool ledger_ok =
      fwrite(row, 1, static_cast<size_t>(length), ledger) ==
          static_cast<size_t>(length) &&
      SyncFile(ledger);
  fclose(ledger);
  if (!ledger_ok) return false;

  if (decision.winner == kDonorWins) {
    const std::string winners_path = WinnersPath(ledger_path);
    FILE* winners = fopen(winners_path.c_str(), "ab");
    if (!winners) return false;
    const bool winner_ok =
        fwrite(row, 1, static_cast<size_t>(length), winners) ==
            static_cast<size_t>(length) &&
        SyncFile(winners);
    fclose(winners);
    if (!winner_ok) return false;
  }

  const std::string status_path = StatusPath(ledger_path);
  FILE* status = fopen(status_path.c_str(), "wb");
  if (!status) return false;
  fprintf(status,
      "last_region=%u\nrecipient_offset=%" PRIu64
      "\ndonor_offset=%u\nbaseline_bytes=%" PRIu64
      "\ndonor_bytes=%" PRIu64 "\ngain_bytes=%" PRId64
      "\nwinner=%s\nwinner_pid=%ld\n",
      region, static_cast<uint64_t>(region) * DonorPlan::kChunkSize,
      donor_offset, decision.baseline_bytes, decision.donor_bytes, gain,
      winner, static_cast<long>(getpid()));
  const bool status_ok = SyncFile(status);
  fclose(status);
  return status_ok;
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

bool ApplyRecordedDecision(const RecordedDecision& decision,
    uint32_t expected_donor, uint64_t position, Predictor* predictor,
    DonorPlan* donor_plan) {
  if (decision.donor_offset != expected_donor) return false;
  if (!decision.donor_won) return true;
  return donor_plan->ReplayAt(position, predictor);
}

bool TrialRegion(const std::string& ledger_path, uint32_t region,
    uint32_t donor_offset, const char* bytes, size_t size, uint64_t position,
    Encoder* encoder, Predictor* predictor, DonorPlan* donor_plan) {
  const uint64_t start_size = encoder->OutputSize();

  // Probe the donor in a disposable child while this process remains at the
  // exact recipient boundary. Keeping only one mutating branch alive avoids
  // the roughly 16 GB peak of two simultaneous full Predictor branches.
  int donor_pipe[2] = {-1, -1};
  if (pipe(donor_pipe) != 0) return false;
  const pid_t donor_probe = fork();
  if (donor_probe < 0) return false;
  if (donor_probe == 0) {
    close(donor_pipe[0]);
    SetCpuFromEnvironment("FX4_DONOR_BASELINE_CPU");
    BaselineMessage donor {};
    const bool replay_ok = donor_plan->ReplayAt(position, predictor);
    donor.ok = replay_ok && EncodeRegion(
        bytes, size, position, encoder, donor_plan) ? 1u : 0u;
    donor.payload_bytes = encoder->OutputSize() - start_size;
    const bool sent = WriteAll(donor_pipe[1], &donor, sizeof(donor));
    close(donor_pipe[1]);
    _exit(sent && donor.ok ? 0 : 2);
  }
  close(donor_pipe[1]);
  BaselineMessage donor {};
  const bool donor_read = ReadAll(donor_pipe[0], &donor, sizeof(donor));
  close(donor_pipe[0]);
  int donor_status = 0;
  if (!donor_read || waitpid(donor_probe, &donor_status, 0) != donor_probe ||
      !WIFEXITED(donor_status) || WEXITSTATUS(donor_status) != 0 ||
      !donor.ok) {
    return false;
  }

  // Probe baseline second. If baseline wins, this child becomes the new
  // long-lived worker. If donor wins, it exits and the untouched parent
  // reproduces the already measured donor path exactly once.
  int baseline_pipe[2] = {-1, -1};
  int decision_pipe[2] = {-1, -1};
  if (pipe(baseline_pipe) != 0 || pipe(decision_pipe) != 0) return false;
  const pid_t baseline_probe = fork();
  if (baseline_probe < 0) return false;
  if (baseline_probe == 0) {
    close(baseline_pipe[0]);
    close(decision_pipe[1]);
    SetCpuFromEnvironment("FX4_DONOR_BASELINE_CPU");

    BaselineMessage baseline {};
    baseline.ok = EncodeRegion(
        bytes, size, position, encoder, donor_plan) ? 1u : 0u;
    baseline.payload_bytes = encoder->OutputSize() - start_size;
    if (!WriteAll(baseline_pipe[1], &baseline, sizeof(baseline))) _exit(3);
    close(baseline_pipe[1]);

    DecisionMessage decision {};
    if (!ReadAll(decision_pipe[0], &decision, sizeof(decision))) _exit(4);
    close(decision_pipe[0]);
    if (!decision.ok) _exit(5);
    if (decision.winner != kBaselineWins) _exit(kTrialLoserExit);
    if (!AppendDecision(ledger_path, region, donor_offset, decision)) {
      _exit(6);
    }
    return true;
  }

  close(baseline_pipe[1]);
  close(decision_pipe[0]);
  BaselineMessage baseline {};
  const bool baseline_read =
      ReadAll(baseline_pipe[0], &baseline, sizeof(baseline));
  close(baseline_pipe[0]);

  DecisionMessage decision {};
  decision.baseline_bytes = baseline.payload_bytes;
  decision.donor_bytes = donor.payload_bytes;
  decision.ok = baseline_read && baseline.ok ? 1u : 0u;
  decision.winner = decision.ok && donor.payload_bytes < baseline.payload_bytes
      ? kDonorWins
      : kBaselineWins;
  const bool sent = WriteAll(decision_pipe[1], &decision, sizeof(decision));
  close(decision_pipe[1]);
  if (!sent || !decision.ok) {
    waitpid(baseline_probe, nullptr, 0);
    return false;
  }

  if (decision.winner == kBaselineWins) {
    _exit(kHandoffExit);
  }

  int baseline_status = 0;
  if (waitpid(baseline_probe, &baseline_status, 0) != baseline_probe ||
      !WIFEXITED(baseline_status) ||
      WEXITSTATUS(baseline_status) != kTrialLoserExit) {
    return false;
  }
  if (!donor_plan->ReplayAt(position, predictor) ||
      !EncodeRegion(bytes, size, position, encoder, donor_plan) ||
      encoder->OutputSize() - start_size != donor.payload_bytes) {
    return false;
  }
  return AppendDecision(ledger_path, region, donor_offset, decision);
}
bool RunWorker(const std::string& input_path,
    const std::string& scratch_output_path, uint64_t input_bytes,
    const std::vector<bool>& vocab, FILE* dictionary,
    bool pretrain_dictionary, DonorPlan* donor_plan,
    const std::string& ledger_path) {
  std::ifstream input(input_path, std::ios::in | std::ios::binary);
  std::ofstream output(
      scratch_output_path, std::ios::out | std::ios::binary);
  if (!input.is_open() || !output.is_open()) return false;

  Predictor predictor(vocab);
  if (pretrain_dictionary) preprocessor::Pretrain(&predictor, dictionary);
  Encoder encoder(&output, &predictor);

  const size_t complete_regions =
      static_cast<size_t>(input_bytes / DonorPlan::kChunkSize);
  std::vector<RecordedDecision> decisions(complete_regions);
  if (!LoadDecisions(ledger_path, &decisions)) return false;

  std::vector<char> buffer(DonorPlan::kChunkSize);
  uint64_t position = 0;
  while (position < input_bytes) {
    const size_t wanted = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), input_bytes - position));
    input.read(buffer.data(), static_cast<std::streamsize>(wanted));
    const size_t count = static_cast<size_t>(input.gcount());
    if (count != wanted) return false;

    const uint32_t region =
        static_cast<uint32_t>(position / DonorPlan::kChunkSize);
    const uint32_t donor_offset = donor_plan->DonorOffset(region);
    if (count == DonorPlan::kChunkSize &&
        donor_offset != DonorPlan::kNoDonor) {
      if (region < decisions.size() && decisions[region].valid) {
        if (!ApplyRecordedDecision(
            decisions[region], donor_offset, position, &predictor,
            donor_plan)) {
          return false;
        }
        if (!EncodeRegion(buffer.data(), count, position, &encoder,
            donor_plan)) {
          return false;
        }
      } else if (!TrialRegion(ledger_path, region, donor_offset,
          buffer.data(), count, position, &encoder, &predictor, donor_plan)) {
        return false;
      }
    } else if (!EncodeRegion(
        buffer.data(), count, position, &encoder, donor_plan)) {
      return false;
    }
    position += count;
  }

  encoder.Flush();
  output.close();
  if (!output.good()) return false;

  const std::string complete_path = CompletePath(ledger_path);
  FILE* complete = fopen(complete_path.c_str(), "wb");
  if (!complete) return false;
  fprintf(complete, "regions=%zu\ninput_bytes=%" PRIu64
      "\nfinal_worker_pid=%ld\n",
      complete_regions, input_bytes, static_cast<long>(getpid()));
  const bool ok = SyncFile(complete);
  fclose(complete);
  return ok;
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
  if (!path || !*path || !donor_plan || donor_plan->empty()) return false;
  const std::string ledger_path(path);
  if (!EnsureLedgers(ledger_path)) return false;
  unlink(CompletePath(ledger_path).c_str());

  if (prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) return false;
  const pid_t worker = fork();
  if (worker < 0) return false;
  if (worker == 0) {
    const bool ok = RunWorker(input_path, scratch_output_path, input_bytes,
        vocab, dictionary, pretrain_dictionary, donor_plan, ledger_path);
    _exit(ok ? 0 : 1);
  }

  bool final_worker_succeeded = false;
  for (;;) {
    int status = 0;
    const pid_t child = waitpid(-1, &status, 0);
    if (child < 0) {
      if (errno == EINTR) continue;
      if (errno == ECHILD) break;
      return false;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
      final_worker_succeeded = true;
    }
  }
  if (!final_worker_succeeded ||
      access(CompletePath(ledger_path).c_str(), R_OK) != 0) {
    return false;
  }
  struct stat info {};
  *output_bytes =
      stat(scratch_output_path.c_str(), &info) == 0 ? info.st_size : 0;
  discovery_completed = true;
  return true;
#endif
}

bool DonorForkDiscoveryCompleted() {
  return discovery_completed;
}
