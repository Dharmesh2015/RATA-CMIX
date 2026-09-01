#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "../src/models/delta-memory.h"
#include "../src/models/token-ngram-bias.h"

namespace {

#pragma pack(push, 1)
struct RecordV1 {
  std::uint16_t base_p;
  std::uint16_t final_p;
  std::uint8_t ppmd_logit;
  std::uint8_t lstm_logit;
  std::uint8_t fxcm_logit;
  std::uint8_t flags;
  std::uint8_t ppm_meta;
  std::uint8_t match_length;
};

struct RecordV2 : RecordV1 {
  std::uint8_t transformer_logit;
};
#pragma pack(pop)

static_assert(sizeof(RecordV1) == 10, "FXOT v1 layout changed");
static_assert(sizeof(RecordV2) == 11, "FXOT v2 layout changed");

std::uint16_t ReadU16(std::istream* input) {
  std::uint16_t value = 0;
  for (unsigned int shift = 0; shift < 16; shift += 8) {
    value |= static_cast<std::uint16_t>(input->get() & 0xff) << shift;
  }
  return value;
}

std::uint64_t ReadU64(std::istream* input) {
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(input->get() & 0xff) << shift;
  }
  return value;
}

double ClampProbability(double probability) {
  return std::max(1.0 / 65536.0,
      std::min(65535.0 / 65536.0, probability));
}

float Logit(float probability) {
  probability = static_cast<float>(ClampProbability(probability));
  return std::log(probability / (1.0f - probability));
}

double Logistic(double logit) {
  if (logit >= 20.0) return 1.0 - 1.0e-9;
  if (logit <= -20.0) return 1.0e-9;
  return 1.0 / (1.0 + std::exp(-logit));
}

double DequantizedProbability(std::uint8_t value) {
  return Logistic(static_cast<double>(value) * (16.0 / 255.0) - 8.0);
}

double BitLoss(double probability, int bit) {
  probability = ClampProbability(probability);
  return -std::log2(bit ? probability : 1.0 - probability);
}

struct Losses {
  long double full = 0.0;
  long double holdout = 0.0;

  void Add(double loss, bool in_holdout) {
    full += loss;
    if (in_holdout) holdout += loss;
  }
};

struct Candidate {
  const char* name;
  Losses loss;
};

struct BlockOracle {
  explicit BlockOracle(std::uint64_t size) : block_size(size) {}

  void Add(const std::array<double, 4>& model_losses, bool in_holdout) {
    for (unsigned int i = 0; i < model_losses.size(); ++i) {
      current[i] += model_losses[i];
    }
    ++used;
    if (used == block_size) Flush(in_holdout);
  }

  void Finish(bool in_holdout) {
    if (used != 0) Flush(in_holdout);
  }

  std::uint64_t block_size;
  std::uint64_t used = 0;
  std::array<long double, 4> current{};
  Losses selected;
  std::uint64_t blocks = 0;

 private:
  void Flush(bool in_holdout) {
    const long double selector_bits = 2.0L;
    const long double best =
        *std::min_element(current.begin(), current.end()) + selector_bits;
    selected.Add(static_cast<double>(best), in_holdout);
    current.fill(0.0);
    used = 0;
    ++blocks;
  }
};

std::string FormatCandidate(const Candidate& candidate,
    const Losses& baseline, std::uint64_t records,
    std::uint64_t holdout_records, std::uint64_t entropy_bytes) {
  const long double full_saving_bits = baseline.full - candidate.loss.full;
  const long double holdout_saving_bits =
      baseline.holdout - candidate.loss.holdout;
  const long double projected_bytes = holdout_records == 0 ? 0.0L :
      holdout_saving_bits / holdout_records * entropy_bytes;
  std::ostringstream output;
  output << std::left << std::setw(24) << candidate.name << std::right
         << " full_bytes=" << std::setw(12) << std::fixed
         << std::setprecision(2)
         << static_cast<double>(candidate.loss.full / 8.0L)
         << " full_save=" << std::setw(10)
         << static_cast<double>(full_saving_bits / 8.0L)
         << " holdout_save=" << std::setw(10)
         << static_cast<double>(holdout_saving_bits / 8.0L)
         << " projected_587mb=" << std::setw(12)
         << static_cast<double>(projected_bytes);
  return output.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: fxot_analyze TRACE.fxot [REPORT.txt]\n";
    return 2;
  }

  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    std::cerr << "cannot open " << argv[1] << '\n';
    return 1;
  }
  char magic[4] = {};
  input.read(magic, sizeof(magic));
  const std::uint16_t version = ReadU16(&input);
  const std::uint16_t record_size = ReadU16(&input);
  const std::uint64_t stream_bytes = ReadU64(&input);
  const std::uint64_t limit_bytes = ReadU64(&input);
  const std::uint64_t record_count = ReadU64(&input);
  if (!input || std::string(magic, 4) != "FXOT" ||
      !((version == 1 && record_size == sizeof(RecordV1)) ||
        (version == 2 && record_size == sizeof(RecordV2))) ||
      record_count == 0) {
    std::cerr << "unsupported or incomplete FXOT trace\n";
    return 1;
  }

  constexpr std::uint64_t kEntropyBytes = 587138826ull;
  const std::uint64_t holdout_start = record_count * 7 / 10;
  const std::uint64_t holdout_records = record_count - holdout_start;
  Candidate baseline{"coded baseline"};
  Candidate predictor_base{"pre-BitLSTM base"};
  Candidate ppmd{"PPMd"};
  Candidate lstm{"online LSTM"};
  Candidate fxcm{"FXCM aggregate"};
  Candidate delta{"terminal DeltaMemory8"};
  Candidate token{"terminal TokenNgram"};
  Candidate token_delta{"TokenNgram + Delta"};
  Candidate per_bit_oracle{"primary per-bit oracle"};
  DeltaMemory8 delta_model;
  TokenNgramBias token_model;
  TokenNgramBias combo_token_model;
  DeltaMemory8 combo_delta_model;
  BlockOracle block256(256);
  BlockOracle block1024(1024);
  BlockOracle block4096(4096);
  std::uint64_t transformer_active = 0;

  for (std::uint64_t index = 0; index < record_count; ++index) {
    RecordV2 record{};
    input.read(reinterpret_cast<char*>(&record), record_size);
    if (!input) {
      std::cerr << "trace ended at record " << index << '\n';
      return 1;
    }
    const bool in_holdout = index >= holdout_start;
    const int bit = record.flags & 1;
    const double base_probability =
        ClampProbability(static_cast<double>(record.final_p) / 65536.0);
    const double predictor_probability =
        ClampProbability(static_cast<double>(record.base_p) / 65536.0);
    const double ppmd_probability = DequantizedProbability(record.ppmd_logit);
    const double lstm_probability = DequantizedProbability(record.lstm_logit);
    const double fxcm_probability = DequantizedProbability(record.fxcm_logit);
    const float base_logit = Logit(static_cast<float>(base_probability));
    const float ppmd_logit = Logit(static_cast<float>(ppmd_probability));
    const float lstm_logit = Logit(static_cast<float>(lstm_probability));
    const float fxcm_logit = Logit(static_cast<float>(fxcm_probability));

    const float delta_probability = delta_model.Predict(
        static_cast<float>(base_probability), base_logit, ppmd_logit,
        lstm_logit, fxcm_logit);
    const float token_probability = token_model.Predict(
        static_cast<float>(base_probability), base_logit);
    const float combo_token_probability = combo_token_model.Predict(
        static_cast<float>(base_probability), base_logit);
    const float combo_probability = combo_delta_model.Predict(
        combo_token_probability, Logit(combo_token_probability), ppmd_logit,
        lstm_logit, fxcm_logit);

    const double base_loss = BitLoss(base_probability, bit);
    const std::array<double, 4> primary_losses = {
        base_loss, BitLoss(ppmd_probability, bit),
        BitLoss(lstm_probability, bit), BitLoss(fxcm_probability, bit)};
    baseline.loss.Add(base_loss, in_holdout);
    predictor_base.loss.Add(BitLoss(predictor_probability, bit), in_holdout);
    ppmd.loss.Add(primary_losses[1], in_holdout);
    lstm.loss.Add(primary_losses[2], in_holdout);
    fxcm.loss.Add(primary_losses[3], in_holdout);
    delta.loss.Add(BitLoss(delta_probability, bit), in_holdout);
    token.loss.Add(BitLoss(token_probability, bit), in_holdout);
    token_delta.loss.Add(BitLoss(combo_probability, bit), in_holdout);
    per_bit_oracle.loss.Add(
        *std::min_element(primary_losses.begin(), primary_losses.end()),
        in_holdout);
    block256.Add(primary_losses, in_holdout);
    block1024.Add(primary_losses, in_holdout);
    block4096.Add(primary_losses, in_holdout);
    if (version == 2 && record.transformer_logit != 0xffu) {
      ++transformer_active;
    }

    delta_model.Perceive(bit);
    token_model.Perceive(bit);
    combo_token_model.Perceive(bit);
    combo_delta_model.Perceive(bit);
  }
  block256.Finish(true);
  block1024.Finish(true);
  block4096.Finish(true);

  std::ostringstream report;
  report << "trace=" << argv[1] << '\n'
         << "version=" << version << " record_size=" << record_size
         << " records=" << record_count << " traced_bytes=" << limit_bytes
         << " stream_bytes=" << stream_bytes << '\n'
         << "holdout_records=" << holdout_records
         << " transformer_active_records=" << transformer_active << '\n'
         << "projection_warning=holdout cross-entropy projection is a screen, "
            "not an archive-size result\n\n";
  const std::array<Candidate*, 9> candidates = {&baseline, &predictor_base,
      &ppmd, &lstm, &fxcm, &delta, &token, &token_delta, &per_bit_oracle};
  for (const Candidate* candidate : candidates) {
    report << FormatCandidate(*candidate, baseline.loss, record_count,
        holdout_records, kEntropyBytes) << '\n';
  }
  for (const BlockOracle* block : {&block256, &block1024, &block4096}) {
    Candidate candidate{""};
    std::string name = "oracle block " + std::to_string(block->block_size);
    candidate.name = name.c_str();
    candidate.loss = block->selected;
    report << FormatCandidate(candidate, baseline.loss, record_count,
        holdout_records, kEntropyBytes) << " blocks=" << block->blocks << '\n';
  }
  constexpr std::uint64_t kCurrentTotal = 108492825ull;
  constexpr std::uint64_t kTargetTotal = 93000000ull;
  report << "\n93MB_required_total_saving="
         << (kCurrentTotal - kTargetTotal) << " bytes\n";

  std::cout << report.str();
  if (argc == 3) {
    std::ofstream output(argv[2], std::ios::trunc);
    if (!output) {
      std::cerr << "cannot write " << argv[2] << '\n';
      return 1;
    }
    output << report.str();
  }
  return 0;
}
