// Offline donor/recipient projection for the prepared FX4 byte stream.
// This tool is not linked into the Hutter executable.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Config {
  fs::path input;
  fs::path output;
  fs::path known_recipient;
  uint64_t chunk_size = 1u << 20;
  uint64_t known_donor_offset = 407793664ull;
  double known_gain = 322.0;
  unsigned feature_bits = 12;
  unsigned ngram_stride = 4;
  unsigned top_per_recipient = 32;
  unsigned max_profiles = 8;
  unsigned seed_size = 4096;
  unsigned seed_stride = 256;
  double profile_cost = 8.0;
  double assignment_cost = 4.0;
  unsigned materialize = 8;
  unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

struct Assignment {
  unsigned donor = 0;
  unsigned recipient = 0;
  double similarity = 0.0;
  double gain = 0.0;
  uint64_t seed_offset = 0;
  double seed_score = 0.0;
  uint64_t seed_hash = 0;
};

uint64_t ParseUnsigned(const char* value, const char* name) {
  try {
    size_t used = 0;
    const uint64_t result = std::stoull(value, &used, 0);
    if (value[used] != '\0') throw std::invalid_argument("trailing data");
    return result;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
}

double ParseDouble(const char* value, const char* name) {
  try {
    size_t used = 0;
    const double result = std::stod(value, &used);
    if (value[used] != '\0') throw std::invalid_argument("trailing data");
    return result;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  }
}

void Usage(const char* program) {
  std::cerr
      << "usage: " << program << " INPUT OUTPUT_DIR [options]\n"
      << "  --chunk-size N\n"
      << "  --feature-bits N\n"
      << "  --ngram-stride N\n"
      << "  --known-recipient FILE\n"
      << "  --known-donor-offset N\n"
      << "  --known-gain N\n"
      << "  --top-per-recipient N\n"
      << "  --max-profiles N\n"
      << "  --seed-size N\n"
      << "  --seed-stride N\n"
      << "  --profile-cost N\n"
      << "  --assignment-cost N\n"
      << "  --materialize N\n"
      << "  --threads N\n";
}

Config ParseArguments(int argc, char** argv) {
  if (argc < 3) {
    Usage(argv[0]);
    throw std::runtime_error("missing INPUT or OUTPUT_DIR");
  }
  Config config;
  config.input = argv[1];
  config.output = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
    const char* value = argv[++i];
    if (option == "--chunk-size") {
      config.chunk_size = ParseUnsigned(value, "chunk size");
    } else if (option == "--feature-bits") {
      config.feature_bits = ParseUnsigned(value, "feature bits");
    } else if (option == "--ngram-stride") {
      config.ngram_stride = ParseUnsigned(value, "ngram stride");
    } else if (option == "--known-recipient") {
      config.known_recipient = value;
    } else if (option == "--known-donor-offset") {
      config.known_donor_offset = ParseUnsigned(value, "known donor offset");
    } else if (option == "--known-gain") {
      config.known_gain = ParseDouble(value, "known gain");
    } else if (option == "--top-per-recipient") {
      config.top_per_recipient = ParseUnsigned(value, "top count");
    } else if (option == "--max-profiles") {
      config.max_profiles = ParseUnsigned(value, "profile count");
    } else if (option == "--seed-size") {
      config.seed_size = ParseUnsigned(value, "seed size");
    } else if (option == "--seed-stride") {
      config.seed_stride = ParseUnsigned(value, "seed stride");
    } else if (option == "--profile-cost") {
      config.profile_cost = ParseDouble(value, "profile cost");
    } else if (option == "--assignment-cost") {
      config.assignment_cost = ParseDouble(value, "assignment cost");
    } else if (option == "--materialize") {
      config.materialize = ParseUnsigned(value, "materialize count");
    } else if (option == "--threads") {
      config.threads = ParseUnsigned(value, "thread count");
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (config.chunk_size < 4096 || config.feature_bits < 8 ||
      config.feature_bits > 18 || config.ngram_stride == 0 ||
      config.seed_size < 16 || config.seed_size > config.chunk_size ||
      config.seed_stride == 0 || config.seed_stride > config.seed_size ||
      config.seed_size % 4 != 0 ||
      config.seed_stride % 4 != 0 || config.profile_cost < 0.0 ||
      config.assignment_cost < 0.0 || config.threads == 0) {
    throw std::runtime_error("invalid numeric configuration");
  }
  return config;
}

uint32_t Mix32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

uint64_t Fnv1a(const uint8_t* data, size_t size) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

unsigned StructuralClass(uint8_t byte) {
  if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z')) return 0;
  if (byte >= '0' && byte <= '9') return 1;
  if (byte == ' ' || byte == '\t') return 2;
  if (byte == '\n' || byte == '\r') return 3;
  switch (byte) {
    case '<': return 4;
    case '>': return 5;
    case '{': return 6;
    case '}': return 7;
    case '[': return 8;
    case ']': return 9;
    case '|': return 10;
    case '=': return 11;
    case '/': return 12;
    case ':': return 13;
    default: return byte >= 128 ? 14 : 15;
  }
}

bool IsWordByte(uint8_t byte) {
  return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
      (byte >= '0' && byte <= '9') || byte == '_';
}

void AddChunkFeatures(const uint8_t* data, size_t data_size,
    unsigned dimensions, unsigned stride, uint32_t* counts,
    std::array<float, 16>* structure) {
  const uint32_t mask = dimensions - 1;
  for (size_t i = 0; i < data_size; ++i) {
    ++(*structure)[StructuralClass(data[i])];
  }
  for (size_t i = 0; i + 3 < data_size; i += stride) {
    const uint32_t value = static_cast<uint32_t>(data[i]) |
        (static_cast<uint32_t>(data[i + 1]) << 8) |
        (static_cast<uint32_t>(data[i + 2]) << 16) |
        (static_cast<uint32_t>(data[i + 3]) << 24);
    ++counts[Mix32(value) & mask];
  }

  uint32_t word_hash = 2166136261u;
  unsigned word_length = 0;
  for (size_t i = 0; i < data_size; ++i) {
    const uint8_t byte = data[i];
    if (IsWordByte(byte)) {
      uint8_t folded = byte;
      if (folded >= 'A' && folded <= 'Z') folded += 'a' - 'A';
      word_hash = (word_hash ^ folded) * 16777619u;
      ++word_length;
    } else if (word_length) {
      const unsigned weight = std::min(word_length, 12u);
      counts[Mix32(word_hash ^ 0x9e3779b9u) & mask] += weight;
      word_hash = 2166136261u;
      word_length = 0;
    }
  }
  if (word_length) {
    counts[Mix32(word_hash ^ 0x9e3779b9u) & mask] +=
        std::min(word_length, 12u);
  }
}
double Dot(const float* left, const float* right, unsigned size) {
  double result = 0.0;
  for (unsigned i = 0; i < size; ++i) result += left[i] * right[i];
  return result;
}

double ProjectedGain(double similarity, double calibration_similarity,
    double known_gain) {
  if (calibration_similarity <= 0.0 ||
      similarity < calibration_similarity * 0.98) {
    return 0.0;
  }
  const double ratio = similarity / calibration_similarity;
  return known_gain * std::min(4.0, ratio * ratio);
}

std::vector<uint8_t> ReadRange(const fs::path& path, uint64_t offset,
    size_t size) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  input.seekg(static_cast<std::streamoff>(offset));
  std::vector<uint8_t> data(size);
  input.read(reinterpret_cast<char*>(data.data()), data.size());
  if (input.gcount() != static_cast<std::streamsize>(data.size())) {
    throw std::runtime_error("short read from " + path.string());
  }
  return data;
}

std::pair<uint64_t, double> BestSeed(const Config& config,
    const uint8_t* corpus, unsigned donor,
    const float* recipient_features, unsigned dimensions) {
  const uint64_t donor_offset = static_cast<uint64_t>(donor) * config.chunk_size;
  const uint8_t* chunk = corpus + donor_offset;
  const uint32_t mask = dimensions - 1;
  const size_t group_count = config.chunk_size / 4;
  const size_t window_groups = config.seed_size / 4;
  const size_t step_groups = config.seed_stride / 4;
  std::vector<float> contribution(group_count);
  for (size_t group = 0; group < group_count; ++group) {
    const size_t index = group * 4;
    const uint32_t value = static_cast<uint32_t>(chunk[index]) |
        (static_cast<uint32_t>(chunk[index + 1]) << 8) |
        (static_cast<uint32_t>(chunk[index + 2]) << 16) |
        (static_cast<uint32_t>(chunk[index + 3]) << 24);
    contribution[group] = recipient_features[Mix32(value) & mask];
  }

  double score = 0.0;
  for (size_t group = 0; group < window_groups; ++group) {
    score += contribution[group];
  }
  double best_score = score;
  size_t best_group = 0;
  for (size_t start_group = step_groups;
       start_group + window_groups <= group_count;
       start_group += step_groups) {
    const size_t previous = start_group - step_groups;
    for (size_t i = 0; i < step_groups; ++i) {
      score -= contribution[previous + i];
      score += contribution[previous + window_groups + i];
    }
    if (score > best_score) {
      best_score = score;
      best_group = start_group;
    }
  }
  return {donor_offset + best_group * 4, best_score};
}
void WriteBinary(const fs::path& path, const std::vector<uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create " + path.string());
  output.write(reinterpret_cast<const char*>(data.data()), data.size());
  if (!output) throw std::runtime_error("cannot write " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = ParseArguments(argc, argv);
    const uint64_t input_size = fs::file_size(config.input);
    const unsigned chunks = input_size / config.chunk_size;
    if (chunks < 2) throw std::runtime_error("input has fewer than two chunks");
    const unsigned dimensions = 1u << config.feature_bits;
    fs::create_directories(config.output);

    std::cout << "input=" << config.input << " bytes=" << input_size
              << " complete_chunks=" << chunks
              << " feature_dimensions=" << dimensions << '\n';

    std::vector<uint32_t> counts(static_cast<size_t>(chunks) * dimensions);
    std::vector<std::array<float, 16>> structure(chunks);
    std::vector<uint64_t> chunk_hashes(chunks);
    std::ifstream input(config.input, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    std::vector<uint8_t> corpus(input_size);
    input.read(reinterpret_cast<char*>(corpus.data()), corpus.size());
    if (input.gcount() != static_cast<std::streamsize>(corpus.size())) {
      throw std::runtime_error("short corpus read");
    }
    const uint64_t stream_hash = Fnv1a(corpus.data(), corpus.size());
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
      const uint8_t* bytes = corpus.data() +
          static_cast<size_t>(chunk) * config.chunk_size;
      chunk_hashes[chunk] = Fnv1a(bytes, config.chunk_size);
      AddChunkFeatures(bytes, config.chunk_size, dimensions,
          config.ngram_stride,
          counts.data() + static_cast<size_t>(chunk) * dimensions,
          &structure[chunk]);
      if ((chunk + 1) % 32 == 0 || chunk + 1 == chunks) {
        std::cout << "features=" << (chunk + 1) << '/' << chunks << '\n';
      }
    }

    std::vector<uint8_t> known_data;
    if (!config.known_recipient.empty()) {
      const uint64_t known_size = fs::file_size(config.known_recipient);
      if (known_size == 0 || known_size > config.chunk_size) {
        throw std::runtime_error(
            "known recipient must contain 1 byte through one complete region");
      }
      known_data = ReadRange(
          config.known_recipient, 0, static_cast<size_t>(known_size));
      std::cout << "known_recipient_bytes=" << known_data.size() << '\n';
    }
    std::vector<unsigned> document_frequency(dimensions);
    for (unsigned feature = 0; feature < dimensions; ++feature) {
      for (unsigned chunk = 0; chunk < chunks; ++chunk) {
        if (counts[static_cast<size_t>(chunk) * dimensions + feature]) {
          ++document_frequency[feature];
        }
      }
    }

    std::vector<float> features(static_cast<size_t>(chunks) * dimensions);
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
      double norm = 0.0;
      for (unsigned feature = 0; feature < dimensions; ++feature) {
        const size_t index = static_cast<size_t>(chunk) * dimensions + feature;
        const double idf = std::log(
            (static_cast<double>(chunks) + 1.0) /
            (static_cast<double>(document_frequency[feature]) + 1.0)) + 1.0;
        const float value = static_cast<float>(std::log1p(counts[index]) * idf);
        features[index] = value;
        norm += static_cast<double>(value) * value;
      }
      const float inverse = norm > 0.0 ? 1.0f / std::sqrt(norm) : 0.0f;
      for (unsigned feature = 0; feature < dimensions; ++feature) {
        features[static_cast<size_t>(chunk) * dimensions + feature] *= inverse;
      }

      double structure_norm = 0.0;
      for (float value : structure[chunk]) structure_norm += value * value;
      const float structure_inverse = structure_norm > 0.0
          ? 1.0f / std::sqrt(structure_norm) : 0.0f;
      for (float& value : structure[chunk]) value *= structure_inverse;
    }
    std::vector<float> known_features;
    std::array<float, 16> known_structure{};
    if (!known_data.empty()) {
      std::vector<uint32_t> known_counts(dimensions);
      AddChunkFeatures(known_data.data(), known_data.size(), dimensions,
          config.ngram_stride, known_counts.data(), &known_structure);
      known_features.resize(dimensions);
      double norm = 0.0;
      for (unsigned feature = 0; feature < dimensions; ++feature) {
        const double idf = std::log(
            (static_cast<double>(chunks) + 1.0) /
            (static_cast<double>(document_frequency[feature]) + 1.0)) + 1.0;
        const float value = static_cast<float>(
            std::log1p(known_counts[feature]) * idf);
        known_features[feature] = value;
        norm += static_cast<double>(value) * value;
      }
      const float inverse = norm > 0.0 ? 1.0f / std::sqrt(norm) : 0.0f;
      for (float& value : known_features) value *= inverse;
      double structure_norm = 0.0;
      for (float value : known_structure) structure_norm += value * value;
      const float structure_inverse = structure_norm > 0.0
          ? 1.0f / std::sqrt(structure_norm) : 0.0f;
      for (float& value : known_structure) value *= structure_inverse;
    }
    counts.clear();
    counts.shrink_to_fit();

    std::vector<float> similarities(static_cast<size_t>(chunks) * chunks, 1.0f);
    std::atomic<unsigned> next_row{0};
    auto similarity_worker = [&]() {
      for (;;) {
        const unsigned left = next_row.fetch_add(1);
        if (left >= chunks) return;
        const float* left_features =
            features.data() + static_cast<size_t>(left) * dimensions;
        for (unsigned right = left + 1; right < chunks; ++right) {
          const float* right_features =
              features.data() + static_cast<size_t>(right) * dimensions;
          const double lexical = Dot(left_features, right_features, dimensions);
          double structural = 0.0;
          for (unsigned i = 0; i < structure[left].size(); ++i) {
            structural += structure[left][i] * structure[right][i];
          }
          const float score = static_cast<float>(
              std::max(0.0, std::min(1.0, lexical * 0.90 + structural * 0.10)));
          similarities[static_cast<size_t>(left) * chunks + right] = score;
          similarities[static_cast<size_t>(right) * chunks + left] = score;
        }
      }
    };
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < std::min(config.threads, chunks); ++i) {
      workers.emplace_back(similarity_worker);
    }
    for (std::thread& worker : workers) worker.join();
    std::cout << "similarity_matrix=complete threads=" << workers.size() << '\n';

    std::vector<float> distinct_scores;
    distinct_scores.reserve(static_cast<size_t>(chunks) * (chunks - 1) / 2);
    for (unsigned left = 0; left < chunks; ++left) {
      for (unsigned right = left + 1; right < chunks; ++right) {
        distinct_scores.push_back(
            similarities[static_cast<size_t>(left) * chunks + right]);
      }
    }
    std::sort(distinct_scores.begin(), distinct_scores.end());
    const size_t percentile_index = static_cast<size_t>(
        0.995 * static_cast<double>(distinct_scores.size() - 1));
    double calibration_similarity = distinct_scores[percentile_index];
    const unsigned known_donor_chunk =
        config.known_donor_offset / config.chunk_size;
    if (!known_features.empty() && known_donor_chunk < chunks &&
        config.known_donor_offset + config.seed_size <= corpus.size()) {
      const float* donor_features = features.data() +
          static_cast<size_t>(known_donor_chunk) * dimensions;
      const double lexical = Dot(
          donor_features, known_features.data(), dimensions);
      double structural = 0.0;
      for (unsigned i = 0; i < known_structure.size(); ++i) {
        structural += structure[known_donor_chunk][i] * known_structure[i];
      }
      calibration_similarity = std::max(
          0.0, std::min(1.0, lexical * 0.90 + structural * 0.10));
      std::cout << "calibration=known_external_pair donor_region="
                << known_donor_chunk << " donor_seed_offset="
                << config.known_donor_offset << '\n';
    } else {
      std::cout << "calibration=p99.5_proxy\n";
    }
    std::cout << std::fixed << std::setprecision(8)
              << "calibration_similarity=" << calibration_similarity
              << " known_gain=" << config.known_gain << '\n';

    // A donor must be among the strongest candidates for that recipient.
    // This prevents broad corpus similarity from masquerading as reusable
    // donor knowledge.
    std::vector<uint8_t> eligible(static_cast<size_t>(chunks) * chunks);
    for (unsigned recipient = 0; recipient < chunks; ++recipient) {
      std::vector<std::pair<float, unsigned>> ranking;
      ranking.reserve(chunks - 1);
      for (unsigned donor = 0; donor < chunks; ++donor) {
        if (donor == recipient) continue;
        ranking.emplace_back(
            similarities[static_cast<size_t>(donor) * chunks + recipient],
            donor);
      }
      const unsigned keep = std::min<unsigned>(config.top_per_recipient,
          ranking.size());
      std::partial_sort(ranking.begin(), ranking.begin() + keep, ranking.end(),
          std::greater<std::pair<float, unsigned>>());
      for (unsigned index = 0; index < keep; ++index) {
        eligible[static_cast<size_t>(ranking[index].second) * chunks +
            recipient] = 1;
      }
    }

    std::vector<Assignment> candidate_edges;
    candidate_edges.reserve(static_cast<size_t>(chunks) *
                            config.top_per_recipient);
    std::ofstream all_pairs(config.output / "all_pairs.csv");
    all_pairs << "donor_chunk,donor_offset,recipient_chunk,recipient_offset,"
                 "similarity,projected_gain_bytes,top_for_recipient,"
                 "donor_precedes\n";
    all_pairs << std::fixed << std::setprecision(8);
    for (unsigned donor = 0; donor < chunks; ++donor) {
      for (unsigned recipient = 0; recipient < chunks; ++recipient) {
        if (donor == recipient) continue;
        const double similarity =
            similarities[static_cast<size_t>(donor) * chunks + recipient];
        const bool is_eligible =
            eligible[static_cast<size_t>(donor) * chunks + recipient] != 0;
        const double projected_gain = is_eligible
            ? ProjectedGain(
                  similarity, calibration_similarity, config.known_gain)
            : 0.0;
        if (is_eligible) {
          candidate_edges.push_back(
              {donor, recipient, similarity, projected_gain, 0, 0.0, 0});
        }
        all_pairs << donor << ',' << donor * config.chunk_size << ','
                  << recipient << ',' << recipient * config.chunk_size << ','
                  << similarity << ',' << projected_gain << ','
                  << (is_eligible ? 1 : 0) << ','
                  << (donor < recipient ? 1 : 0) << '\n';
      }
    }

    std::atomic<size_t> next_candidate{0};
    auto seed_worker = [&]() {
      for (;;) {
        const size_t index = next_candidate.fetch_add(1);
        if (index >= candidate_edges.size()) return;
        Assignment& edge = candidate_edges[index];
        const float* recipient_features = features.data() +
            static_cast<size_t>(edge.recipient) * dimensions;
        std::tie(edge.seed_offset, edge.seed_score) = BestSeed(
            config, corpus.data(), edge.donor, recipient_features, dimensions);
        edge.seed_hash = Fnv1a(
            corpus.data() + edge.seed_offset, config.seed_size);
      }
    };
    workers.clear();
    for (unsigned i = 0;
         i < std::min<unsigned>(config.threads,
                               std::max<size_t>(1, candidate_edges.size()));
         ++i) {
      workers.emplace_back(seed_worker);
    }
    for (std::thread& worker : workers) worker.join();
    std::sort(candidate_edges.begin(), candidate_edges.end(),
        [](const Assignment& left, const Assignment& right) {
          if (left.recipient != right.recipient) {
            return left.recipient < right.recipient;
          }
          if (left.gain != right.gain) return left.gain > right.gain;
          return left.seed_score > right.seed_score;
        });
    std::ofstream candidate_output(config.output / "candidate_edges.csv");
    candidate_output << "donor_region,donor_region_offset,donor_seed_offset,"
                        "donor_seed_hash,seed_size,recipient_region,"
                        "recipient_offset,similarity,seed_score,"
                        "projected_fx4_gain_bytes,donor_precedes,"
                        "requires_reorder,validation\n";
    candidate_output << std::fixed << std::setprecision(8);
    for (const Assignment& edge : candidate_edges) {
      candidate_output << edge.donor << ','
                       << static_cast<uint64_t>(edge.donor) * config.chunk_size
                       << ',' << edge.seed_offset << ',' << std::hex
                       << edge.seed_hash << std::dec << ',' << config.seed_size
                       << ',' << edge.recipient << ','
                       << static_cast<uint64_t>(edge.recipient) *
                              config.chunk_size
                       << ',' << edge.similarity << ',' << edge.seed_score
                       << ',' << edge.gain << ','
                       << (edge.donor < edge.recipient ? 1 : 0) << ','
                       << (edge.donor < edge.recipient ? 0 : 1)
                       << ",proxy_only\n";
    }
    std::cout << "candidate_edges=" << candidate_edges.size()
              << " seed_offsets=resolved\n";
    const double assignment_cost = config.assignment_cost;
    const double profile_cost = config.profile_cost;
    std::vector<double> current_gain(chunks);
    std::vector<int> current_donor(chunks, -1);
    std::vector<unsigned> selected_donors;
    std::ofstream profile_output(config.output / "profiles.csv");
    profile_output << "iteration,donor_chunk,donor_offset,recipients,"
                      "gross_marginal,profile_cost,assignment_cost,net_marginal\n";
    for (unsigned iteration = 0; iteration < config.max_profiles; ++iteration) {
      int best_donor = -1;
      double best_net = 0.0;
      double best_gross = 0.0;
      unsigned best_recipients = 0;
      for (unsigned donor = 0; donor < chunks; ++donor) {
        if (std::find(selected_donors.begin(), selected_donors.end(), donor) !=
            selected_donors.end()) continue;
        double gross = 0.0;
        unsigned recipients = 0;
        for (unsigned recipient = 0; recipient < chunks; ++recipient) {
          if (recipient == donor) continue;
          if (!eligible[static_cast<size_t>(donor) * chunks + recipient]) {
            continue;
          }
          const double similarity =
              similarities[static_cast<size_t>(donor) * chunks + recipient];
          const double gain = ProjectedGain(
              similarity, calibration_similarity, config.known_gain);
          const double marginal = gain - current_gain[recipient] - assignment_cost;
          if (marginal > 0.0) {
            gross += marginal;
            ++recipients;
          }
        }
        const double net = gross - profile_cost;
        if (net > best_net) {
          best_net = net;
          best_gross = gross;
          best_donor = donor;
          best_recipients = recipients;
        }
      }
      if (best_donor < 0) break;
      selected_donors.push_back(best_donor);
      for (unsigned recipient = 0; recipient < chunks; ++recipient) {
        if (recipient == static_cast<unsigned>(best_donor)) continue;
        if (!eligible[static_cast<size_t>(best_donor) * chunks + recipient]) {
          continue;
        }
        const double similarity = similarities[
            static_cast<size_t>(best_donor) * chunks + recipient];
        const double gain = ProjectedGain(
            similarity, calibration_similarity, config.known_gain);
        if (gain > current_gain[recipient] + assignment_cost) {
          current_gain[recipient] = gain;
          current_donor[recipient] = best_donor;
        }
      }
      profile_output << iteration << ',' << best_donor << ','
                     << best_donor * config.chunk_size << ',' << best_recipients
                     << ',' << best_gross << ',' << profile_cost << ','
                     << assignment_cost * best_recipients << ',' << best_net
                     << '\n';
    }

    std::vector<Assignment> assignments;
    for (unsigned recipient = 0; recipient < chunks; ++recipient) {
      if (current_donor[recipient] < 0) continue;
      const unsigned donor = current_donor[recipient];
      assignments.push_back({donor, recipient,
          similarities[static_cast<size_t>(donor) * chunks + recipient],
          current_gain[recipient], 0, 0.0});
    }
    for (Assignment& assignment : assignments) {
      const auto edge = std::find_if(candidate_edges.begin(),
          candidate_edges.end(), [&](const Assignment& candidate) {
            return candidate.donor == assignment.donor &&
                candidate.recipient == assignment.recipient;
          });
      if (edge != candidate_edges.end()) {
        assignment.seed_offset = edge->seed_offset;
        assignment.seed_score = edge->seed_score;
        assignment.seed_hash = edge->seed_hash;
      }
    }
    std::sort(assignments.begin(), assignments.end(),
        [](const Assignment& left, const Assignment& right) {
          return std::tie(left.gain, left.similarity) >
              std::tie(right.gain, right.similarity);
        });

    std::ofstream assignment_output(config.output / "assignments.csv");
    assignment_output << "donor_chunk,donor_offset,seed_offset,seed_hash,"
                         "seed_size,recipient_chunk,recipient_offset,"
                         "similarity,seed_score,projected_gain_bytes,"
                         "requires_reorder,validation\n";
    assignment_output << std::fixed << std::setprecision(8);
    for (const Assignment& assignment : assignments) {
      assignment_output << assignment.donor << ','
                        << assignment.donor * config.chunk_size << ','
                        << assignment.seed_offset << ',' << std::hex
                        << assignment.seed_hash << std::dec << ','
                        << config.seed_size << ',' << assignment.recipient << ','
                        << assignment.recipient * config.chunk_size << ','
                        << assignment.similarity << ','
                        << assignment.seed_score << ',' << assignment.gain << ','
                        << (assignment.donor < assignment.recipient ? 0 : 1)
                        << ",proxy_only\n";
    }

    fs::create_directories(config.output / "materialized");
    std::ofstream shortlist(config.output / "shortlist.csv");
    shortlist << "rank,donor_chunk,donor_offset,seed_offset,seed_score,"
                 "recipient_chunk,recipient_offset,similarity,"
                 "projected_gain_bytes,recipient_file,seed_file\n";
    shortlist << std::fixed << std::setprecision(8);
    const unsigned materialized =
        std::min<unsigned>(config.materialize, assignments.size());
    for (unsigned rank = 0; rank < materialized; ++rank) {
      Assignment& assignment = assignments[rank];
      const float* recipient_features = features.data() +
          static_cast<size_t>(assignment.recipient) * dimensions;
      std::tie(assignment.seed_offset, assignment.seed_score) = BestSeed(
          config, corpus.data(), assignment.donor, recipient_features,
          dimensions);
      assignment.seed_hash = Fnv1a(
          corpus.data() + assignment.seed_offset, config.seed_size);
      const fs::path recipient_path = config.output / "materialized" /
          ("recipient_" + std::to_string(assignment.recipient) + ".bin");
      const fs::path seed_path = config.output / "materialized" /
          ("seed_d" + std::to_string(assignment.donor) + "_r" +
           std::to_string(assignment.recipient) + ".bin");
      WriteBinary(recipient_path, ReadRange(config.input,
          static_cast<uint64_t>(assignment.recipient) * config.chunk_size,
          config.chunk_size));
      WriteBinary(seed_path, ReadRange(
          config.input, assignment.seed_offset, config.seed_size));
      shortlist << rank << ',' << assignment.donor << ','
                << assignment.donor * config.chunk_size << ','
                << assignment.seed_offset << ',' << assignment.seed_score << ','
                << assignment.recipient << ','
                << assignment.recipient * config.chunk_size << ','
                << assignment.similarity << ',' << assignment.gain << ','
                << recipient_path.string() << ',' << seed_path.string() << '\n';
    }

    double projected_gross = 0.0;
    for (double gain : current_gain) projected_gross += gain;
    const double projected_cost = selected_donors.size() * profile_cost +
        assignments.size() * assignment_cost;
    std::ofstream summary(config.output / "summary.txt");
    summary << std::fixed << std::setprecision(8)
            << "input=" << config.input << '\n'
            << "input_bytes=" << input_size << '\n'
            << "input_fnv1a64=" << std::hex << stream_hash << std::dec << '\n'
            << "chunk_size=" << config.chunk_size << '\n'
            << "complete_chunks=" << chunks << '\n'
            << "directed_pairs=" << static_cast<uint64_t>(chunks) * (chunks - 1)
            << '\n'
            << "top_candidates_per_recipient=" << config.top_per_recipient
            << '\n'
            << "candidate_edges=" << candidate_edges.size() << '\n'
            << "seed_size=" << config.seed_size << '\n'
            << "seed_stride=" << config.seed_stride << '\n'
            << "known_donor_offset=" << config.known_donor_offset << '\n'
            << "known_donor_fnv1a64=" << std::hex
            << Fnv1a(corpus.data() + config.known_donor_offset,
                     config.seed_size)
            << std::dec << '\n'
            << "calibration_similarity=" << calibration_similarity << '\n'
            << "calibration_gain_bytes=" << config.known_gain << '\n'
            << "profile_cost_bytes=" << config.profile_cost << '\n'
            << "assignment_cost_bytes=" << config.assignment_cost << '\n'
            << "selected_profiles=" << selected_donors.size() << '\n'
            << "assigned_recipients=" << assignments.size() << '\n'
            << "projected_gross_gain_bytes=" << projected_gross << '\n'
            << "projected_profile_and_assignment_cost=" << projected_cost << '\n'
            << "projected_net_gain_bytes=" << projected_gross - projected_cost
            << '\n'
            << "warning=projection_only_exact_fx4_validation_required\n";

    std::cout << "directed_pairs="
              << static_cast<uint64_t>(chunks) * (chunks - 1)
              << " selected_profiles=" << selected_donors.size()
              << " assigned_recipients=" << assignments.size()
              << " projected_net=" << projected_gross - projected_cost << '\n';
    std::cout << "output=" << config.output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
