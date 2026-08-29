#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/models/scr2_tokens.h"

namespace {

constexpr unsigned int kPatternCount = scr2::kTokenCount;
constexpr unsigned int kContextKinds = 11;
constexpr unsigned int kContextSlots = 512;

struct Counts {
  std::uint64_t triggers = 0;
  std::uint64_t takes = 0;
};

double Probability(const Counts& counts) {
  return (static_cast<double>(counts.takes) + 1.0) /
      (static_cast<double>(counts.triggers) + 2.0);
}

double HierarchicalProbability(
    const Counts& local, double parent_probability, double strength) {
  return (static_cast<double>(local.takes) + 1.0 +
      strength * parent_probability) /
      (static_cast<double>(local.triggers) + 2.0 + strength);
}

double BitCost(bool bit, double probability) {
  const double mass = bit ? probability : 1.0 - probability;
  return -std::log2(std::max(mass, 1.0 / 65536.0));
}

unsigned int ByteClass(std::uint8_t value) {
  if (value == '\n' || value == '\r') return 0;
  if (value == ' ' || value == '\t') return 1;
  if (value >= '0' && value <= '9') return 2;
  if (value >= 'A' && value <= 'Z') return 3;
  if (value >= 'a' && value <= 'z') return 4;
  if (value >= 0x80) return 5;
  if (std::strchr("<>[]{}|=:/#*&;", value) != nullptr) return 6;
  return 7;
}

unsigned int LineBucket(const std::uint8_t* data, std::uint64_t start) {
  std::uint64_t distance = 0;
  while (distance < 64 && distance < start &&
      data[start - distance - 1] != '\n') {
    ++distance;
  }
  if (distance == 0) return 0;
  if (distance == 1) return 1;
  if (distance <= 3) return 2;
  if (distance <= 7) return 3;
  if (distance <= 15) return 4;
  if (distance <= 31) return 5;
  return 6;
}

const char* ContextName(unsigned int kind) {
  static constexpr const char* names[kContextKinds] = {
      "global", "byte_class", "line_bucket", "class_line",
      "prev_byte", "prev2_hash", "stream_band16", "stream_band64",
      "band16_line", "band64_line", "band64_byte_class"};
  return names[kind];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr <<
        "usage: scan_causal_scr2 INPUT_POST_R1 OUTPUT_CSV "
        "[MIN_PREFIX] [PATTERN_LIST]\n";
    return 2;
  }
  const unsigned int minimum_prefix = argc == 4
      ? static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10)) : 6u;
  if (minimum_prefix < 1 || minimum_prefix > 6) {
    std::cerr << "MIN_PREFIX must be in 1..6\n";
    return 2;
  }
  std::array<bool, kPatternCount + 1> enabled_patterns{};
  enabled_patterns.fill(true);
  enabled_patterns[0] = false;
  if (argc == 5) {
    enabled_patterns.fill(false);
    const char* cursor = argv[4];
    while (*cursor) {
      char* end = nullptr;
      const unsigned long pattern = std::strtoul(cursor, &end, 10);
      if (end == cursor || pattern < 1 || pattern > kPatternCount) {
        std::cerr << "invalid PATTERN_LIST\n";
        return 2;
      }
      enabled_patterns[pattern] = true;
      if (*end == '\0') break;
      if (*end != ',') {
        std::cerr << "invalid PATTERN_LIST\n";
        return 2;
      }
      cursor = end + 1;
    }
  }

  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    std::perror("open input");
    return 1;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    std::perror("fstat input");
    close(fd);
    return 1;
  }
  const std::uint64_t size = static_cast<std::uint64_t>(st.st_size);
  void* mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    std::perror("mmap input");
    close(fd);
    return 1;
  }
#ifdef MADV_SEQUENTIAL
  madvise(mapping, size, MADV_SEQUENTIAL);
#endif
  const auto* data = static_cast<const std::uint8_t*>(mapping);

  std::array<std::uint16_t, kPatternCount + 1> prefix_length{};
  std::array<std::int16_t, 256> bucket_head{};
  std::array<std::int16_t, kPatternCount + 1> bucket_next{};
  bucket_head.fill(-1);
  bucket_next.fill(-1);
  for (std::uint16_t code = 1; code <= kPatternCount; ++code) {
    if (!enabled_patterns[code]) continue;
    const std::uint16_t pattern_length = scr2::kTokens[code].length;
    if (pattern_length <= minimum_prefix) continue;
    const std::uint8_t* pattern = scr2::PatternData(code);
    for (std::uint16_t length = minimum_prefix;
         length < pattern_length; ++length) {
      bool unique = true;
      for (std::uint16_t other = 1; other <= kPatternCount; ++other) {
        if (!enabled_patterns[other] || other == code ||
            scr2::kTokens[other].length < length) {
          continue;
        }
        if (std::memcmp(pattern, scr2::PatternData(other), length) == 0) {
          unique = false;
          break;
        }
      }
      if (unique) {
        prefix_length[code] = length;
        break;
      }
    }
    if (prefix_length[code] == 0) continue;
    const std::uint8_t bucket = pattern[prefix_length[code] - 1];
    bucket_next[code] = bucket_head[bucket];
    bucket_head[bucket] = static_cast<std::int16_t>(code);
  }

  using ContextTable = std::array<Counts, kContextSlots>;
  using PatternTable = std::array<ContextTable, kContextKinds>;
  static std::array<PatternTable, kPatternCount + 1> counts{};
  static constexpr std::array<double, 5> kBackoffStrength = {
      2.0, 4.0, 8.0, 16.0, 32.0};
  static constexpr std::array<double, 3> kLocalBlendWeight = {
      0.25, 0.5, 0.75};
  double local_bits = 0.0;
  double line_bits = 0.0;
  double band_bits = 0.0;
  double global_bits = 0.0;
  std::array<double, 5> line_backoff_bits{};
  std::array<double, 5> band_backoff_bits{};
  std::array<double, 5> dual_backoff_bits{};
  std::array<std::array<double, 3>, kContextKinds> blend_bits{};
  std::array<double, 3> backed_context_blend_bits{};

  std::uint64_t position = 0;
  std::uint64_t replayed_bytes = 0;
  while (position < size) {
    std::uint16_t best = 0;
    std::uint16_t best_prefix = 0;
    std::uint16_t best_suffix = 0;
    if (position != 0) {
      const std::uint8_t bucket = data[position - 1];
      for (std::int16_t current = bucket_head[bucket]; current >= 0;
           current = bucket_next[current]) {
        const auto code = static_cast<std::uint16_t>(current);
        const std::uint16_t prefix = prefix_length[code];
        const std::uint16_t suffix = scr2::kTokens[code].length - prefix;
        if (prefix > position || suffix > size - position ||
            suffix <= best_suffix) {
          continue;
        }
        if (std::memcmp(data + position - prefix,
                scr2::PatternData(code), prefix) == 0) {
          best = code;
          best_prefix = prefix;
          best_suffix = suffix;
        }
      }
    }

    if (best == 0) {
      ++position;
      continue;
    }

    const bool take = std::memcmp(data + position,
        scr2::PatternData(best) + best_prefix, best_suffix) == 0;
    const std::uint64_t start = position - best_prefix;
    const std::uint8_t previous = start == 0 ? 0 : data[start - 1];
    const std::uint8_t previous2 = start < 2 ? 0 : data[start - 2];
    const unsigned int byte_class = ByteClass(previous);
    const unsigned int line_bucket = LineBucket(data, start);
    const unsigned int band16 = static_cast<unsigned int>(
        (position * 16u) / size);
    const unsigned int band64 = static_cast<unsigned int>(
        (position * 64u) / size);
    const std::array<unsigned int, kContextKinds> contexts = {
        0,
        byte_class,
        line_bucket,
        byte_class * 8 + line_bucket,
        previous,
        static_cast<unsigned int>((previous2 * 257u + previous) & 255u),
        band16,
        band64,
        band16 * 8u + line_bucket,
        band64 * 8u + line_bucket,
        band64 * 8u + byte_class};
    const Counts& local_counts = counts[best][9][contexts[9]];
    const Counts& line_counts = counts[best][2][contexts[2]];
    const Counts& band_counts = counts[best][7][contexts[7]];
    const Counts& global_counts = counts[best][0][0];
    const double local_probability = Probability(local_counts);
    const double line_probability = Probability(line_counts);
    const double band_probability = Probability(band_counts);
    const double global_probability = Probability(global_counts);
    const Counts& class_local_counts = counts[best][10][contexts[10]];
    const Counts& class_parent_counts = counts[best][1][contexts[1]];
    const double class_parent_probability = Probability(class_parent_counts);
    const double line_backoff_probability = HierarchicalProbability(
        local_counts, line_probability, 8.0);
    const double class_backoff_probability = HierarchicalProbability(
        class_local_counts, class_parent_probability, 8.0);
    local_bits += BitCost(take, local_probability);
    line_bits += BitCost(take, line_probability);
    band_bits += BitCost(take, band_probability);
    global_bits += BitCost(take, global_probability);
    for (std::size_t index = 0; index < kBackoffStrength.size(); ++index) {
      const double strength = kBackoffStrength[index];
      line_backoff_bits[index] += BitCost(take,
          HierarchicalProbability(
              local_counts, line_probability, strength));
      band_backoff_bits[index] += BitCost(take,
          HierarchicalProbability(
              local_counts, band_probability, strength));
      dual_backoff_bits[index] += BitCost(take,
          HierarchicalProbability(local_counts,
              (line_probability + band_probability) * 0.5, strength));
    }
    for (unsigned int kind = 0; kind < kContextKinds; ++kind) {
      if (kind == 9) continue;
      const double alternate_probability =
          Probability(counts[best][kind][contexts[kind]]);
      for (std::size_t index = 0; index < kLocalBlendWeight.size(); ++index) {
        const double local_weight = kLocalBlendWeight[index];
        blend_bits[kind][index] += BitCost(take,
            local_weight * local_probability +
            (1.0 - local_weight) * alternate_probability);
      }
    }
    for (std::size_t index = 0; index < kLocalBlendWeight.size(); ++index) {
      const double line_weight = kLocalBlendWeight[index];
      backed_context_blend_bits[index] += BitCost(take,
          line_weight * line_backoff_probability +
          (1.0 - line_weight) * class_backoff_probability);
    }
    for (unsigned int kind = 0; kind < kContextKinds; ++kind) {
      Counts& item = counts[best][kind][contexts[kind]];
      ++item.triggers;
      if (take) ++item.takes;
    }
    if (take) replayed_bytes += best_suffix;
    position += take ? best_suffix : 1;
  }

  std::ofstream output(argv[2]);
  if (!output.is_open()) {
    std::cerr << "cannot open output: " << argv[2] << '\n';
    munmap(mapping, size);
    close(fd);
    return 1;
  }
  output << "pattern,prefix_length,suffix_length,context_kind,context,"
            "triggers,takes\n";
  for (unsigned int pattern = 1; pattern <= kPatternCount; ++pattern) {
    if (prefix_length[pattern] == 0) continue;
    const unsigned int suffix =
        scr2::kTokens[pattern].length - prefix_length[pattern];
    for (unsigned int kind = 0; kind < kContextKinds; ++kind) {
      for (unsigned int context = 0; context < kContextSlots; ++context) {
        const Counts& item = counts[pattern][kind][context];
        if (item.triggers == 0) continue;
        output << pattern << ',' << prefix_length[pattern] << ',' << suffix
               << ',' << ContextName(kind) << ',' << context << ','
               << item.triggers << ',' << item.takes << '\n';
      }
    }
  }

  std::uint64_t total_triggers = 0;
  std::uint64_t total_takes = 0;
  for (unsigned int pattern = 1; pattern <= kPatternCount; ++pattern) {
    total_triggers += counts[pattern][0][0].triggers;
    total_takes += counts[pattern][0][0].takes;
  }
  std::cerr << "scanned=" << size << " triggers=" << total_triggers
            << " takes=" << total_takes
            << " replayed=" << replayed_bytes
            << " min_prefix=" << minimum_prefix << '\n';
  std::cerr << "online_model=local bits=" << local_bits
            << " delta_bytes=0\n";
  std::cerr << "online_model=line bits=" << line_bits
            << " delta_bytes=" << (local_bits - line_bits) / 8.0 << '\n';
  std::cerr << "online_model=band bits=" << band_bits
            << " delta_bytes=" << (local_bits - band_bits) / 8.0 << '\n';
  std::cerr << "online_model=global bits=" << global_bits
            << " delta_bytes=" << (local_bits - global_bits) / 8.0 << '\n';
  for (std::size_t index = 0; index < kBackoffStrength.size(); ++index) {
    std::cerr << "online_model=line_backoff_" << kBackoffStrength[index]
              << " bits=" << line_backoff_bits[index]
              << " delta_bytes="
              << (local_bits - line_backoff_bits[index]) / 8.0 << '\n';
    std::cerr << "online_model=band_backoff_" << kBackoffStrength[index]
              << " bits=" << band_backoff_bits[index]
              << " delta_bytes="
              << (local_bits - band_backoff_bits[index]) / 8.0 << '\n';
    std::cerr << "online_model=dual_backoff_" << kBackoffStrength[index]
              << " bits=" << dual_backoff_bits[index]
              << " delta_bytes="
              << (local_bits - dual_backoff_bits[index]) / 8.0 << '\n';
  }
  for (unsigned int kind = 0; kind < kContextKinds; ++kind) {
    if (kind == 9) continue;
    for (std::size_t index = 0; index < kLocalBlendWeight.size(); ++index) {
      std::cerr << "online_model=blend_local_"
                << kLocalBlendWeight[index] << '_' << ContextName(kind)
                << " bits=" << blend_bits[kind][index]
                << " delta_bytes="
                << (local_bits - blend_bits[kind][index]) / 8.0 << '\n';
    }
  }
  for (std::size_t index = 0; index < kLocalBlendWeight.size(); ++index) {
    std::cerr << "online_model=backed_line_class_"
              << kLocalBlendWeight[index]
              << " bits=" << backed_context_blend_bits[index]
              << " delta_bytes="
              << (local_bits - backed_context_blend_bits[index]) / 8.0
              << '\n';
  }
  munmap(mapping, size);
  close(fd);
  return output.good() ? 0 : 1;
}
