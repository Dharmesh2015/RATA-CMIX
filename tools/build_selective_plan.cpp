#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/models/scr2_tokens.h"

namespace {

constexpr std::uint32_t kChunkSize = 1u << 20;
constexpr std::uint16_t kCostPositiveScr2 = 0xffffu;
constexpr std::uint64_t kIoBlockBytes = 1u << 20;

struct Donor {
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  bool operator<(const Donor& other) const {
    return std::tie(offset, length) < std::tie(other.offset, other.length);
  }
  bool operator==(const Donor& other) const {
    return offset == other.offset && length == other.length;
  }
};
using DonorSequence = std::vector<Donor>;

struct Winner {
  std::uint32_t region = 0;
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  std::string mode;
  std::uint32_t mask = 0;
  std::uint16_t mini_mask = 0;
  std::uint8_t strength = 0;
  DonorSequence donors;
  std::uint16_t vr_min_length = 0;
  std::uint32_t vr_event_count = 0;
  std::uint64_t baseline = 0;
  std::uint64_t candidate = 0;
  std::uint64_t standalone_side = 0;
  std::int64_t standalone_net = 0;
  std::uint8_t stream_class = 10;
};

struct Assignment {
  std::uint16_t profile = 0;
  std::uint32_t offset = 0;
  std::uint32_t length = 0;
  std::uint8_t order = 0;
};
struct Span {
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t mask = 0;
  std::uint8_t stream_class = 0;
  std::uint8_t profile = 0;
  std::uint16_t mini_mask = 0;
};
struct ReplayEvent {
  std::uint64_t offset = 0;
  std::uint16_t pattern = 0;
};
struct Header {
  std::uint32_t chunk_size = 0;
  std::uint64_t stream_size = 0;
  std::array<std::uint8_t, 32> digest{};
};
struct Options {
  std::string candidate_plan;
  std::string output_f4cp;
  std::string output_f4vr;
  std::string output_selected;
  std::string summary;
  std::string post_r1;
  std::vector<std::string> selected_csvs;
  std::vector<std::string> trial_csvs;
  std::vector<std::string> event_csvs;
  std::uint64_t baseline_s1 = 0;
  std::uint64_t candidate_s1 = 0;
  std::int64_t minimum_net = 1;
  bool allow_f4cp = true;
  bool allow_f4vr = true;
};
using Fields = std::unordered_map<std::string, std::string>;

void Usage() {
  std::fprintf(stderr,
      "usage: build_selective_plan --candidate-plan FILE "
      "--output-f4cp FILE --output-f4vr FILE --output-selected FILE "
      "--summary FILE [--post-r1 FILE] [--selected FILE]... "
      "[--trials FILE]... [--events FILE]... "
      "[--baseline-s1 N --candidate-s1 N --minimum-net N] "
      "[--allow-f4cp 0|1 --allow-f4vr 0|1]\n");
}

bool ParseU64(const std::string& text, std::uint64_t* value) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}
bool ParseI64(const std::string& text, std::int64_t* value) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long long parsed = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *value = static_cast<std::int64_t>(parsed);
  return true;
}
bool U64(const Fields& row, const char* name, std::uint64_t* value) {
  const auto found = row.find(name);
  return found != row.end() && ParseU64(found->second, value);
}
bool I64(const Fields& row, const char* name, std::int64_t* value) {
  const auto found = row.find(name);
  return found != row.end() && ParseI64(found->second, value);
}
bool U32(const Fields& row, const char* name, std::uint32_t* value) {
  std::uint64_t parsed = 0;
  if (!U64(row, name, &parsed) || parsed > UINT32_MAX) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}
bool U16(const Fields& row, const char* name, std::uint16_t* value) {
  std::uint32_t parsed = 0;
  if (!U32(row, name, &parsed) || parsed > UINT16_MAX) return false;
  *value = static_cast<std::uint16_t>(parsed);
  return true;
}

std::vector<std::string> Split(const std::string& line, char separator) {
  std::vector<std::string> result;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = line.find(separator, begin);
    result.push_back(line.substr(
        begin, end == std::string::npos ? end : end - begin));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

template <typename Function>
bool ReadCsv(const std::string& path, Function function) {
  std::ifstream input(path);
  if (!input.is_open()) return false;
  std::string line;
  if (!std::getline(input, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  const std::vector<std::string> header = Split(line, ',');
  std::uint64_t row_number = 1;
  while (std::getline(input, line)) {
    ++row_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const std::vector<std::string> values = Split(line, ',');
    if (values.size() != header.size()) {
      std::fprintf(stderr, "invalid CSV row %llu in %s\n",
          static_cast<unsigned long long>(row_number), path.c_str());
      return false;
    }
    Fields fields;
    for (std::size_t index = 0; index < header.size(); ++index) {
      fields.emplace(header[index], values[index]);
    }
    if (!function(fields, row_number)) return false;
  }
  return input.eof();
}

bool ParseDonors(const std::string& text, DonorSequence* donors) {
  donors->clear();
  if (text.empty() || text == "-") return true;
  for (const std::string& item : Split(text, ';')) {
    const std::size_t colon = item.find(':');
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    if (colon == std::string::npos ||
        !ParseU64(item.substr(0, colon), &offset) ||
        !ParseU64(item.substr(colon + 1), &length) ||
        offset > UINT32_MAX || length > UINT32_MAX) {
      return false;
    }
    donors->push_back({static_cast<std::uint32_t>(offset),
        static_cast<std::uint32_t>(length)});
  }
  return true;
}

std::string DonorText(const DonorSequence& donors) {
  if (donors.empty()) return "-";
  std::string result;
  for (std::size_t index = 0; index < donors.size(); ++index) {
    if (index != 0) result.push_back(';');
    result += std::to_string(donors[index].offset) + ":" +
        std::to_string(donors[index].length);
  }
  return result;
}

bool ParseWinner(const Fields& row, std::uint64_t row_number,
    Winner* winner) {
  std::uint32_t profile_id = 0;
  std::uint32_t stream_class = 0;
  std::uint32_t vr_min = 0;
  std::int64_t recorded_net = 0;
  const auto mode = row.find("mode");
  const auto donors = row.find("donors");
  if (!U32(row, "recipient_region", &winner->region) ||
      !U64(row, "recipient_offset", &winner->offset) ||
      !U32(row, "recipient_length", &winner->length) ||
      mode == row.end() || !U32(row, "expert_mask", &winner->mask) ||
      !U16(row, "mini_model_mask", &winner->mini_mask) ||
      !U32(row, "profile_id", &profile_id) ||
      !U32(row, "stream_class", &stream_class) ||
      donors == row.end() || !ParseDonors(donors->second, &winner->donors) ||
      !U32(row, "vr_min_length", &vr_min) ||
      !U32(row, "vr_event_count", &winner->vr_event_count) ||
      !U64(row, "baseline_payload_bytes", &winner->baseline) ||
      !U64(row, "candidate_payload_bytes", &winner->candidate) ||
      !U64(row, "standalone_plan_bytes", &winner->standalone_side) ||
      !I64(row, "net_gain_bytes", &recorded_net) ||
      profile_id > 255 || stream_class > 255 || vr_min > UINT16_MAX) {
    std::fprintf(stderr, "invalid winner row %llu\n",
        static_cast<unsigned long long>(row_number));
    return false;
  }
  winner->mode = mode->second;
  winner->strength = static_cast<std::uint8_t>((profile_id >> 6) & 3u);
  winner->stream_class = static_cast<std::uint8_t>(stream_class);
  winner->vr_min_length = static_cast<std::uint16_t>(vr_min);
  const std::int64_t expected =
      static_cast<std::int64_t>(winner->baseline) -
      static_cast<std::int64_t>(winner->candidate) -
      static_cast<std::int64_t>(winner->standalone_side);
  if (recorded_net != expected) {
    std::fprintf(stderr, "inconsistent net at row %llu\n",
        static_cast<unsigned long long>(row_number));
    return false;
  }
  winner->standalone_net = recorded_net;
  return true;
}

std::string WinnerKey(const Winner& winner) {
  return std::to_string(winner.region) + "|" +
      std::to_string(winner.offset) + "|" + std::to_string(winner.length) +
      "|" + std::to_string(winner.mask) + "|" +
      std::to_string(winner.mini_mask) + "|" +
      std::to_string(winner.strength) + "|" + DonorText(winner.donors) +
      "|" + std::to_string(winner.vr_min_length) + "|" +
      std::to_string(winner.vr_event_count) + "|" +
      std::to_string(winner.stream_class);
}
std::string FamilyKey(const Winner& winner) {
  return std::to_string(winner.mask) + "|" +
      std::to_string(winner.mini_mask) + "|" +
      std::to_string(winner.strength) + "|" + DonorText(winner.donors) +
      "|" + std::to_string(winner.stream_class);
}

bool ReadU16(std::istream* input, std::uint16_t* value) {
  std::uint8_t b[2] = {};
  input->read(reinterpret_cast<char*>(b), sizeof(b));
  if (!*input) return false;
  *value = static_cast<std::uint16_t>(b[0] | (b[1] << 8));
  return true;
}
bool ReadU32(std::istream* input, std::uint32_t* value) {
  std::uint8_t b[4] = {};
  input->read(reinterpret_cast<char*>(b), sizeof(b));
  if (!*input) return false;
  *value = static_cast<std::uint32_t>(b[0]) |
      (static_cast<std::uint32_t>(b[1]) << 8) |
      (static_cast<std::uint32_t>(b[2]) << 16) |
      (static_cast<std::uint32_t>(b[3]) << 24);
  return true;
}
bool ReadU64(std::istream* input, std::uint64_t* value) {
  std::uint8_t b[8] = {};
  input->read(reinterpret_cast<char*>(b), sizeof(b));
  if (!*input) return false;
  *value = 0;
  for (unsigned int i = 0; i < 8; ++i) {
    *value |= static_cast<std::uint64_t>(b[i]) << (i * 8);
  }
  return true;
}
void WriteU16(std::ostream* output, std::uint16_t value) {
  output->put(static_cast<char>(value));
  output->put(static_cast<char>(value >> 8));
}
void WriteU32(std::ostream* output, std::uint32_t value) {
  for (unsigned int i = 0; i < 4; ++i) {
    output->put(static_cast<char>(value >> (i * 8)));
  }
}
void WriteU64(std::ostream* output, std::uint64_t value) {
  for (unsigned int i = 0; i < 8; ++i) {
    output->put(static_cast<char>(value >> (i * 8)));
  }
}
void WriteVarint(std::ostream* output, std::uint64_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    output->put(static_cast<char>(byte));
  } while (value != 0);
}
std::size_t VarintSize(std::uint64_t value) {
  std::size_t result = 1;
  while (value >= 128) {
    value >>= 7;
    ++result;
  }
  return result;
}
std::uint64_t SignedDelta(std::int64_t value) {
  return value >= 0 ? static_cast<std::uint64_t>(value) << 1 :
      (static_cast<std::uint64_t>(-(value + 1)) << 1) | 1u;
}

bool LoadHeader(const std::string& path, Header* header) {
  std::ifstream input(path, std::ios::binary);
  char magic[4] = {};
  std::uint16_t version = 0;
  std::uint16_t flags = 0;
  std::uint32_t seed_size = 0;
  std::uint16_t count = 0;
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, "F4CD", 4) != 0 ||
      !ReadU16(&input, &version) || !ReadU16(&input, &flags) ||
      !ReadU32(&input, &header->chunk_size) ||
      !ReadU32(&input, &seed_size) ||
      !ReadU64(&input, &header->stream_size) ||
      !ReadU16(&input, &count)) {
    return false;
  }
  input.read(reinterpret_cast<char*>(header->digest.data()),
      header->digest.size());
  return input.good() && version >= 1 && version <= 6 && flags == 1 &&
      header->chunk_size == kChunkSize && seed_size <= 65536;
}

bool BuildLayout(const std::vector<Winner>& winners,
    std::vector<Assignment>* assignments, std::vector<Span>* spans,
    std::size_t* donor_profile_count) {
  std::set<DonorSequence> donor_set;
  for (const Winner& winner : winners) {
    if (winner.mask != 0 && !winner.donors.empty()) {
      donor_set.insert(winner.donors);
    }
  }
  if (donor_set.size() > 64) return false;
  std::map<DonorSequence, std::uint8_t> profiles;
  std::uint8_t next_profile = 0;
  for (const DonorSequence& donors : donor_set) {
    profiles.emplace(donors, next_profile++);
  }

  assignments->clear();
  for (const auto& entry : profiles) {
    for (std::size_t order = 0; order < entry.first.size(); ++order) {
      const Donor& donor = entry.first[order];
      if ((donor.offset & 255u) != 0 || donor.length < 256 ||
          donor.length > 65536 ||
          (donor.length & (donor.length - 1u)) != 0 || order > 255) {
        return false;
      }
      assignments->push_back({entry.second, donor.offset, donor.length,
          static_cast<std::uint8_t>(order)});
    }
  }

  spans->clear();
  std::uint64_t previous_end = 0;
  std::vector<Winner> sorted = winners;
  std::sort(sorted.begin(), sorted.end(),
      [](const Winner& left, const Winner& right) {
        return left.offset < right.offset;
      });
  for (const Winner& winner : sorted) {
    if (winner.mask == 0) continue;
    if (winner.offset < previous_end ||
        winner.length > UINT64_MAX - winner.offset) {
      return false;
    }
    for (const Donor& donor : winner.donors) {
      if (static_cast<std::uint64_t>(donor.offset) + donor.length >
          winner.offset) {
        return false;
      }
    }
    std::uint8_t donor_profile = 0;
    if (!winner.donors.empty()) donor_profile = profiles.at(winner.donors);
    spans->push_back({winner.offset, winner.length, winner.mask,
        winner.stream_class,
        static_cast<std::uint8_t>((winner.strength << 6) | donor_profile),
        winner.mini_mask});
    previous_end = winner.offset + winner.length;
  }
  *donor_profile_count = profiles.size();
  return assignments->size() <= UINT16_MAX;
}

std::size_t ArchiveF4cpSize(const std::vector<Assignment>& assignments,
    const std::vector<Span>& spans) {
  if (assignments.empty() && spans.empty()) return 0;
  std::map<std::uint16_t, std::vector<Assignment>> groups;
  for (const Assignment& assignment : assignments) {
    groups[assignment.profile].push_back(assignment);
  }
  std::size_t bytes = 2 + VarintSize(groups.size());
  for (auto& entry : groups) {
    std::sort(entry.second.begin(), entry.second.end(),
        [](const Assignment& left, const Assignment& right) {
          return left.order < right.order;
        });
    bytes += VarintSize(entry.first) + VarintSize(entry.second.size());
    std::int64_t previous = 0;
    for (std::size_t index = 0; index < entry.second.size(); ++index) {
      const Assignment& assignment = entry.second[index];
      const std::int64_t shifted = assignment.offset >> 8;
      const std::uint64_t encoded = index == 0
          ? static_cast<std::uint64_t>(shifted)
          : SignedDelta(shifted - previous);
      bytes += VarintSize(encoded) + 1;
      previous = shifted;
    }
  }

  using Profile = std::tuple<std::uint32_t, std::uint8_t, std::uint8_t,
      std::uint16_t>;
  std::vector<Profile> profiles;
  std::vector<std::size_t> span_profiles;
  for (const Span& span : spans) {
    const Profile profile{
        span.mask, span.stream_class, span.profile, span.mini_mask};
    const auto found = std::find(profiles.begin(), profiles.end(), profile);
    if (found == profiles.end()) {
      profiles.push_back(profile);
      span_profiles.push_back(profiles.size() - 1);
    } else {
      span_profiles.push_back(
          static_cast<std::size_t>(found - profiles.begin()));
    }
  }
  bytes += VarintSize(profiles.size());
  for (const Profile& profile : profiles) {
    bytes += VarintSize(std::get<0>(profile)) + 2 +
        VarintSize(std::get<3>(profile));
  }
  bytes += VarintSize(spans.size());
  std::uint64_t previous_end = 0;
  for (std::size_t index = 0; index < spans.size(); ++index) {
    const Span& span = spans[index];
    bytes += VarintSize(span.offset - previous_end) +
        VarintSize(span.length) + VarintSize(span_profiles[index]);
    previous_end = span.offset + span.length;
  }
  return bytes;
}

bool F4cpObjective(const std::vector<Winner>& winners,
    std::int64_t* score) {
  std::vector<Assignment> assignments;
  std::vector<Span> spans;
  std::size_t profiles = 0;
  if (!BuildLayout(winners, &assignments, &spans, &profiles)) return false;
  std::int64_t gross = 0;
  for (const Winner& winner : winners) {
    if (winner.mask != 0) {
      gross += static_cast<std::int64_t>(winner.baseline) -
          static_cast<std::int64_t>(winner.candidate);
    }
  }
  *score = gross -
      static_cast<std::int64_t>(ArchiveF4cpSize(assignments, spans));
  return true;
}

void OptimizeSharedProfiles(const std::vector<Winner>& trials,
    std::vector<Winner>* winners) {
  std::set<std::uint32_t> occupied;
  for (const Winner& winner : *winners) occupied.insert(winner.region);
  std::map<std::string, Winner> unique;
  for (const Winner& trial : trials) {
    if (trial.mask == 0 || trial.vr_min_length != 0 ||
        trial.baseline <= trial.candidate ||
        occupied.count(trial.region) != 0) {
      continue;
    }
    const std::string key = WinnerKey(trial);
    const auto found = unique.find(key);
    if (found == unique.end() ||
        trial.candidate < found->second.candidate) {
      unique[key] = trial;
    }
  }

  std::int64_t current_score = 0;
  if (!F4cpObjective(*winners, &current_score)) return;
  while (true) {
    occupied.clear();
    for (const Winner& winner : *winners) occupied.insert(winner.region);
    std::map<std::string, std::map<std::uint32_t, Winner>> families;
    for (const auto& entry : unique) {
      const Winner& trial = entry.second;
      if (occupied.count(trial.region) != 0) continue;
      auto& by_region = families[FamilyKey(trial)];
      const auto found = by_region.find(trial.region);
      if (found == by_region.end() ||
          trial.candidate < found->second.candidate) {
        by_region[trial.region] = trial;
      }
    }

    std::int64_t best_delta = 0;
    std::vector<Winner> best_addition;
    for (auto& family : families) {
      std::vector<Winner> ranked;
      for (const auto& item : family.second) ranked.push_back(item.second);
      std::sort(ranked.begin(), ranked.end(),
          [](const Winner& left, const Winner& right) {
            const std::int64_t left_gross =
                static_cast<std::int64_t>(left.baseline) -
                static_cast<std::int64_t>(left.candidate);
            const std::int64_t right_gross =
                static_cast<std::int64_t>(right.baseline) -
                static_cast<std::int64_t>(right.candidate);
            return left_gross != right_gross
                ? left_gross > right_gross : left.offset < right.offset;
          });
      for (std::size_t count = 1; count <= ranked.size(); ++count) {
        std::vector<Winner> candidate = *winners;
        candidate.insert(
            candidate.end(), ranked.begin(), ranked.begin() + count);
        std::int64_t score = 0;
        if (!F4cpObjective(candidate, &score)) continue;
        if (score - current_score > best_delta) {
          best_delta = score - current_score;
          best_addition.assign(ranked.begin(), ranked.begin() + count);
        }
      }
    }
    if (best_addition.empty()) break;
    winners->insert(
        winners->end(), best_addition.begin(), best_addition.end());
    current_score += best_delta;
  }
}

bool PatternMatches(std::ifstream* stream, std::uint64_t offset,
    std::uint16_t code) {
  if (code == 0 || code > scr2::kTokenCount) return false;
  const scr2::TokenInfo& token = scr2::kTokens[code];
  std::vector<std::uint8_t> bytes(token.length);
  stream->clear();
  stream->seekg(static_cast<std::streamoff>(offset));
  stream->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  return stream->good() &&
      std::memcmp(bytes.data(), scr2::PatternData(code), bytes.size()) == 0;
}

std::vector<ReplayEvent> ScanPatterns(
    std::ifstream* stream, const Winner& winner) {
  std::vector<std::uint8_t> bytes(winner.length);
  stream->clear();
  stream->seekg(static_cast<std::streamoff>(winner.offset));
  stream->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!stream->good()) return {};
  std::vector<ReplayEvent> events;
  std::size_t local = 0;
  while (local < bytes.size()) {
    std::uint16_t best = 0;
    std::size_t best_length = 0;
    for (std::uint16_t code = 1; code <= scr2::kTokenCount; ++code) {
      const scr2::TokenInfo& token = scr2::kTokens[code];
      if (token.length < winner.vr_min_length ||
          token.length <= best_length ||
          local + token.length > bytes.size() ||
          bytes[local] != token.first) {
        continue;
      }
      const std::uint64_t absolute = winner.offset + local;
      const std::uint64_t end = absolute + token.length;
      if (absolute / kIoBlockBytes != (end - 1) / kIoBlockBytes) continue;
      if (std::memcmp(bytes.data() + local,
          scr2::PatternData(code), token.length) == 0) {
        best = code;
        best_length = token.length;
      }
    }
    if (best == 0) {
      ++local;
    } else {
      events.push_back({winner.offset + local, best});
      local += best_length;
    }
  }
  return events;
}

std::string EventKey(std::uint32_t region, const std::string& mode) {
  return std::to_string(region) + "|" + mode;
}

bool BuildReplayPayload(const std::vector<Winner>& winners,
    const Options& options, const Header& header,
    std::vector<std::uint8_t>* payload, std::size_t* replay_regions,
    std::size_t* event_count, std::size_t* pattern_count) {
  std::unordered_map<std::string,
      std::set<std::pair<std::uint64_t, std::uint16_t>>> recorded;
  for (const std::string& path : options.event_csvs) {
    std::ifstream probe(path);
    if (!probe.is_open()) continue;
    probe.close();
    if (!ReadCsv(path, [&](const Fields& row, std::uint64_t number) {
      std::uint32_t region = 0;
      std::uint64_t offset = 0;
      std::uint16_t pattern = 0;
      const auto mode = row.find("mode");
      if (!U32(row, "recipient_region", &region) ||
          !U64(row, "event_offset", &offset) ||
          !U16(row, "pattern", &pattern) || mode == row.end()) {
        std::fprintf(stderr, "invalid event row %llu in %s\n",
            static_cast<unsigned long long>(number), path.c_str());
        return false;
      }
      recorded[EventKey(region, mode->second)].insert({offset, pattern});
      return true;
    })) {
      return false;
    }
  }

  bool needs_replay = false;
  for (const Winner& winner : winners) {
    if (winner.vr_min_length != 0) needs_replay = true;
  }
  if (!needs_replay) {
    payload->clear();
    *replay_regions = *event_count = *pattern_count = 0;
    return true;
  }
  if (options.post_r1.empty()) {
    std::fprintf(stderr, "--post-r1 is required by selected SCR2 actions\n");
    return false;
  }
  std::ifstream stream(options.post_r1, std::ios::binary);
  stream.seekg(0, std::ios::end);
  if (!stream || static_cast<std::uint64_t>(stream.tellg()) !=
      header.stream_size) {
    std::fprintf(stderr, "post-R1 stream size mismatch\n");
    return false;
  }

  std::vector<ReplayEvent> events;
  *replay_regions = 0;
  for (const Winner& winner : winners) {
    if (winner.vr_min_length == 0) continue;
    ++*replay_regions;
    std::vector<ReplayEvent> found;
    if (winner.vr_min_length == kCostPositiveScr2) {
      const auto stored = recorded.find(EventKey(winner.region, winner.mode));
      if (stored == recorded.end()) {
        std::fprintf(stderr,
            "missing SCR2 events for region %u mode %s\n",
            winner.region, winner.mode.c_str());
        return false;
      }
      for (const auto& event : stored->second) {
        found.push_back({event.first, event.second});
      }
    } else {
      found = ScanPatterns(&stream, winner);
    }
    if (found.size() != winner.vr_event_count) {
      std::fprintf(stderr, "SCR2 event mismatch in region %u: %zu != %u\n",
          winner.region, found.size(), winner.vr_event_count);
      return false;
    }
    std::uint64_t previous_end = winner.offset;
    for (const ReplayEvent& event : found) {
      if (event.pattern == 0 || event.pattern > scr2::kTokenCount) return false;
      const std::uint64_t length = scr2::kTokens[event.pattern].length;
      const std::uint64_t end = event.offset + length;
      if (event.offset < winner.offset || event.offset < previous_end ||
          end > winner.offset + winner.length ||
          event.offset / kIoBlockBytes != (end - 1) / kIoBlockBytes ||
          !PatternMatches(&stream, event.offset, event.pattern)) {
        std::fprintf(stderr, "invalid SCR2 event %llu:%u in region %u\n",
            static_cast<unsigned long long>(event.offset), event.pattern,
            winner.region);
        return false;
      }
      previous_end = end;
      events.push_back(event);
    }
  }
  std::sort(events.begin(), events.end(),
      [](const ReplayEvent& left, const ReplayEvent& right) {
        return std::tie(left.offset, left.pattern) <
            std::tie(right.offset, right.pattern);
      });

  std::vector<std::uint16_t> used;
  std::map<std::uint16_t, std::uint16_t> compact;
  for (const ReplayEvent& event : events) {
    if (compact.count(event.pattern) == 0) {
      compact[event.pattern] = static_cast<std::uint16_t>(used.size());
      used.push_back(event.pattern);
    }
  }
  std::ostringstream encoded(std::ios::binary);
  encoded.put(1);
  WriteU16(&encoded, static_cast<std::uint16_t>(used.size()));
  for (const std::uint16_t code : used) {
    const scr2::TokenInfo& token = scr2::kTokens[code];
    WriteU16(&encoded, token.length);
    encoded.write(reinterpret_cast<const char*>(scr2::PatternData(code)),
        token.length);
  }
  WriteU32(&encoded, static_cast<std::uint32_t>(events.size()));
  std::uint64_t previous_end = 0;
  for (const ReplayEvent& event : events) {
    if (event.offset < previous_end) {
      std::fprintf(stderr, "overlapping global SCR2 events\n");
      return false;
    }
    WriteVarint(&encoded, event.offset - previous_end);
    WriteVarint(&encoded, compact[event.pattern]);
    previous_end = event.offset + scr2::kTokens[event.pattern].length;
  }
  const std::string bytes = encoded.str();
  payload->assign(bytes.begin(), bytes.end());
  *event_count = events.size();
  *pattern_count = used.size();
  return true;
}

bool WriteF4cp(const std::string& path, const Header& header,
    const std::vector<Assignment>& assignments,
    const std::vector<Span>& spans) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write("F4CP", 4);
  WriteU16(&output, 6);
  WriteU16(&output, 0);
  WriteU32(&output, header.chunk_size);
  std::uint32_t max_length = 0;
  for (const Assignment& assignment : assignments) {
    max_length = std::max(max_length, assignment.length);
  }
  WriteU32(&output, max_length);
  WriteU64(&output, header.stream_size);
  WriteU16(&output, static_cast<std::uint16_t>(assignments.size()));
  output.write(reinterpret_cast<const char*>(header.digest.data()),
      header.digest.size());
  for (const Assignment& assignment : assignments) {
    WriteU16(&output, assignment.profile);
    WriteU32(&output, assignment.offset);
    WriteU16(&output, assignment.length == 65536 ? 0 :
        static_cast<std::uint16_t>(assignment.length));
    output.put(static_cast<char>(assignment.order));
  }
  WriteU32(&output, static_cast<std::uint32_t>(spans.size()));
  for (const Span& span : spans) {
    WriteU64(&output, span.offset);
    WriteU32(&output, span.length);
    WriteU32(&output, span.mask);
    output.put(static_cast<char>(span.stream_class));
    output.put(static_cast<char>(span.profile));
    WriteU16(&output, span.mini_mask);
  }
  return output.good();
}

bool WriteF4vr(const std::string& path, const Header& header,
    const std::vector<std::uint8_t>& payload) {
  if (payload.empty()) {
    std::remove(path.c_str());
    return true;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write("F4VR", 4);
  WriteU16(&output, 1);
  WriteU16(&output, 0);
  WriteU64(&output, header.stream_size);
  output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  return output.good();
}

bool WriteSelected(const std::string& path,
    std::vector<Winner> winners) {
  std::sort(winners.begin(), winners.end(),
      [](const Winner& left, const Winner& right) {
        return left.offset < right.offset;
      });
  std::ofstream output(path, std::ios::trunc);
  output << "region,offset,length,mode,mask,mini_mask,strength_code,"
      "donors,vr_min_length,vr_event_count,gross_gain,standalone_side,"
      "standalone_net\n";
  for (const Winner& winner : winners) {
    output << winner.region << ',' << winner.offset << ',' << winner.length
        << ',' << winner.mode << ',' << winner.mask << ','
        << winner.mini_mask << ',' << static_cast<unsigned>(winner.strength)
        << ',' << DonorText(winner.donors) << ',' << winner.vr_min_length
        << ',' << winner.vr_event_count << ','
        << (static_cast<std::int64_t>(winner.baseline) -
            static_cast<std::int64_t>(winner.candidate))
        << ',' << winner.standalone_side << ','
        << winner.standalone_net << '\n';
  }
  return output.good();
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) return false;
    const std::string value = argv[++index];
    if (option == "--candidate-plan") options->candidate_plan = value;
    else if (option == "--output-f4cp") options->output_f4cp = value;
    else if (option == "--output-f4vr") options->output_f4vr = value;
    else if (option == "--output-selected") options->output_selected = value;
    else if (option == "--summary") options->summary = value;
    else if (option == "--post-r1") options->post_r1 = value;
    else if (option == "--selected") options->selected_csvs.push_back(value);
    else if (option == "--trials") options->trial_csvs.push_back(value);
    else if (option == "--events") options->event_csvs.push_back(value);
    else if (option == "--baseline-s1") {
      if (!ParseU64(value, &options->baseline_s1)) return false;
    } else if (option == "--candidate-s1") {
      if (!ParseU64(value, &options->candidate_s1)) return false;
    } else if (option == "--minimum-net") {
      if (!ParseI64(value, &options->minimum_net)) return false;
    } else if (option == "--allow-f4cp") {
      options->allow_f4cp = value == "1";
      if (value != "0" && value != "1") return false;
    } else if (option == "--allow-f4vr") {
      options->allow_f4vr = value == "1";
      if (value != "0" && value != "1") return false;
    } else {
      return false;
    }
  }
  return !options->candidate_plan.empty() &&
      !options->output_f4cp.empty() && !options->output_f4vr.empty() &&
      !options->output_selected.empty() && !options->summary.empty() &&
      !options->selected_csvs.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    Usage();
    return 2;
  }
  Header header;
  if (!LoadHeader(options.candidate_plan, &header)) {
    std::fprintf(stderr, "invalid F4CD candidate plan\n");
    return 2;
  }

  std::map<std::uint32_t, Winner> accepted;
  for (const std::string& path : options.selected_csvs) {
    std::ifstream probe(path);
    if (!probe.is_open()) continue;
    probe.close();
    if (!ReadCsv(path, [&](const Fields& row, std::uint64_t number) {
      Winner winner;
      if (!ParseWinner(row, number, &winner)) return false;
      const auto status = row.find("status");
      const bool active = winner.mask != 0 || winner.vr_min_length != 0;
      if (!active || status == row.end() ||
          status->second != "accepted_after_side" ||
          winner.standalone_net < options.minimum_net) {
        return true;
      }
      if ((!options.allow_f4cp && winner.mask != 0) ||
          (!options.allow_f4vr && winner.vr_min_length != 0)) {
        return true;
      }
      const auto found = accepted.find(winner.region);
      if (found == accepted.end() ||
          winner.standalone_net > found->second.standalone_net) {
        accepted[winner.region] = winner;
      }
      return true;
    })) {
      std::fprintf(stderr, "cannot parse selected ledger: %s\n", path.c_str());
      return 2;
    }
  }

  std::map<std::string, Winner> trial_map;
  for (const std::string& path : options.trial_csvs) {
    std::ifstream probe(path);
    if (!probe.is_open()) continue;
    probe.close();
    if (!ReadCsv(path, [&](const Fields& row, std::uint64_t number) {
      Winner winner;
      if (!ParseWinner(row, number, &winner)) return false;
      if (winner.mask == 0 || winner.vr_min_length != 0 ||
          winner.baseline <= winner.candidate) {
        return true;
      }
      const std::string key = WinnerKey(winner);
      const auto found = trial_map.find(key);
      if (found == trial_map.end() ||
          winner.candidate < found->second.candidate) {
        trial_map[key] = winner;
      }
      return true;
    })) {
      std::fprintf(stderr, "cannot parse trial ledger: %s\n", path.c_str());
      return 2;
    }
  }

  std::vector<Winner> winners;
  for (const auto& entry : accepted) winners.push_back(entry.second);
  std::vector<Winner> trials;
  for (const auto& entry : trial_map) trials.push_back(entry.second);
  if (options.allow_f4cp) OptimizeSharedProfiles(trials, &winners);
  std::sort(winners.begin(), winners.end(),
      [](const Winner& left, const Winner& right) {
        return left.offset < right.offset;
      });
  std::uint64_t previous_end = 0;
  for (const Winner& winner : winners) {
    if (winner.offset < previous_end ||
        winner.offset + winner.length > header.stream_size) {
      std::fprintf(stderr, "overlapping/out-of-range winner %u\n",
          winner.region);
      return 2;
    }
    previous_end = winner.offset + winner.length;
  }

  std::vector<Assignment> assignments;
  std::vector<Span> spans;
  std::size_t donor_profiles = 0;
  if (!BuildLayout(winners, &assignments, &spans, &donor_profiles)) {
    std::fprintf(stderr, "selected F4CP plan is not representable\n");
    return 2;
  }
  std::vector<std::uint8_t> replay_payload;
  std::size_t replay_regions = 0;
  std::size_t replay_events = 0;
  std::size_t replay_patterns = 0;
  if (!BuildReplayPayload(winners, options, header, &replay_payload,
      &replay_regions, &replay_events, &replay_patterns)) {
    return 2;
  }

  std::int64_t gross = 0;
  std::int64_t standalone_net = 0;
  for (const Winner& winner : winners) {
    gross += static_cast<std::int64_t>(winner.baseline) -
        static_cast<std::int64_t>(winner.candidate);
    standalone_net += winner.standalone_net;
  }
  std::size_t f4cp_archive = ArchiveF4cpSize(assignments, spans);
  std::size_t f4vr_archive = replay_payload.size();
  std::int64_t plan_net = gross -
      static_cast<std::int64_t>(f4cp_archive + f4vr_archive);
  const std::uint64_t s1_growth =
      options.candidate_s1 > options.baseline_s1
      ? options.candidate_s1 - options.baseline_s1 : 0;
  std::int64_t hutter_net =
      plan_net - static_cast<std::int64_t>(s1_growth);
  const bool fallback = !winners.empty() && hutter_net <= 0;
  if (fallback) {
    winners.clear();
    assignments.clear();
    spans.clear();
    replay_payload.clear();
    donor_profiles = replay_regions = replay_events = replay_patterns = 0;
    gross = standalone_net = plan_net = hutter_net = 0;
    f4cp_archive = f4vr_archive = 0;
  }

  if (!WriteF4cp(options.output_f4cp, header, assignments, spans) ||
      !WriteF4vr(options.output_f4vr, header, replay_payload) ||
      !WriteSelected(options.output_selected, winners)) {
    std::fprintf(stderr, "cannot write selected plans\n");
    return 2;
  }

  std::ofstream summary(options.summary, std::ios::trunc);
  summary << "format=F4CP-v6-external/F4CP-v7-archive+F4VR-v1\n"
      << "stream_bytes=" << header.stream_size << '\n'
      << "selected_regions=" << winners.size() << '\n'
      << "f4cp_required=" << (!assignments.empty() || !spans.empty()) << '\n'
      << "f4vr_required=" << (!replay_payload.empty()) << '\n'
      << "donor_profiles=" << donor_profiles << '\n'
      << "donor_assignments=" << assignments.size() << '\n'
      << "expert_spans=" << spans.size() << '\n'
      << "virtual_replay_regions=" << replay_regions << '\n'
      << "virtual_replay_events=" << replay_events << '\n'
      << "virtual_replay_patterns=" << replay_patterns << '\n'
      << "gross_payload_gain_bytes=" << gross << '\n'
      << "sum_conservative_standalone_net_bytes=" << standalone_net << '\n'
      << "f4cp_compact_archive_bytes=" << f4cp_archive << '\n'
      << "f4vr_compact_archive_bytes=" << f4vr_archive << '\n'
      << "combined_archive_plan_bytes=" << (f4cp_archive + f4vr_archive)
      << '\n'
      << "plan_net_bytes=" << plan_net << '\n'
      << "baseline_s1_bytes=" << options.baseline_s1 << '\n'
      << "candidate_s1_bytes=" << options.candidate_s1 << '\n'
      << "s1_growth_bytes=" << s1_growth << '\n'
      << "projected_hutter_net_bytes=" << hutter_net << '\n'
      << "fell_back_to_baseline=" << fallback << '\n';
  if (!summary.good()) return 2;
  summary.close();
  std::ifstream report(options.summary);
  std::cout << report.rdbuf();
  return 0;
}
