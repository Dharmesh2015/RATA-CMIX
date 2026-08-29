#include "virtual_replay_plan.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "fx4_config.h"
#include "models/scr2_tokens.h"

namespace {

constexpr char kExternalMagic[4] = {'F', '4', 'V', 'R'};
constexpr std::uint16_t kExternalVersion = 1;
constexpr std::uint8_t kExplicitArchiveVersion = 1;
constexpr std::uint8_t kCausalScr2ArchiveVersion = 7;
constexpr std::uint8_t kCausalScr2AllProfile = 0;
constexpr std::uint8_t kCausalScr2Group9Profile = 1;
constexpr std::uint8_t kCausalScr2PositiveProfile = 2;
constexpr std::uint8_t kCausalScr2CustomProfile = 255;
constexpr std::uint32_t kMaximumPatterns = 65535;
constexpr std::uint32_t kMaximumPatternBytes = 65535;
constexpr std::uint32_t kMaximumEvents = 16u * 1024u * 1024u;

using CausalScr2Mask = std::array<std::uint8_t, 16>;

constexpr CausalScr2Mask kCausalScr2Group9Mask = {
    0x05, 0x20, 0x60, 0x00, 0x80, 0x60, 0x00, 0x44,
    0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

// Exact warm region-0 mining retained patterns 1, 3, 7, 11, 15, 38 and 76.
// This profile is encoded by ID, so the archive still pays only seven bytes.
constexpr CausalScr2Mask kCausalScr2PositiveMask = {
    0x45, 0x44, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

std::uint8_t CausalScr2ProfileId(const CausalScr2Mask& mask) {
  CausalScr2Mask all;
  all.fill(0xffu);
  if (mask == all) return kCausalScr2AllProfile;
  if (mask == kCausalScr2Group9Mask) return kCausalScr2Group9Profile;
  if (mask == kCausalScr2PositiveMask) {
    return kCausalScr2PositiveProfile;
  }
  return kCausalScr2CustomProfile;
}

bool LoadCausalScr2Profile(
    std::uint8_t profile, CausalScr2Mask* mask) {
  if (profile == kCausalScr2AllProfile) {
    mask->fill(0xffu);
    return true;
  }
  if (profile == kCausalScr2Group9Profile) {
    *mask = kCausalScr2Group9Mask;
    return true;
  }
  if (profile == kCausalScr2PositiveProfile) {
    *mask = kCausalScr2PositiveMask;
    return true;
  }
  return profile == kCausalScr2CustomProfile;
}

bool ReadU8(std::istream* input, std::uint8_t* value) {
  const int c = input->get();
  if (c == EOF) return false;
  *value = static_cast<std::uint8_t>(c);
  return true;
}

bool WriteU8(std::ostream* output, std::uint8_t value) {
  output->put(static_cast<char>(value));
  return output->good();
}

bool ReadU16(std::istream* input, std::uint16_t* value) {
  std::uint8_t lo = 0;
  std::uint8_t hi = 0;
  if (!ReadU8(input, &lo) || !ReadU8(input, &hi)) return false;
  *value = static_cast<std::uint16_t>(lo | (hi << 8));
  return true;
}

bool WriteU16(std::ostream* output, std::uint16_t value) {
  return WriteU8(output, static_cast<std::uint8_t>(value)) &&
      WriteU8(output, static_cast<std::uint8_t>(value >> 8));
}

bool ReadU32(std::istream* input, std::uint32_t* value) {
  *value = 0;
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    std::uint8_t byte = 0;
    if (!ReadU8(input, &byte)) return false;
    *value |= static_cast<std::uint32_t>(byte) << shift;
  }
  return true;
}

bool WriteU32(std::ostream* output, std::uint32_t value) {
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    if (!WriteU8(output, static_cast<std::uint8_t>(value >> shift))) {
      return false;
    }
  }
  return true;
}

bool ReadU64(std::istream* input, std::uint64_t* value) {
  *value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    std::uint8_t byte = 0;
    if (!ReadU8(input, &byte)) return false;
    *value |= static_cast<std::uint64_t>(byte) << shift;
  }
  return true;
}

bool ReadVarint(std::istream* input, std::uint64_t* value) {
  *value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 7) {
    std::uint8_t byte = 0;
    if (!ReadU8(input, &byte)) return false;
    *value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return true;
  }
  return false;
}

bool WriteVarint(std::ostream* output, std::uint64_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    if (!WriteU8(output, byte)) return false;
  } while (value != 0);
  return true;
}

}  // namespace

bool VirtualReplayPlan::LoadExternal(const std::string& path,
    std::uint64_t stream_size) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  char magic[4] = {};
  input.read(magic, sizeof(magic));
  std::uint16_t version = 0;
  std::uint16_t flags = 0;
  std::uint64_t recorded_size = 0;
  if (input.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
      std::memcmp(magic, kExternalMagic, sizeof(magic)) != 0 ||
      !ReadU16(&input, &version) || !ReadU16(&input, &flags) ||
      !ReadU64(&input, &recorded_size) ||
      version != kExternalVersion || flags != 0 ||
      recorded_size != stream_size) {
    return false;
  }
  std::uint8_t archive_version = 0;
  if (!ReadU8(&input, &archive_version) ||
      archive_version != kExplicitArchiveVersion) {
    return false;
  }
  causal_scr2_ = false;
  causal_all_ = false;
  causal_ranges_.clear();
  return ReadPayload(&input, stream_size) && input.peek() == EOF;
}

bool VirtualReplayPlan::ConfigureCausalScr2(
    const std::string& specification, std::uint64_t stream_size,
    unsigned int prior_code, const std::string& pattern_specification) {
  if (stream_size == 0 || prior_code > 2) return false;
  patterns_.clear();
  events_.clear();
  causal_ranges_.clear();
  causal_scr2_ = true;
  causal_prior_code_ = static_cast<std::uint8_t>(prior_code);
  causal_pattern_mask_.fill(0xffu);
  if (!pattern_specification.empty() && pattern_specification != "all") {
    causal_pattern_mask_.fill(0);
    const char* cursor = pattern_specification.c_str();
    while (*cursor) {
      char* end = nullptr;
      unsigned long first = std::strtoul(cursor, &end, 10);
      if (end == cursor || first < 1 || first > scr2::kTokenCount) {
        return false;
      }
      unsigned long last = first;
      if (*end == '-') {
        cursor = end + 1;
        last = std::strtoul(cursor, &end, 10);
        if (end == cursor || last < first || last > scr2::kTokenCount) {
          return false;
        }
      }
      for (unsigned long code = first; code <= last; ++code) {
        causal_pattern_mask_[(code - 1) >> 3] |=
            static_cast<std::uint8_t>(1u << ((code - 1) & 7u));
      }
      if (*end == '\0') break;
      if (*end != ',') return false;
      cursor = end + 1;
    }
  }
  causal_all_ = specification == "1" || specification == "all";
  if (causal_all_) return true;

  std::ifstream input(specification);
  if (!input.is_open()) return false;
  std::string line;
  while (std::getline(input, line)) {
    const char* begin = line.c_str();
    while (*begin == ' ' || *begin == '\t') ++begin;
    if (*begin < '0' || *begin > '9') continue;
    char* end = nullptr;
    const unsigned long long offset = std::strtoull(begin, &end, 10);
    if (end == begin || *end != ',') return false;
    begin = end + 1;
    const unsigned long long length = std::strtoull(begin, &end, 10);
    if (end == begin || length == 0 ||
        offset > stream_size || length > stream_size - offset) {
      return false;
    }
    causal_ranges_.push_back({offset, length});
  }
  if (causal_ranges_.empty()) return false;
  std::sort(causal_ranges_.begin(), causal_ranges_.end(),
      [](const Range& left, const Range& right) {
        return left.offset < right.offset;
      });
  std::uint64_t previous_end = 0;
  for (const Range& range : causal_ranges_) {
    if (range.offset < previous_end) return false;
    previous_end = range.offset + range.length;
  }
  return true;
}

bool VirtualReplayPlan::WriteArchive(std::ofstream* output) const {
  if (causal_scr2_) {
    const std::uint8_t profile =
        CausalScr2ProfileId(causal_pattern_mask_);
    if (!WriteU8(output, kCausalScr2ArchiveVersion) ||
        !WriteU8(output, causal_prior_code_) ||
        !WriteU8(output, profile)) {
      return false;
    }
    if (profile == kCausalScr2CustomProfile) {
      output->write(
          reinterpret_cast<const char*>(causal_pattern_mask_.data()),
          causal_pattern_mask_.size());
      if (!output->good()) return false;
    }
    if (causal_all_) return WriteU32(output, 0);
    if (causal_ranges_.empty() ||
        causal_ranges_.size() > std::numeric_limits<std::uint32_t>::max() ||
        !WriteU32(output,
            static_cast<std::uint32_t>(causal_ranges_.size()))) {
      return false;
    }
    std::uint64_t previous_end = 0;
    for (const Range& range : causal_ranges_) {
      if (!WriteVarint(output, range.offset - previous_end) ||
          !WriteVarint(output, range.length)) {
        return false;
      }
      previous_end = range.offset + range.length;
    }
    return output->good();
  }
  if (!WriteU8(output, kExplicitArchiveVersion) ||
      !WriteU16(output, static_cast<std::uint16_t>(patterns_.size()))) {
    return false;
  }
  for (const auto& pattern : patterns_) {
    if (!WriteU16(output, static_cast<std::uint16_t>(pattern.size()))) {
      return false;
    }
    output->write(reinterpret_cast<const char*>(pattern.data()), pattern.size());
    if (!output->good()) return false;
  }
  if (!WriteU32(output, static_cast<std::uint32_t>(events_.size()))) {
    return false;
  }
  std::uint64_t previous_end = 0;
  for (const Event& event : events_) {
    const std::uint64_t gap = event.offset - previous_end;
    if (!WriteVarint(output, gap) || !WriteVarint(output, event.pattern)) {
      return false;
    }
    previous_end = event.offset + patterns_[event.pattern].size();
  }
  return output->good();
}

bool VirtualReplayPlan::ReadArchive(std::ifstream* input,
    std::uint64_t stream_size) {
  std::uint8_t version = 0;
  if (!ReadU8(input, &version)) return false;
  if (version == kExplicitArchiveVersion) {
    causal_scr2_ = false;
    causal_all_ = false;
    causal_ranges_.clear();
    return ReadPayload(input, stream_size);
  }
  if (version != kCausalScr2ArchiveVersion) return false;
  std::uint8_t prior_code = 0;
  std::uint8_t profile = 0;
  std::uint32_t range_count = 0;
  if (!ReadU8(input, &prior_code) || prior_code > 2 ||
      !ReadU8(input, &profile) ||
      !LoadCausalScr2Profile(profile, &causal_pattern_mask_)) {
    return false;
  }
  if (profile == kCausalScr2CustomProfile) {
    input->read(reinterpret_cast<char*>(causal_pattern_mask_.data()),
        causal_pattern_mask_.size());
    if (input->gcount() !=
        static_cast<std::streamsize>(causal_pattern_mask_.size())) {
      return false;
    }
  }
  if (!ReadU32(input, &range_count)) return false;
  patterns_.clear();
  events_.clear();
  causal_ranges_.clear();
  causal_scr2_ = true;
  causal_prior_code_ = prior_code;
  causal_all_ = range_count == 0;
  if (causal_all_) return stream_size != 0;
  causal_ranges_.reserve(range_count);
  std::uint64_t previous_end = 0;
  for (std::uint32_t i = 0; i < range_count; ++i) {
    std::uint64_t gap = 0;
    std::uint64_t length = 0;
    if (!ReadVarint(input, &gap) || !ReadVarint(input, &length) ||
        length == 0 || gap > stream_size - previous_end) {
      return false;
    }
    const std::uint64_t offset = previous_end + gap;
    if (length > stream_size - offset) return false;
    causal_ranges_.push_back({offset, length});
    previous_end = offset + length;
  }
  return true;
}

bool VirtualReplayPlan::ReadPayload(std::istream* input,
    std::uint64_t stream_size) {
  patterns_.clear();
  events_.clear();
  std::uint16_t pattern_count = 0;
  if (!ReadU16(input, &pattern_count) ||
      pattern_count == 0 || pattern_count > kMaximumPatterns) {
    return false;
  }
  patterns_.reserve(pattern_count);
  for (std::uint32_t i = 0; i < pattern_count; ++i) {
    std::uint16_t length = 0;
    if (!ReadU16(input, &length) || length == 0 ||
        length > kMaximumPatternBytes) {
      return false;
    }
    std::vector<std::uint8_t> pattern(length);
    input->read(reinterpret_cast<char*>(pattern.data()), length);
    if (input->gcount() != static_cast<std::streamsize>(length)) return false;
    patterns_.push_back(std::move(pattern));
  }
  std::uint32_t event_count = 0;
  if (!ReadU32(input, &event_count) || event_count == 0 ||
      event_count > kMaximumEvents) {
    return false;
  }
  events_.reserve(event_count);
  std::uint64_t previous_end = 0;
  for (std::uint32_t i = 0; i < event_count; ++i) {
    std::uint64_t gap = 0;
    std::uint64_t pattern = 0;
    if (!ReadVarint(input, &gap) || !ReadVarint(input, &pattern) ||
        pattern >= patterns_.size() ||
        gap > std::numeric_limits<std::uint64_t>::max() - previous_end) {
      return false;
    }
    const std::uint64_t offset = previous_end + gap;
    events_.push_back(
        {offset, static_cast<std::uint16_t>(pattern)});
    const std::uint64_t length = patterns_[pattern].size();
    if (offset > std::numeric_limits<std::uint64_t>::max() - length) {
      return false;
    }
    previous_end = offset + length;
  }
  return Validate(stream_size);
}

bool VirtualReplayPlan::Validate(std::uint64_t stream_size) const {
  if (patterns_.empty() || events_.empty()) return false;
  std::uint64_t previous_end = 0;
  for (const Event& event : events_) {
    if (event.pattern >= patterns_.size()) return false;
    const std::uint64_t length = patterns_[event.pattern].size();
    if (event.offset < previous_end || event.offset + length > stream_size) {
      return false;
    }
    // The streaming coder can replay an event without lookahead only when it
    // stays in the same I/O block. This also prevents crossing the current
    // one-MiB donor-recipient boundary.
    if (event.offset / FX4_IO_BUFFER_BYTES !=
        (event.offset + length - 1) / FX4_IO_BUFFER_BYTES) {
      return false;
    }
    previous_end = event.offset + length;
  }
  return true;
}

CausalScr2Matcher::CausalScr2Matcher(unsigned int prior_code,
    const std::array<std::uint8_t, 16>& pattern_mask,
    std::uint64_t stream_size)
    : pattern_mask_(pattern_mask),
      // Prefix experiments must use the same position bands as the eventual
      // full post-R1 Hutter stream. Full-file runs already have this size.
      stream_size_(std::max(stream_size, scr2::kLearnedPostR1Size)),
      prior_code_(static_cast<std::uint8_t>(prior_code)) {
  bucket_head_.fill(-1);
  bucket_next_.fill(-1);
  for (std::uint16_t code = 1; code <= scr2::kTokenCount; ++code) {
    const std::uint16_t pattern_length = scr2::kTokens[code].length;
    if (pattern_length <= 6) continue;
    const std::uint8_t* pattern = scr2::PatternData(code);
    std::uint16_t unique_length = 0;
    for (std::uint16_t length = 6; length < pattern_length; ++length) {
      bool unique = true;
      for (std::uint16_t other = 1; other <= scr2::kTokenCount; ++other) {
        if (other == code || scr2::kTokens[other].length < length) continue;
        if (std::memcmp(pattern, scr2::PatternData(other), length) == 0) {
          unique = false;
          break;
        }
      }
      if (unique) {
        unique_length = length;
        break;
      }
    }
    if (unique_length == 0) continue;
    prefix_length_[code] = unique_length;
    const std::uint8_t bucket = pattern[unique_length - 1];
    bucket_next_[code] = bucket_head_[bucket];
    bucket_head_[bucket] = static_cast<std::int16_t>(code);
  }
  ResetModel();
}

void CausalScr2Matcher::ResetModel() {
  std::uint16_t no = 1;
  std::uint16_t yes = 1;
  if (prior_code_ == 1) yes = 3;
  if (prior_code_ == 2) yes = 7;
  for (auto& pattern_counts : counts_) {
    for (Counts& counts : pattern_counts) {
      counts.no = no;
      counts.yes = yes;
    }
  }
  for (auto& pattern_counts : line_counts_) {
    for (Counts& counts : pattern_counts) {
      counts.no = no;
      counts.yes = yes;
    }
  }
  for (auto& pattern_counts : class_counts_) {
    for (Counts& counts : pattern_counts) {
      counts.no = no;
      counts.yes = yes;
    }
  }
  for (auto& pattern_counts : byte_class_counts_) {
    for (Counts& counts : pattern_counts) {
      counts.no = no;
      counts.yes = yes;
    }
  }
  for (auto& pattern_counts : ppmd_counts_) {
    for (Counts& counts : pattern_counts) {
      counts.no = no;
      counts.yes = yes;
    }
  }
}

void CausalScr2Matcher::ObserveByte(std::uint8_t value) {
  history_[bytes_seen_ & 255u] = value;
  ++bytes_seen_;
}

bool CausalScr2Matcher::PrefixMatches(
    std::uint16_t pattern, std::uint16_t length) const {
  if (bytes_seen_ < length) return false;
  const std::uint8_t* expected = scr2::PatternData(pattern);
  const std::uint64_t begin = bytes_seen_ - length;
  for (std::uint16_t i = 0; i < length; ++i) {
    if (history_[(begin + i) & 255u] != expected[i]) return false;
  }
  return true;
}

unsigned int CausalScr2Matcher::LineBucket(
    std::uint64_t pattern_start) const {
  std::uint64_t distance = 0;
  while (distance < 64 && distance < pattern_start &&
      history_[(pattern_start - distance - 1) & 255u] != '\n') {
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

unsigned int CausalScr2Matcher::ByteClass(std::uint8_t value) const {
  if (value == '\n' || value == '\r') return 0;
  if (value == ' ' || value == '\t') return 1;
  if (value >= '0' && value <= '9') return 2;
  if (value >= 'A' && value <= 'Z') return 3;
  if (value >= 'a' && value <= 'z') return 4;
  if (value >= 0x80) return 5;
  if (std::strchr("<>[]{}|=:/#*&;", value) != nullptr) return 6;
  return 7;
}

std::uint16_t CausalScr2Matcher::PpmdContext(float byte_probability,
    unsigned int effective_order, unsigned int escape_depth,
    float escape_rate) const {
  const unsigned int probability_bucket = byte_probability < 0.01f ? 0u :
      byte_probability < 0.03f ? 1u : byte_probability < 0.08f ? 2u :
      byte_probability < 0.16f ? 3u : byte_probability < 0.32f ? 4u :
      byte_probability < 0.55f ? 5u : byte_probability < 0.80f ? 6u : 7u;
  const unsigned int order_band = effective_order <= 3u ? 0u :
      effective_order <= 8u ? 1u : effective_order <= 16u ? 2u : 3u;
  const unsigned int escape_band =
      escape_depth != 0u || escape_rate >= 0.25f ? 1u : 0u;
  return static_cast<std::uint16_t>(
      probability_bucket | (order_band << 3) | (escape_band << 5));
}

bool CausalScr2Matcher::FindCandidate(Candidate* candidate) const {
  if (!candidate || bytes_seen_ == 0) return false;
  const std::uint8_t bucket = history_[(bytes_seen_ - 1) & 255u];
  std::uint16_t best = 0;
  std::uint16_t best_suffix = 0;
  for (std::int16_t current = bucket_head_[bucket]; current >= 0;
       current = bucket_next_[current]) {
    const std::uint16_t code = static_cast<std::uint16_t>(current);
    if ((pattern_mask_[(code - 1) >> 3] &
        (1u << ((code - 1) & 7u))) == 0) continue;
    const std::uint16_t prefix = prefix_length_[code];
    const std::uint16_t suffix = scr2::kTokens[code].length - prefix;
    if (suffix > best_suffix && PrefixMatches(code, prefix)) {
      best = code;
      best_suffix = suffix;
    }
  }
  if (best == 0) return false;
  candidate->pattern = best;
  candidate->prefix_length = prefix_length_[best];
  candidate->suffix_length = best_suffix;
  const std::uint64_t pattern_start = bytes_seen_ - prefix_length_[best];
  const unsigned int band = stream_size_ == 0 ? 0 :
      static_cast<unsigned int>(std::min<std::uint64_t>(
          63u, (bytes_seen_ * 64u) / stream_size_));
  candidate->context = static_cast<std::uint16_t>(
      band * 8u + LineBucket(pattern_start));
  const std::uint8_t previous = pattern_start == 0 ? 0 :
      history_[(pattern_start - 1) & 255u];
  candidate->class_context = static_cast<std::uint16_t>(
      band * 8u + ByteClass(previous));
  return true;
}

unsigned int CausalScr2Matcher::Probability(
    std::uint16_t pattern, std::uint16_t context,
    std::uint16_t class_context, std::uint16_t ppmd_context) const {
  const Counts& counts = counts_[pattern][context];
  const Counts& line_counts = line_counts_[pattern][context & 7u];
  const Counts& class_counts = class_counts_[pattern][class_context];
  const Counts& byte_class_counts =
      byte_class_counts_[pattern][class_context & 7u];
  const Counts& ppmd_counts = ppmd_counts_[pattern][ppmd_context & 63u];
  const unsigned int total = counts.no + counts.yes;
  const unsigned int line_total = line_counts.no + line_counts.yes;
  const unsigned int class_total = class_counts.no + class_counts.yes;
  const unsigned int byte_class_total =
      byte_class_counts.no + byte_class_counts.yes;
  const unsigned int ppmd_total = ppmd_counts.no + ppmd_counts.yes;
  const unsigned int line_probability =
      (static_cast<unsigned int>(line_counts.yes) * 65536u) / line_total;
  const unsigned int byte_class_probability =
      (static_cast<unsigned int>(byte_class_counts.yes) * 65536u) /
      byte_class_total;
  const unsigned int ppmd_probability =
      (static_cast<unsigned int>(ppmd_counts.yes) * 65536u) / ppmd_total;
  constexpr unsigned int kLineBackoffStrength = 8u;
  const unsigned int backed_line_probability = static_cast<unsigned int>(
      (static_cast<std::uint64_t>(counts.yes) * 65536u +
          kLineBackoffStrength * line_probability) /
      (total + kLineBackoffStrength));
  const unsigned int backed_class_probability = static_cast<unsigned int>(
      (static_cast<std::uint64_t>(class_counts.yes) * 65536u +
          kLineBackoffStrength * byte_class_probability) /
      (class_total + kLineBackoffStrength));
  const unsigned int structural_probability =
      (backed_line_probability + backed_class_probability) >> 1;
  unsigned int probability =
      (3u * structural_probability + ppmd_probability) >> 2;
  if (probability < 1u) probability = 1u;
  if (probability > 65534u) probability = 65534u;
  return probability;
}

void CausalScr2Matcher::Update(
    std::uint16_t pattern, std::uint16_t context,
    std::uint16_t class_context, std::uint16_t ppmd_context, bool take) {
  Counts& counts = counts_[pattern][context];
  Counts& line_counts = line_counts_[pattern][context & 7u];
  Counts& class_counts = class_counts_[pattern][class_context];
  Counts& byte_class_counts =
      byte_class_counts_[pattern][class_context & 7u];
  Counts& ppmd_counts = ppmd_counts_[pattern][ppmd_context & 63u];
  const auto update_counts = [take](Counts* item) {
    if (take) {
      if (item->yes != 65535u) ++item->yes;
    } else if (item->no != 65535u) {
      ++item->no;
    }
    if (static_cast<unsigned int>(item->no) + item->yes > 4096u) {
      item->no = static_cast<std::uint16_t>((item->no + 1u) >> 1);
      item->yes = static_cast<std::uint16_t>((item->yes + 1u) >> 1);
      if (item->no == 0) item->no = 1;
      if (item->yes == 0) item->yes = 1;
    }
  };
  update_counts(&counts);
  update_counts(&line_counts);
  update_counts(&class_counts);
  update_counts(&byte_class_counts);
  update_counts(&ppmd_counts);
}

const std::uint8_t* CausalScr2Matcher::PatternData(
    std::uint16_t pattern) const {
  return scr2::PatternData(pattern);
}
