#include "virtual_replay_plan.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "fx4_config.h"

namespace {

constexpr char kExternalMagic[4] = {'F', '4', 'V', 'R'};
constexpr std::uint16_t kExternalVersion = 1;
constexpr std::uint8_t kArchiveVersion = 1;
constexpr std::uint32_t kMaximumPatterns = 65535;
constexpr std::uint32_t kMaximumPatternBytes = 65535;
constexpr std::uint32_t kMaximumEvents = 16u * 1024u * 1024u;

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
      archive_version != kArchiveVersion) {
    return false;
  }
  return ReadPayload(&input, stream_size) && input.peek() == EOF;
}

bool VirtualReplayPlan::WriteArchive(std::ofstream* output) const {
  if (!WriteU8(output, kArchiveVersion) ||
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
  if (!ReadU8(input, &version) || version != kArchiveVersion) return false;
  return ReadPayload(input, stream_size);
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
