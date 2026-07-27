#include "donor_plan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "predictor.h"

namespace {

uint16_t ReadU16(std::istream* input) {
  const uint16_t lo = static_cast<uint8_t>(input->get());
  const uint16_t hi = static_cast<uint8_t>(input->get());
  return lo | (hi << 8);
}

uint32_t ReadU24(std::istream* input) {
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 24; shift += 8) {
    value |= static_cast<uint32_t>(
        static_cast<uint8_t>(input->get())) << shift;
  }
  return value;
}

uint32_t ReadU32(std::istream* input) {
  uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<uint32_t>(
        static_cast<uint8_t>(input->get())) << shift;
  }
  return value;
}

uint64_t ReadU64(std::istream* input) {
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<uint64_t>(
        static_cast<uint8_t>(input->get())) << shift;
  }
  return value;
}

void WriteU16(std::ostream* output, uint16_t value) {
  output->put(static_cast<char>(value));
  output->put(static_cast<char>(value >> 8));
}

void WriteU24(std::ostream* output, uint32_t value) {
  output->put(static_cast<char>(value));
  output->put(static_cast<char>(value >> 8));
  output->put(static_cast<char>(value >> 16));
}

void Replay(const std::vector<uint8_t>& bytes, Predictor* predictor) {
  for (uint8_t byte : bytes) {
    for (int bit = 7; bit >= 0; --bit) {
      predictor->Predict();
      predictor->Perceive((byte >> bit) & 1);
    }
  }
}

}  // namespace

bool DonorPlan::LoadExternal(const char* path, uint64_t stream_size) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open()) return false;

  std::array<char, 4> magic{};
  input.read(magic.data(), magic.size());
  const uint16_t version = ReadU16(&input);
  const uint16_t flags = ReadU16(&input);
  const uint32_t chunk_size = ReadU32(&input);
  const uint32_t seed_size = ReadU32(&input);
  const uint64_t planned_size = ReadU64(&input);
  const uint16_t count = ReadU16(&input);
  std::array<char, 32> digest{};
  input.read(digest.data(), digest.size());
  if (!input || std::memcmp(magic.data(), "F4CP", 4) != 0 ||
      version != 1 || flags != 0 || chunk_size != kChunkSize ||
      seed_size != kSeedSize || planned_size != stream_size) {
    return false;
  }

  assignments_.clear();
  assignments_.reserve(count);
  for (uint16_t index = 0; index < count; ++index) {
    Assignment assignment{ReadU16(&input), ReadU32(&input)};
    if (!input) return false;
    assignments_.push_back(assignment);
  }
  return Initialize(stream_size);
}

bool DonorPlan::ReadArchive(std::ifstream* input, uint64_t stream_size) {
  const uint16_t count = ReadU16(input);
  if (!*input) return false;
  assignments_.clear();
  assignments_.reserve(count);
  for (uint16_t index = 0; index < count; ++index) {
    const uint16_t recipient = ReadU16(input);
    const uint32_t shifted_offset = ReadU24(input);
    if (!*input) return false;
    assignments_.push_back({recipient, shifted_offset << 8});
  }
  return Initialize(stream_size);
}

bool DonorPlan::WriteArchive(std::ofstream* output) const {
  if (assignments_.size() > 0xffff) return false;
  WriteU16(output, static_cast<uint16_t>(assignments_.size()));
  for (const Assignment& assignment : assignments_) {
    if ((assignment.donor_offset & 255u) != 0 ||
        (assignment.donor_offset >> 8) >= (1u << 24)) {
      return false;
    }
    WriteU16(output, assignment.recipient);
    WriteU24(output, assignment.donor_offset >> 8);
  }
  return output->good();
}

bool DonorPlan::Initialize(uint64_t stream_size) {
  const uint64_t complete_regions = stream_size / kChunkSize;
  if (complete_regions > 0xffff) return false;

  std::sort(assignments_.begin(), assignments_.end(),
      [](const Assignment& left, const Assignment& right) {
        return left.recipient < right.recipient;
      });
  donor_by_region_.assign(
      static_cast<size_t>(complete_regions), kNoDonor);
  uint16_t previous = 0xffff;
  std::vector<uint32_t> offsets;
  offsets.reserve(assignments_.size());
  for (const Assignment& assignment : assignments_) {
    const uint64_t recipient_offset =
        static_cast<uint64_t>(assignment.recipient) * kChunkSize;
    if (assignment.recipient >= complete_regions ||
        assignment.recipient == previous ||
        static_cast<uint64_t>(assignment.donor_offset) + kSeedSize >
            recipient_offset) {
      return false;
    }
    previous = assignment.recipient;
    donor_by_region_[assignment.recipient] = assignment.donor_offset;
    offsets.push_back(assignment.donor_offset);
  }

  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
  seeds_.clear();
  seeds_.reserve(offsets.size());
  for (uint32_t offset : offsets) {
    Seed seed;
    seed.offset = offset;
    seed.bytes.resize(kSeedSize);
    seeds_.push_back(std::move(seed));
  }
  active_seeds_.clear();
  next_seed_ = 0;
  return true;
}

uint32_t DonorPlan::DonorOffset(uint32_t region) const {
  return region < donor_by_region_.size()
      ? donor_by_region_[region]
      : kNoDonor;
}

bool DonorPlan::ReplayAt(uint64_t position, Predictor* predictor) {
  if (position % kChunkSize != 0) return true;
  const uint32_t donor_offset =
      DonorOffset(static_cast<uint32_t>(position / kChunkSize));
  if (donor_offset == kNoDonor) return true;
  const auto found = std::lower_bound(
      seeds_.begin(), seeds_.end(), donor_offset,
      [](const Seed& seed, uint32_t offset) { return seed.offset < offset; });
  if (found == seeds_.end() || found->offset != donor_offset ||
      found->filled != kSeedSize) {
    return false;
  }
  Replay(found->bytes, predictor);
  return true;
}

void DonorPlan::CaptureByte(uint64_t position, uint8_t byte) {
  while (next_seed_ < seeds_.size() &&
      seeds_[next_seed_].offset <= position) {
    if (position < static_cast<uint64_t>(seeds_[next_seed_].offset) +
        kSeedSize) {
      active_seeds_.push_back(next_seed_);
    }
    ++next_seed_;
  }

  size_t output = 0;
  for (size_t seed_index : active_seeds_) {
    Seed& seed = seeds_[seed_index];
    const uint64_t end = static_cast<uint64_t>(seed.offset) + kSeedSize;
    if (position < end) {
      seed.bytes[static_cast<size_t>(position - seed.offset)] = byte;
      ++seed.filled;
      active_seeds_[output++] = seed_index;
    }
  }
  active_seeds_.resize(output);
}
