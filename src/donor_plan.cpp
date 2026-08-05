#include "donor_plan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

#include "predictor.h"
#include "models/postr1_experts.h"

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

void WriteU32(std::ostream* output, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output->put(static_cast<char>(value >> shift));
  }
}

bool ReadVarint(std::istream* input, uint64_t* value) {
  *value = 0;
  for (unsigned shift = 0; shift < 64; shift += 7) {
    const int next = input->get();
    if (next == EOF) return false;
    const uint8_t byte = static_cast<uint8_t>(next);
    *value |= static_cast<uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return true;
  }
  return false;
}

bool WriteVarint(std::ostream* output, uint64_t value) {
  do {
    uint8_t byte = static_cast<uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    output->put(static_cast<char>(byte));
  } while (value != 0 && *output);
  return output->good();
}

}  // namespace

bool DonorPlan::ReadExpertSpans(std::istream* input, uint32_t count,
    bool archive_format) {
  expert_spans_.clear();
  expert_spans_.reserve(count);
  uint64_t previous_end = 0;
  for (uint32_t index = 0; index < count; ++index) {
    ExpertSpan span{};
    if (archive_format) {
      uint64_t gap = 0;
      uint64_t length = 0;
      uint64_t mask = 0;
      if (!ReadVarint(input, &gap) || !ReadVarint(input, &length) ||
          !ReadVarint(input, &mask) ||
          gap > UINT64_MAX - previous_end || length > UINT32_MAX ||
          mask > UINT32_MAX) {
        return false;
      }
      span.offset = previous_end + gap;
      span.length = static_cast<uint32_t>(length);
      span.expert_mask = static_cast<uint32_t>(mask);
    } else {
      span.offset = ReadU64(input);
      span.length = ReadU32(input);
      span.expert_mask = ReadU32(input);
    }
    const int stream_class = input->get();
    const int profile_id = input->get();
    if (!*input || stream_class < 0 || profile_id < 0) return false;
    span.stream_class = static_cast<uint8_t>(stream_class);
    span.profile_id = static_cast<uint8_t>(profile_id);
    expert_spans_.push_back(span);
    previous_end = span.offset + span.length;
  }
  return true;
}

bool DonorPlan::LoadExternal(const char* path, uint64_t stream_size) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  assignments_.clear();
  expert_spans_.clear();
  discovery_candidates_ = false;
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
  const bool production =
      std::memcmp(magic.data(), "F4CP", 4) == 0 && flags == 0;
  const bool discovery =
      std::memcmp(magic.data(), "F4CD", 4) == 0 && flags == 1;
  if (!input || (!production && !discovery) ||
      (version != 1 && version != 2 && version != 3 && version != 4) ||
      chunk_size != kChunkSize || planned_size != stream_size ||
      (version == 1 && seed_size != kSeedSize) ||
      (version >= 2 && seed_size > 65536u)) {
    return false;
  }

  assignments_.clear();
  expert_spans_.clear();
  assignments_.reserve(count);
  for (uint16_t index = 0; index < count; ++index) {
    Assignment assignment{
        ReadU16(&input), ReadU32(&input),
        version == 1 ? kSeedSize : ReadU16(&input),
        version >= 3
            ? static_cast<uint16_t>(static_cast<uint8_t>(input.get()))
            : index};
    if (!input) return false;
    assignments_.push_back(assignment);
  }
  if (version == 4) {
    const uint32_t span_count = ReadU32(&input);
    if (!input || span_count > (1u << 20) ||
        !ReadExpertSpans(&input, span_count, false)) {
      return false;
    }
  }
  discovery_candidates_ = discovery;
  return Initialize(stream_size, discovery || version >= 2);
}

bool DonorPlan::ReadArchive(std::ifstream* input, uint64_t stream_size) {
  const int version = input->get();
  if (version != 3 && version != 4) return false;
  const uint16_t count = ReadU16(input);
  if (!*input) return false;
  assignments_.clear();
  expert_spans_.clear();
  assignments_.reserve(count);
  for (uint16_t index = 0; index < count; ++index) {
    const uint16_t recipient = ReadU16(input);
    const uint32_t shifted_offset = ReadU24(input);
    const int length_log2 = input->get();
    const int order = input->get();
    if (!*input || length_log2 < 8 || length_log2 > 16 ||
        order < 0) {
      return false;
    }
    assignments_.push_back({
        recipient, shifted_offset << 8, 1u << length_log2,
        static_cast<uint16_t>(order)});
  }
  if (version == 4) {
    const uint32_t span_count = ReadU32(input);
    if (!*input || span_count > (1u << 20) ||
        !ReadExpertSpans(input, span_count, true)) {
      return false;
    }
  }
  discovery_candidates_ = false;
  return Initialize(stream_size, true);
}

bool DonorPlan::WriteArchive(std::ofstream* output) const {
  if (discovery_candidates_) return false;
  if (assignments_.size() > 0xffff) return false;
  output->put(4);
  WriteU16(output, static_cast<uint16_t>(assignments_.size()));
  for (const Assignment& assignment : assignments_) {
    if ((assignment.donor_offset & 255u) != 0 ||
        (assignment.donor_offset >> 8) >= (1u << 24) ||
        assignment.length < 256u || assignment.length > 65536u ||
        (assignment.length & (assignment.length - 1u)) != 0 ||
        assignment.order > 255u) {
      return false;
    }
    WriteU16(output, assignment.recipient);
    WriteU24(output, assignment.donor_offset >> 8);
    unsigned int length_log2 = 0;
    for (uint32_t value = assignment.length; value > 1; value >>= 1) {
      ++length_log2;
    }
    output->put(static_cast<char>(length_log2));
    output->put(static_cast<char>(assignment.order));
  }
  WriteU32(output, static_cast<uint32_t>(expert_spans_.size()));
  uint64_t previous_end = 0;
  for (const ExpertSpan& span : expert_spans_) {
    if (span.offset < previous_end ||
        !WriteVarint(output, span.offset - previous_end) ||
        !WriteVarint(output, span.length) ||
        !WriteVarint(output, span.expert_mask)) {
      return false;
    }
    output->put(static_cast<char>(span.stream_class));
    output->put(static_cast<char>(span.profile_id));
    previous_end = span.offset + span.length;
  }
  return output->good();
}

bool DonorPlan::Initialize(uint64_t stream_size, bool allow_multiple) {
  const uint64_t complete_regions = stream_size / kChunkSize;
  if (complete_regions > 0xffff) return false;

  std::sort(expert_spans_.begin(), expert_spans_.end(),
      [](const ExpertSpan& left, const ExpertSpan& right) {
        return left.offset < right.offset;
      });
  uint64_t previous_span_end = 0;
  portfolio_mask_ = 0;
  for (const ExpertSpan& span : expert_spans_) {
    if (span.length == 0 || span.expert_mask == 0 ||
        (span.expert_mask & ~static_cast<uint32_t>(
            PostR1Experts::kAllPredictors)) != 0 ||
        span.stream_class > static_cast<uint8_t>(
            PostR1Experts::StreamClass::kMixed) ||
        span.offset < previous_span_end || span.offset > stream_size ||
        span.length > stream_size - span.offset) {
      return false;
    }
    portfolio_mask_ |= span.expert_mask;
    previous_span_end = span.offset + span.length;
  }

  std::sort(assignments_.begin(), assignments_.end(),
      [](const Assignment& left, const Assignment& right) {
        return left.recipient != right.recipient
            ? left.recipient < right.recipient
            : (left.order != right.order
                ? left.order < right.order
                : left.donor_offset < right.donor_offset);
      });
  donor_by_region_.assign(
      static_cast<size_t>(complete_regions), kNoDonor);
  candidates_by_region_.assign(
      static_cast<size_t>(complete_regions), {});
  candidate_specs_by_region_.assign(
      static_cast<size_t>(complete_regions), {});
  assignments_by_region_.assign(
      static_cast<size_t>(complete_regions), {});
  uint16_t previous = 0xffff;
  uint16_t previous_order = 0xffff;
  uint32_t previous_offset = kNoDonor;
  std::vector<std::pair<uint32_t, uint32_t>> offsets;
  offsets.reserve(assignments_.size());
  for (size_t assignment_index = 0;
       assignment_index < assignments_.size(); ++assignment_index) {
    const Assignment& assignment = assignments_[assignment_index];
    const uint64_t recipient_offset =
        static_cast<uint64_t>(assignment.recipient) * kChunkSize;
    if (assignment.recipient >= complete_regions ||
        assignment.length < 256u || assignment.length > 65536u ||
        (assignment.length & (assignment.length - 1u)) != 0 ||
        static_cast<uint64_t>(assignment.donor_offset) +
            assignment.length > recipient_offset) {
      return false;
    }
    if (assignment.recipient == previous) {
      if (!allow_multiple || assignment.donor_offset == previous_offset ||
          assignment.order == previous_order) {
        return false;
      }
    } else {
      donor_by_region_[assignment.recipient] = assignment.donor_offset;
    }
    previous = assignment.recipient;
    previous_order = assignment.order;
    previous_offset = assignment.donor_offset;
    candidates_by_region_[assignment.recipient].push_back(
        assignment.donor_offset);
    candidate_specs_by_region_[assignment.recipient].push_back({
        assignment.donor_offset, assignment.length, assignment.order});
    assignments_by_region_[assignment.recipient].push_back(
        assignment_index);
    offsets.emplace_back(assignment.donor_offset, assignment.length);
  }

  std::sort(offsets.begin(), offsets.end());
  std::vector<std::pair<uint32_t, uint32_t>> merged_offsets;
  for (const auto& offset : offsets) {
    if (!merged_offsets.empty() &&
        merged_offsets.back().first == offset.first) {
      merged_offsets.back().second =
          std::max(merged_offsets.back().second, offset.second);
    } else {
      merged_offsets.push_back(offset);
    }
  }
  seeds_.clear();
  seeds_.reserve(merged_offsets.size());
  for (const auto& offset : merged_offsets) {
    Seed seed;
    seed.offset = offset.first;
    seed.bytes.resize(offset.second);
    seeds_.push_back(std::move(seed));
  }
  active_seeds_.clear();
  next_seed_ = 0;
  next_expert_span_ = 0;
  active_expert_span_ = static_cast<size_t>(-1);
  portfolio_enabled_ = false;
  return true;
}

bool DonorPlan::ApplyExpertSpanAt(
    uint64_t position, Predictor* predictor) {
  if (active_expert_span_ != static_cast<size_t>(-1)) {
    const ExpertSpan& active = expert_spans_[active_expert_span_];
    const uint64_t end = active.offset + active.length;
    if (position < end) return true;
    if (position != end) return false;
    predictor->SetPostR1Span(position, 0, 0, 0);
    active_expert_span_ = static_cast<size_t>(-1);
  }
  if (next_expert_span_ < expert_spans_.size()) {
    const ExpertSpan& next = expert_spans_[next_expert_span_];
    if (next.offset < position) return false;
    if (next.offset == position) {
      predictor->SetPostR1Span(next.offset, next.expert_mask,
          next.stream_class, next.profile_id);
      active_expert_span_ = next_expert_span_++;
    }
  }
  return true;
}

uint32_t DonorPlan::DonorOffset(uint32_t region) const {
  return region < donor_by_region_.size()
      ? donor_by_region_[region]
      : kNoDonor;
}

const std::vector<uint32_t>& DonorPlan::CandidateOffsets(
    uint32_t region) const {
  static const std::vector<uint32_t> empty;
  return region < candidates_by_region_.size()
      ? candidates_by_region_[region]
      : empty;
}

const std::vector<DonorPlan::CandidateSpec>& DonorPlan::CandidateSpecs(
    uint32_t region) const {
  static const std::vector<CandidateSpec> empty;
  return region < candidate_specs_by_region_.size()
      ? candidate_specs_by_region_[region]
      : empty;
}

bool DonorPlan::ReplayCandidate(
    uint32_t region, uint32_t donor_offset, Predictor* predictor) {
  if (region >= assignments_by_region_.size()) return false;
  for (size_t assignment_index : assignments_by_region_[region]) {
    const Assignment& assignment = assignments_[assignment_index];
    if (assignment.donor_offset == donor_offset) {
      return ReplaySeed(
          assignment.donor_offset, assignment.length, predictor);
    }
  }
  return false;
}

bool DonorPlan::ReplaySeed(
    uint32_t donor_offset, uint32_t length, Predictor* predictor) {
  const auto found = std::lower_bound(
      seeds_.begin(), seeds_.end(), donor_offset,
      [](const Seed& seed, uint32_t offset) { return seed.offset < offset; });
  if (found == seeds_.end() || found->offset != donor_offset ||
      found->filled < length || found->bytes.size() < length) {
    return false;
  }
  for (uint32_t index = 0; index < length; ++index) {
    const uint8_t byte = found->bytes[index];
    for (int bit = 7; bit >= 0; --bit) {
      predictor->Predict();
      predictor->Perceive((byte >> bit) & 1);
    }
  }
  return true;
}

bool DonorPlan::ReplayAt(uint64_t position, Predictor* predictor) {
  if (position == 0 && portfolio_mask_ != 0 && !portfolio_enabled_) {
    predictor->EnablePostR1Portfolio(portfolio_mask_);
    portfolio_enabled_ = true;
  }
  if (!ApplyExpertSpanAt(position, predictor)) return false;
  // F4CP expert plans use the causal observer only. Legacy replay remains
  // available for old assignment-only archives, but never mutates the main
  // predictor when a post-R1 specialist portfolio is present.
  if (portfolio_mask_ != 0) return true;
  if (position % kChunkSize != 0) return true;
  const uint64_t region = position / kChunkSize;
  if (region >= assignments_by_region_.size()) return true;
  for (size_t assignment_index : assignments_by_region_[region]) {
    const Assignment& assignment = assignments_[assignment_index];
    if (!ReplaySeed(
        assignment.donor_offset, assignment.length, predictor)) {
      return false;
    }
  }
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
    const uint64_t end = static_cast<uint64_t>(seed.offset) + seed.bytes.size();
    if (position < end) {
      seed.bytes[static_cast<size_t>(position - seed.offset)] = byte;
      ++seed.filled;
      active_seeds_[output++] = seed_index;
    }
  }
  active_seeds_.resize(output);
}
