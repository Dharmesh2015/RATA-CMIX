#ifndef VIRTUAL_REPLAY_PLAN_H
#define VIRTUAL_REPLAY_PLAN_H

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <array>
#include <string>
#include <vector>

// F4VR replaces selected entropy-coded byte ranges with compact archive
// events. The original bytes are still observed by every predictor, keeping
// PPMd, LSTM, FXCM, match, mixer, and donor state exactly synchronized.
class VirtualReplayPlan {
 public:
  struct Event {
    std::uint64_t offset;
    std::uint16_t pattern;
  };

  struct Range {
    std::uint64_t offset;
    std::uint64_t length;
  };

  bool LoadExternal(const std::string& path, std::uint64_t stream_size);
  bool ConfigureCausalScr2(const std::string& specification,
      std::uint64_t stream_size, unsigned int prior_code,
      const std::string& pattern_specification);
  bool WriteArchive(std::ofstream* output) const;
  bool ReadArchive(std::ifstream* input, std::uint64_t stream_size);

  bool empty() const { return !causal_scr2_ && events_.empty(); }
  bool causal_scr2() const { return causal_scr2_; }
  bool causal_all() const { return causal_all_; }
  unsigned int causal_prior_code() const { return causal_prior_code_; }
  const std::array<std::uint8_t, 16>& causal_pattern_mask() const {
    return causal_pattern_mask_;
  }
  const std::vector<Range>& causal_ranges() const { return causal_ranges_; }
  std::size_t event_count() const { return events_.size(); }
  const Event& event(std::size_t index) const { return events_[index]; }
  const std::vector<std::uint8_t>& pattern(std::uint16_t index) const {
    return patterns_[index];
  }

 private:
  bool ReadPayload(std::istream* input, std::uint64_t stream_size);
  bool Validate(std::uint64_t stream_size) const;

  std::vector<std::vector<std::uint8_t>> patterns_;
  std::vector<Event> events_;
  bool causal_scr2_ = false;
  bool causal_all_ = false;
  std::uint8_t causal_prior_code_ = 0;
  std::array<std::uint8_t, 16> causal_pattern_mask_{};
  std::vector<Range> causal_ranges_;
};

// A state-preserving SCR2 shortcut. A phrase becomes eligible only after its
// shortest unique prefix has already been decoded. One adaptive take/not-take
// bit then replaces the remaining suffix; the suffix is still observed by the
// full Predictor, so later PPMd/LSTM/FXCM state is byte-for-byte unchanged.
class CausalScr2Matcher {
 public:
  struct Candidate {
    std::uint16_t pattern = 0;
    std::uint16_t prefix_length = 0;
    std::uint16_t suffix_length = 0;
    std::uint16_t context = 0;
    std::uint16_t class_context = 0;
  };

  CausalScr2Matcher(unsigned int prior_code,
      const std::array<std::uint8_t, 16>& pattern_mask,
      std::uint64_t stream_size);
  void ResetModel();
  void ObserveByte(std::uint8_t value);
  bool FindCandidate(Candidate* candidate) const;
  unsigned int Probability(
      std::uint16_t pattern, std::uint16_t context,
      std::uint16_t class_context, std::uint16_t ppmd_context) const;
  void Update(std::uint16_t pattern, std::uint16_t context,
      std::uint16_t class_context, std::uint16_t ppmd_context, bool take);
  std::uint16_t PpmdContext(float byte_probability,
      unsigned int effective_order, unsigned int escape_depth,
      float escape_rate) const;
  const std::uint8_t* PatternData(std::uint16_t pattern) const;

 private:
  struct Counts {
    std::uint16_t no = 0;
    std::uint16_t yes = 0;
  };

  bool PrefixMatches(std::uint16_t pattern, std::uint16_t length) const;
  unsigned int ByteClass(std::uint8_t value) const;
  unsigned int LineBucket(std::uint64_t pattern_start) const;

  std::array<std::int16_t, 256> bucket_head_{};
  std::array<std::int16_t, 129> bucket_next_{};
  std::array<std::uint16_t, 129> prefix_length_{};
  std::array<std::array<Counts, 512>, 129> counts_{};
  std::array<std::array<Counts, 8>, 129> line_counts_{};
  std::array<std::array<Counts, 512>, 129> class_counts_{};
  std::array<std::array<Counts, 8>, 129> byte_class_counts_{};
  std::array<std::array<Counts, 64>, 129> ppmd_counts_{};
  std::array<std::uint8_t, 16> pattern_mask_{};
  std::array<std::uint8_t, 256> history_{};
  std::uint64_t bytes_seen_ = 0;
  std::uint64_t stream_size_ = 0;
  std::uint8_t prior_code_ = 0;
};

#endif
