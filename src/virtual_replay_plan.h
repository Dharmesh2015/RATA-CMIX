#ifndef VIRTUAL_REPLAY_PLAN_H
#define VIRTUAL_REPLAY_PLAN_H

#include <cstddef>
#include <cstdint>
#include <fstream>
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

  bool LoadExternal(const std::string& path, std::uint64_t stream_size);
  bool WriteArchive(std::ofstream* output) const;
  bool ReadArchive(std::ifstream* input, std::uint64_t stream_size);

  bool empty() const { return events_.empty(); }
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
};

#endif
