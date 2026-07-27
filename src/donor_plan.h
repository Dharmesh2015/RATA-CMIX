#ifndef FX4_DONOR_PLAN_H
#define FX4_DONOR_PLAN_H

#include <cstdint>
#include <fstream>
#include <vector>

class Predictor;

class DonorPlan {
 public:
  static constexpr uint32_t kChunkSize = 1u << 20;
  static constexpr uint32_t kSeedSize = 4096;
  static constexpr uint32_t kNoDonor = 0xffffffffu;

  bool LoadExternal(const char* path, uint64_t stream_size);
  bool ReadArchive(std::ifstream* input, uint64_t stream_size);
  bool WriteArchive(std::ofstream* output) const;

  bool empty() const { return assignments_.empty(); }
  uint32_t DonorOffset(uint32_t region) const;
  bool ReplayAt(uint64_t position, Predictor* predictor);
  void CaptureByte(uint64_t position, uint8_t byte);

 private:
  struct Assignment {
    uint16_t recipient;
    uint32_t donor_offset;
  };

  struct Seed {
    uint32_t offset;
    uint32_t filled = 0;
    std::vector<uint8_t> bytes;
  };

  bool Initialize(uint64_t stream_size);

  std::vector<Assignment> assignments_;
  std::vector<uint32_t> donor_by_region_;
  std::vector<Seed> seeds_;
  std::vector<size_t> active_seeds_;
  size_t next_seed_ = 0;
};

#endif
