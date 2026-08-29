#ifndef FX4_POSTR1_TRANSFORM_H
#define FX4_POSTR1_TRANSFORM_H

#include <cstdint>
#include <string>
#include <vector>

// Selective, reversible transform portfolio between payload_lex/R1 and the
// entropy models. The external F4TX plan is a research artifact; the encoded
// F4PT stream is self-contained and is the only representation stored in an
// archive. Every block has a RAW fallback.
namespace postr1 {

constexpr std::uint8_t kArchiveVersion = 2;

enum Stage : std::uint16_t {
  kPmd1Permutation = 1u << 0,
  kTypedStreams = 1u << 1,
  kScr2Shorthand = 1u << 2,
  kCxWordSymbols = 1u << 3,
  kSczBytePairs = 1u << 4,
  kWctRemap = 1u << 5,
  kScrrRecordResidual = 1u << 6,
  kApproximateRlz = 1u << 7,
  kXorReference = 1u << 8,
  kWikiNative2 = 1u << 9,
  kPhraseMacros = 1u << 10,
  // Uses the immutable SCR2 table compiled into S1. Unlike
  // kScr2Shorthand, the dictionary is not repeated in every block payload.
  kScr2SharedShorthand = 1u << 11,
  kAllStages = (1u << 12) - 1u,
};

struct Operation {
  std::uint8_t type = 0;  // 1=rotate, 2=segment rotate
  std::uint32_t a = 0;
  std::uint32_t b = 0;
  std::uint32_t c = 0;
};

struct BlockSpec {
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t physical_order = 0;
  std::uint16_t stage_mask = 0;
  std::uint8_t stream_class = 10;
  std::vector<std::uint64_t> references;
  std::vector<Operation> operations;
};

// Replaces path with a self-contained F4PT stream. Plans must partition the
// complete original stream. Blocks may be physically reordered; references
// must point to blocks already emitted in physical order.
bool EncodeFile(const std::string& path, const std::string& plan_path,
    std::uint64_t* original_size, std::uint64_t* transformed_size);

// Replaces an F4PT stream with the exact original post-R1 stream.
bool DecodeFile(const std::string& path, std::uint64_t* restored_size);

// Cheap format probe used by focused round-trip tools.
bool IsEncodedFile(const std::string& path);

}  // namespace postr1

#endif
