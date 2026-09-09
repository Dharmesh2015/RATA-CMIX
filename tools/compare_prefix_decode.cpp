// Decode counterpart to compare_prefix.cpp: reads the 37-byte header
// (limit + full-stream vocabulary bitmap) that prefix_bench wrote, rebuilds
// the identical Predictor, decodes exactly `limit` bytes, and diffs the
// result against the same prefix of the real stream -- a genuine round-trip
// check of the 1% harness's own archive, not just a size measurement.
//
// usage: prefix_bench_decode full_stream weights archive_in output_decoded
//
//   full_stream      the same file prefix_bench read from (used only for
//                    its size -- options.num_input_bytes must match what
//                    the encoder used -- and for the direct comparison)
//   weights          the same transformer weights path used to encode
//   archive_in       the archive prefix_bench wrote
//   output_decoded   where the decoded bytes are written
//
// Prints ROUNDTRIP=IDENTICAL or ROUNDTRIP=DIFFER (with the first differing
// offset) after decoding, so a script can grep the result without a
// separate cmp step.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "predictor.h"
#include "coder/decoder.h"
#include "preprocess/preprocessor.h"

using Clock = std::chrono::steady_clock;
static double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
        "usage: prefix_bench_decode full_stream weights archive_in output_decoded\n");
    return 2;
  }
  const auto start = Clock::now();
  const std::string stream_path = argv[1];
  const std::string weights_path = argv[2];
  const std::string archive_path = argv[3];
  const std::string output_path = argv[4];

  std::ifstream stream_size_probe(stream_path,
      std::ios::binary | std::ios::ate);
  if (!stream_size_probe) {
    std::fprintf(stderr, "cannot open full_stream: %s\n", stream_path.c_str());
    return 2;
  }
  const std::uint64_t size =
      static_cast<std::uint64_t>(stream_size_probe.tellg());
  stream_size_probe.close();

  std::ifstream archive(archive_path, std::ios::binary);
  if (!archive) {
    std::fprintf(stderr, "cannot open archive_in: %s\n", archive_path.c_str());
    return 2;
  }

  // Same 37-byte header compare_prefix.cpp wrote: 5 bytes of limit (top
  // bit of the first byte is a framing flag, cleared here), then the
  // 256-bit full-stream vocabulary as 32 bytes.
  unsigned char header[37];
  archive.read(reinterpret_cast<char*>(header), sizeof(header));
  if (archive.gcount() != static_cast<std::streamsize>(sizeof(header))) {
    std::fprintf(stderr, "archive_in is shorter than the 37-byte header\n");
    return 2;
  }
  std::uint64_t limit = static_cast<std::uint64_t>(header[0] & 0x7f);
  for (int i = 1; i < 5; ++i) {
    limit = (limit << 8) | header[i];
  }
  std::vector<bool> vocab(256, false);
  for (unsigned int i = 0; i < 32; ++i) {
    for (unsigned int j = 0; j < 8; ++j) {
      if (header[5 + i] & (1u << j)) vocab[i * 8 + j] = true;
    }
  }
  std::printf("header: limit=%llu vocab=%zu\n",
      static_cast<unsigned long long>(limit),
      std::count(vocab.begin(), vocab.end(), true));
  std::fflush(stdout);

  if (setenv("FX4_TRANSFORMER_WEIGHTS", weights_path.c_str(), 1) != 0) {
    return 2;
  }
  std::srand(SEED);
  const auto init_start = Clock::now();
#ifdef PREFIX_FX2
  PredictorOptions options;
  options.transformer_weights = weights_path;
  options.num_input_bytes = size;
  auto predictor = std::make_unique<Predictor>(vocab, options);
#else
  auto predictor = std::make_unique<Predictor>(vocab, true);
#endif
  const double init_seconds = Seconds(init_start);

  FILE* dictionary = std::fopen(".dict", "rb");
  if (!dictionary) {
    std::fprintf(stderr, "cannot open .dict in the current directory\n");
    return 2;
  }
  const auto pretrain_start = Clock::now();
  preprocessor::Pretrain(predictor.get(), dictionary);
  std::fclose(dictionary);
  const double pretrain_seconds = Seconds(pretrain_start);

  Decoder decoder(&archive, predictor.get());
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::fprintf(stderr, "cannot open output_decoded: %s\n",
        output_path.c_str());
    return 2;
  }

  std::uint64_t ckpt = 1u << 16;
  if (const char* e = std::getenv("FX2_CKPT")) {
    const auto v = std::strtoull(e, nullptr, 10);
    if (v >= 1024 && v <= (1ull << 24)) ckpt = v;
  }

  const auto entropy_start = Clock::now();
  std::vector<unsigned char> decoded;
  decoded.reserve(static_cast<size_t>(limit));
  std::uint64_t reported = 0;
  for (std::uint64_t pos = 0; pos < limit; ++pos) {
    unsigned char c = 0;
    for (int bit = 0; bit < 8; ++bit) {
      c = static_cast<unsigned char>((c << 1) | decoder.Decode());
    }
    decoded.push_back(c);
    if (pos + 1 - reported >= ckpt || pos + 1 == limit) {
      reported = pos + 1;
      std::printf("progress_bytes=%llu seconds=%.3f\n",
          static_cast<unsigned long long>(reported), Seconds(entropy_start));
      std::fflush(stdout);
    }
  }
  output.write(reinterpret_cast<const char*>(decoded.data()),
      static_cast<std::streamsize>(decoded.size()));
  output.flush();
  if (!output.good()) return 2;

  // Direct comparison against the same prefix of the real stream, so a
  // caller does not need a separate cmp step to know whether this round
  // tripped.
  std::ifstream original(stream_path, std::ios::binary);
  std::vector<unsigned char> original_prefix(static_cast<size_t>(limit));
  original.read(reinterpret_cast<char*>(original_prefix.data()),
      static_cast<std::streamsize>(limit));
  const bool read_ok =
      original.gcount() == static_cast<std::streamsize>(limit);

  bool identical = read_ok && decoded == original_prefix;
  std::uint64_t first_diff = 0;
  if (!identical && read_ok) {
    for (std::uint64_t i = 0; i < limit; ++i) {
      if (decoded[i] != original_prefix[i]) { first_diff = i; break; }
    }
  }

  std::printf(
      "RESULT decoded_bytes=%llu init_seconds=%.3f pretrain_seconds=%.3f "
      "entropy_seconds=%.3f total_seconds=%.3f\n",
      static_cast<unsigned long long>(decoded.size()), init_seconds,
      pretrain_seconds, Seconds(entropy_start), Seconds(start));
  if (!read_ok) {
    std::printf("ROUNDTRIP=ERROR could not read %llu bytes back from %s\n",
        static_cast<unsigned long long>(limit), stream_path.c_str());
  } else if (identical) {
    std::printf("ROUNDTRIP=IDENTICAL %llu bytes match the source prefix\n",
        static_cast<unsigned long long>(limit));
  } else {
    std::printf("ROUNDTRIP=DIFFER first mismatch at offset %llu\n",
        static_cast<unsigned long long>(first_diff));
  }
  std::fflush(stdout);
  return identical ? 0 : 1;
}
