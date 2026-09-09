// Diagnostic driver: production predictors/coders, canonical input, bounded prefix.
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
#ifndef PREFIX_SCAN_ONLY
#include "predictor.h"
#include "coder/encoder.h"
#include "preprocess/preprocessor.h"
#endif

using Clock = std::chrono::steady_clock;
static double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: prefix_bench full_stream prefix_bytes weights output\n");
    return 2;
  }
  const auto start = Clock::now();
  const std::uint64_t limit = std::strtoull(argv[2], nullptr, 10);
  if (!limit) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) return 2;
  std::vector<bool> vocab(256, false), prefix_vocab(256, false);
  std::array<char, 65536> buffer{};
  std::uint64_t size = 0;
  while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
    const auto n = input.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
      const auto c = static_cast<unsigned char>(buffer[i]);
      vocab[c] = true;
      if (size + static_cast<std::uint64_t>(i) < limit) prefix_vocab[c] = true;
    }
    size += n;
  }
  if (!input.eof() || size < limit) return 2;
  std::printf("stream_bytes=%llu prefix_bytes=%llu full_vocab=%zu prefix_vocab=%zu\n",
      static_cast<unsigned long long>(size), static_cast<unsigned long long>(limit),
      std::count(vocab.begin(), vocab.end(), true),
      std::count(prefix_vocab.begin(), prefix_vocab.end(), true));
  std::printf("full_vocab_hex=");
  for (unsigned int i = 0; i < 256; ++i) if (vocab[i]) std::printf("%02x", i);
  std::printf("\nprefix_vocab_hex=");
  for (unsigned int i = 0; i < 256; ++i) if (prefix_vocab[i]) std::printf("%02x", i);
  std::printf("\n");
  std::fflush(stdout);
#ifdef PREFIX_SCAN_ONLY
  return 0;
#else
  input.clear();
  input.seekg(0);
  if (setenv("FX4_TRANSFORMER_WEIGHTS", argv[3], 1) != 0) return 2;
  std::srand(SEED);
  const auto init_start = Clock::now();
#ifdef PREFIX_FX2
  PredictorOptions options;
  options.transformer_weights = argv[3];
  options.num_input_bytes = size;
  auto predictor = std::make_unique<Predictor>(vocab, options);
#else
  auto predictor = std::make_unique<Predictor>(vocab, true);
#endif
  const double init_seconds = Seconds(init_start);
  FILE* dictionary = std::fopen(".dict", "rb");
  if (!dictionary) return 2;
  const auto pretrain_start = Clock::now();
  preprocessor::Pretrain(predictor.get(), dictionary);
  std::fclose(dictionary);
  const double pretrain_seconds = Seconds(pretrain_start);
  std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
  if (!output) return 2;
  // Common 37-byte framing, matching both coders' normal length/vocabulary header.
  // This diagnostic container restores a prefix, not a complete enwik9 archive.
  for (int i = 4; i >= 0; --i) {
    unsigned char c = static_cast<unsigned char>(limit >> (8 * i));
    if (i == 4) c |= 0x80;
    output.put(static_cast<char>(c));
  }
  for (int i = 0; i < 32; ++i) {
    unsigned char c = 0;
    for (int j = 0; j < 8; ++j) if (vocab[i * 8 + j]) c |= 1u << j;
    output.put(static_cast<char>(c));
  }
  Encoder encoder(&output, predictor.get());
  std::ofstream progress(std::string(argv[4]) + ".progress.csv");
  progress << "input_bytes,emitted_payload_bytes,entropy_seconds\n";
  const auto entropy_start = Clock::now();

  // Checkpoint stride. 65536 by default, which is what every measurement
  // on this branch has used, so the default output is unchanged. FX2_CKPT
  // narrows it for the ranges where a curve turns: two wrong calls have
  // now been made on this branch by reading a trend from 64 KB samples
  // taken before the trend existed.
  std::uint64_t ckpt = buffer.size();
  if (const char* e = std::getenv("FX2_CKPT")) {
    const auto v = std::strtoull(e, nullptr, 10);
    if (v >= 1024 && v <= (1ull << 24)) ckpt = v;
  }

  std::uint64_t done = 0;
  std::uint64_t reported = 0;
  while (done < limit) {
    // Chunk by the stride when it is the smaller of the two, so a
    // checkpoint lands exactly on each multiple rather than at the next
    // buffer boundary after it.
    const auto n = static_cast<std::streamsize>(
        std::min<std::uint64_t>(std::min<std::uint64_t>(buffer.size(), ckpt),
                                limit - done));
    input.read(buffer.data(), n);
    if (input.gcount() != n) return 2;
    for (std::streamsize i = 0; i < n; ++i) {
      const auto c = static_cast<unsigned char>(buffer[i]);
      for (int bit = 7; bit >= 0; --bit) encoder.Encode((c >> bit) & 1);
    }
    done += n;
    // The final partial chunk always reports, which is what keeps the
    // default output byte-for-byte what it has always been: the limit is
    // 5871388, not a multiple of 65536.
    if (done - reported < ckpt && done != limit) continue;
    reported = done;
    progress << done << ',' << encoder.OutputSize() << ',' << Seconds(entropy_start) << '\n';
    progress.flush();
    std::printf("progress_bytes=%llu payload=%zu seconds=%.3f\n",
        static_cast<unsigned long long>(done), encoder.OutputSize(), Seconds(entropy_start));
    std::fflush(stdout);
  }
  encoder.Flush();
  output.flush();
  if (!output.good()) return 2;
  const auto archive_bytes = static_cast<std::uint64_t>(output.tellp());
  std::printf("RESULT input_bytes=%llu archive_bytes=%llu payload_bytes=%zu bpb=%.9f init_seconds=%.3f pretrain_seconds=%.3f entropy_seconds=%.3f total_seconds=%.3f\n",
      static_cast<unsigned long long>(limit), static_cast<unsigned long long>(archive_bytes),
      encoder.OutputSize(), archive_bytes * 8.0 / limit,
      init_seconds, pretrain_seconds, Seconds(entropy_start), Seconds(start));
  std::fflush(stdout);
  return 0;
#endif
}
