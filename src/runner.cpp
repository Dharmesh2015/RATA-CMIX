#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <string>
#include <sys/stat.h>
#include <valarray>
#include <vector>

#include "coder/decoder.h"
#include "coder/encoder.h"
#include "fx4_config.h"
#include "predictor.h"
#include "preprocess/preprocessor.h"
#include "r1_reorder_transform.h"
#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/misc.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/self_extract.h"

namespace {

constexpr unsigned long long kCanonicalTransformerStreamBytes = 587138826ULL;
constexpr int kMinVocabFileSize = 10000;

int Help() {
  std::printf("fx4-cmix\n");
  std::printf("Compress:\n");
  std::printf("    enwik9/Hutter:      cmix -e enwik9 archive9\n");
  std::printf("    with dictionary:    cmix -c dictionary input output\n");
  std::printf("    without dictionary: cmix -c input output\n");
  std::printf("    no preprocessing:   cmix -n input output\n");
  std::printf("    only preprocessing: cmix -s [dictionary] input output\n");
  std::printf("Decompress:\n");
  std::printf("    with dictionary:    cmix -d dictionary input output\n");
  std::printf("    without dictionary: cmix -d input output\n");
  return -1;
}

size_t GetFileSize(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) return 0;
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::fclose(file);
  return size < 0 ? 0 : static_cast<size_t>(size);
}

bool WriteHeader(unsigned long long length, const std::vector<bool>& vocab,
    bool dictionary_used, std::ofstream* output) {
  for (int i = 4; i >= 0; --i) {
    unsigned char value =
        static_cast<unsigned char>((length >> (8 * i)) & 0xffu);
    if (i == 4) {
      // Bits 4-6 were used by discarded discovery archive variants. Keep
      // them clear so a production decoder can reject those formats.
      value &= 0x0fu;
      if (dictionary_used) value |= 0x80u;
    }
    output->put(static_cast<char>(value));
  }
  if (length < kMinVocabFileSize) return output->good();
  for (int i = 0; i < 32; ++i) {
    unsigned char value = 0;
    for (int j = 0; j < 8; ++j) {
      if (vocab[i * 8 + j]) value |= static_cast<unsigned char>(1u << j);
    }
    output->put(static_cast<char>(value));
  }
  return output->good();
}

void WriteStorageHeader(FILE* output, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    unsigned char value = 0;
    if (i == 4 && dictionary_used) value = 0x80u;
    std::putc(value, output);
  }
}

bool ReadHeader(std::ifstream* input, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  *dictionary_used = false;
  for (int i = 0; i <= 4; ++i) {
    const int next = input->get();
    if (next == EOF) return false;
    unsigned char value = static_cast<unsigned char>(next);
    *length <<= 8;
    if (i == 0) {
      *dictionary_used = (value & 0x80u) != 0;
      if ((value & 0x70u) != 0) {
        std::fprintf(stderr,
            "archive uses an unsupported discovery-only FX4 format\n");
        return false;
      }
      value &= 0x0fu;
    }
    *length += value;
  }
  if (*length == 0) return true;
  if (*length < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
    return true;
  }
  for (int i = 0; i < 32; ++i) {
    const int next = input->get();
    if (next == EOF) return false;
    const unsigned char value = static_cast<unsigned char>(next);
    for (int j = 0; j < 8; ++j) {
      if ((value & (1u << j)) != 0) (*vocab)[i * 8 + j] = true;
    }
  }
  return input->good();
}

bool ExtractVocab(unsigned long long num_bytes, std::ifstream* input,
    std::vector<bool>* vocab) {
  for (unsigned long long pos = 0; pos < num_bytes; ++pos) {
    const int value = input->get();
    if (value == EOF) return false;
    (*vocab)[static_cast<unsigned char>(value)] = true;
  }
  return input->good() || input->eof();
}

void ClearOutput() {
#if FX4_STDERR_PROGRESS
  std::fprintf(stderr, "\r                     \r");
  std::fflush(stderr);
#endif
}

bool Compress(unsigned long long input_bytes, std::ifstream* input,
    std::ofstream* output, unsigned long long* output_bytes, Predictor* p) {
  Encoder encoder(output, p);
#if FX4_PROGRESS_LOG
  FILE* progress = std::fopen("./progress.log", "w");
#endif
  const unsigned long long progress_step =
      1 + input_bytes / FX4_PROGRESS_STEPS;
  unsigned long long next_progress = 0;
  std::vector<char> buffer(FX4_IO_BUFFER_BYTES);
  unsigned long long pos = 0;
  ClearOutput();

  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer.size(), input_bytes - pos));
    input->read(buffer.data(), static_cast<std::streamsize>(chunk));
    const size_t got = static_cast<size_t>(input->gcount());
    if (got == 0) break;
    for (size_t i = 0; i < got; ++i) {
      const unsigned char value = static_cast<unsigned char>(buffer[i]);
      for (int bit = 7; bit >= 0; --bit) {
        encoder.Encode((value >> bit) & 1);
      }
      if (pos >= next_progress) {
        const double percent =
            input_bytes == 0 ? 100.0 : 100.0 * pos / input_bytes;
#if FX4_STDERR_PROGRESS
        std::fprintf(stderr, "\rprogress: %.2f%%", percent);
        std::fflush(stderr);
#endif
#if FX4_PROGRESS_LOG
        if (progress) {
          std::fprintf(progress, "%.2f %zu\n", percent,
              encoder.OutputSize());
        }
#endif
        do {
          next_progress += progress_step;
        } while (pos >= next_progress);
      }
      ++pos;
    }
  }

  encoder.Flush();
  const std::streampos end = output->tellp();
  if (end < 0) return false;
  *output_bytes = static_cast<unsigned long long>(end);
#if FX4_PROGRESS_LOG
  if (progress) std::fclose(progress);
#endif
#if FX4_STDERR_PROGRESS
  std::fprintf(stderr, "\rprogress: 100.00%%");
  std::fflush(stderr);
#endif
  return pos == input_bytes && output->good();
}

bool Decompress(unsigned long long output_length, std::ifstream* input,
    std::ofstream* output, Predictor* p) {
  Decoder decoder(input, p);
  const unsigned long long progress_step =
      1 + output_length / FX4_PROGRESS_STEPS;
  unsigned long long next_progress = 0;
  std::vector<char> buffer;
  buffer.reserve(FX4_IO_BUFFER_BYTES);
  ClearOutput();

  for (unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    while (byte < 256) byte += byte + decoder.Decode();
    buffer.push_back(static_cast<char>(byte));
    if (buffer.size() == FX4_IO_BUFFER_BYTES) {
      output->write(buffer.data(),
          static_cast<std::streamsize>(buffer.size()));
      buffer.clear();
      if (!output->good()) return false;
    }
    if (pos >= next_progress) {
      const double percent =
          output_length == 0 ? 100.0 : 100.0 * pos / output_length;
#if FX4_STDERR_PROGRESS
      std::fprintf(stderr, "\rprogress: %.2f%%", percent);
      std::fflush(stderr);
#endif
      do {
        next_progress += progress_step;
      } while (pos >= next_progress);
    }
  }
  if (!buffer.empty()) {
    output->write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  }
#if FX4_STDERR_PROGRESS
  std::fprintf(stderr, "\rprogress: 100.00%%");
  std::fflush(stderr);
#endif
  return output->good();
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* input = std::fopen(input_path.c_str(), "rb");
  if (!input) return false;
  FILE* output = std::fopen(output_path.c_str(), "wb");
  if (!output) {
    std::fclose(input);
    return false;
  }
  std::fseek(input, 0, SEEK_END);
  const long size = std::ftell(input);
  std::fseek(input, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(input);
    std::fclose(output);
    return false;
  }
  *input_bytes = static_cast<unsigned long long>(size);
  WriteStorageHeader(output, dictionary != nullptr);
#if FX4_STDERR_PROGRESS
  std::fprintf(stderr, "\rpreprocessing...");
  std::fflush(stderr);
#endif
  preprocessor::Encode(input, output, *input_bytes, temp_path, dictionary);
  std::fseek(output, 0, SEEK_END);
  const long encoded_size = std::ftell(output);
  std::fclose(input);
  std::fclose(output);
  if (encoded_size < 0) return false;
  *output_bytes = static_cast<unsigned long long>(encoded_size);
  return true;
}

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr,
    bool enable_transformer6m = false) {
  FILE* input = std::fopen(input_path.c_str(), "rb");
  if (!input) return false;
  FILE* temp_output = std::fopen(temp_path.c_str(), "wb");
  if (!temp_output) {
    std::fclose(input);
    return false;
  }

  std::fseek(input, 0, SEEK_END);
  const long size = std::ftell(input);
  std::fseek(input, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(input);
    std::fclose(temp_output);
    return false;
  }
  *input_bytes = static_cast<unsigned long long>(size);

  if (enable_preprocess) {
#if FX4_STDERR_PROGRESS
    std::fprintf(stderr, "\rpreprocessing...");
    std::fflush(stderr);
#endif
    preprocessor::Encode(input, temp_output, *input_bytes, temp_path,
        dictionary);
  } else {
    preprocessor::NoPreprocess(input, temp_output, *input_bytes);
  }
  std::fclose(input);
  std::fclose(temp_output);

  if (post_wrt_side_path &&
      !r1_reorder::ReorderEncodedTailFile(temp_path, post_wrt_side_path)) {
    std::fprintf(stderr, "payload_lex encoded-tail reorder failed\n");
    return false;
  }

  std::ifstream temp_input(temp_path, std::ios::binary);
  if (!temp_input.is_open()) return false;
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) return false;

  temp_input.seekg(0, std::ios::end);
  const std::streampos end = temp_input.tellg();
  if (end < 0) return false;
  const unsigned long long temp_bytes =
      static_cast<unsigned long long>(end);
  temp_input.seekg(0, std::ios::beg);

  if (enable_transformer6m &&
      temp_bytes != kCanonicalTransformerStreamBytes) {
    std::fprintf(stderr,
        "transformer stream mismatch: expected %llu bytes, got %llu\n",
        kCanonicalTransformerStreamBytes, temp_bytes);
    return false;
  }

  std::vector<bool> vocab(256, false);
  if (temp_bytes < kMinVocabFileSize) {
    std::fill(vocab.begin(), vocab.end(), true);
  } else {
    if (!ExtractVocab(temp_bytes, &temp_input, &vocab)) return false;
    temp_input.clear();
    temp_input.seekg(0, std::ios::beg);
  }

  if (!WriteHeader(temp_bytes, vocab, dictionary != nullptr, &output)) {
    return false;
  }
  Predictor predictor(vocab, enable_transformer6m);
  if (enable_preprocess) preprocessor::Pretrain(&predictor, dictionary);
  const bool ok = Compress(temp_bytes, &temp_input, &output, output_bytes,
      &predictor);
  predictor.FreeFxcmMemory();
  temp_input.close();
  output.close();
  std::remove(temp_path.c_str());
  if (!ok) std::remove(output_path.c_str());
  return ok;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr,
    bool enable_transformer6m = false) {
  std::ifstream input(input_path, std::ios::binary);
  if (!input.is_open()) return false;
  input.seekg(0, std::ios::end);
  const std::streampos end = input.tellg();
  if (end < 0) return false;
  *input_bytes = static_cast<unsigned long long>(end);
  input.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
  bool dictionary_used = false;
  if (!ReadHeader(&input, output_bytes, &dictionary_used, &vocab)) {
    return false;
  }
  if (dictionary_used != (dictionary != nullptr)) return false;
  if (enable_transformer6m &&
      *output_bytes != kCanonicalTransformerStreamBytes) {
    std::fprintf(stderr,
        "transformer archive stream mismatch: expected %llu bytes, got %llu\n",
        kCanonicalTransformerStreamBytes, *output_bytes);
    return false;
  }

  if (*output_bytes == 0) {
    input.close();
    FILE* stored_input = std::fopen(input_path.c_str(), "rb");
    if (!stored_input) return false;
    FILE* output = std::fopen(output_path.c_str(), "wb");
    if (!output) {
      std::fclose(stored_input);
      return false;
    }
    std::fseek(stored_input, 5, SEEK_SET);
    preprocessor::Decode(stored_input, output, dictionary);
    std::fseek(output, 0, SEEK_END);
    const long size = std::ftell(output);
    std::fclose(stored_input);
    std::fclose(output);
    if (size < 0) return false;
    *output_bytes = static_cast<unsigned long long>(size);
    return true;
  }

  {
    Predictor predictor(vocab, enable_transformer6m);
    if (dictionary_used) preprocessor::Pretrain(&predictor, dictionary);
    std::ofstream temp_output(temp_path,
        std::ios::binary | std::ios::trunc);
    if (!temp_output.is_open()) return false;
    const bool ok = Decompress(*output_bytes, &input, &temp_output,
        &predictor);
    temp_output.close();
    predictor.FreeFxcmMemory();
    input.close();
    if (!ok) {
      std::remove(temp_path.c_str());
      return false;
    }
  }
  malloc_trim(0);

  if (post_wrt_side_path) {
    if (!r1_reorder::ExtractSideFromFile(temp_path, post_wrt_side_path) ||
        !r1_reorder::RestoreEncodedTailFile(temp_path, post_wrt_side_path)) {
      std::fprintf(stderr, "payload_lex encoded-tail restore failed\n");
      return false;
    }
  }

  FILE* temp_input = std::fopen(temp_path.c_str(), "rb");
  if (!temp_input) return false;
  FILE* output = std::fopen(output_path.c_str(), "wb");
  if (!output) {
    std::fclose(temp_input);
    return false;
  }
  preprocessor::Decode(temp_input, output, dictionary);
  std::fseek(output, 0, SEEK_END);
  const long size = std::ftell(output);
  std::fclose(temp_input);
  std::fclose(output);
  std::remove(temp_path.c_str());
  if (size < 0) return false;
  *output_bytes = static_cast<unsigned long long>(size);
  return true;
}

bool ValidMode(const char* mode) {
  if (!mode || mode[0] != '-' || mode[2] != '\0') return false;
  const char command = mode[1];
  return command == 'c' || command == 'd' || command == 'e' ||
      command == 'h' || command == 'n' || command == 's' || command == 'x';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 1 && !ValidMode(argv[1])) return Help();
  if (argc > 1 && argv[1][1] == 'h') {
    if (argc != 6) return Help();
  } else if (argc != 1 && (argc < 4 || argc > 5)) {
    return Help();
  }

  std::srand(SEED);
  const clock_t start = std::clock();
  bool enable_preprocess = true;
  std::string input_path;
  std::string output_path;
  FILE* dictionary = nullptr;

  if (argc > 1 && argv[1][1] != 'h') {
    if (argv[1][1] == 'n') enable_preprocess = false;
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n' || argv[1][1] == 'e' ||
          argv[1][1] == 'x') {
        return Help();
      }
      dictionary = std::fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
  }

  std::string temp_path = output_path + ".cmix.temp";
  unsigned long long input_bytes = 0;
  unsigned long long output_bytes = 0;

  if (argc == 1) {
    if (selfextract_decomp() != 0) return Help();
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp";
    temp_path = output_path + ".cmix.temp";
    dictionary = std::fopen(".dict", "rb");
    if (!dictionary ||
        !RunDecompression(input_path, temp_path, output_path, dictionary,
            &input_bytes, &output_bytes, ".r1_payload_lex_side_decomp",
            true)) {
      return Help();
    }
    std::fclose(dictionary);
    split4Decomp();
    phda9_resto();
    sort();
    cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d");
    cat("un1_d", ".coda_decomp", "enwik9_uncompressed");
    goto print_end_message;
  }

  if (argv[1][1] == 's') {
    if (!Store(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'c' || argv[1][1] == 'n') {
    std::remove(".dict");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'e') {
    if (selfextract_comp() != 0) return Help();
    split4Comp(input_path.c_str());
    reorder();
    phda9_prepr();
    cat(".main_phda9prepr", ".intro", "un1");
    cat("un1", ".coda", ".ready4cmix");

    input_path = ".ready4cmix";
    temp_path = output_path + ".cmix.temp";
    dictionary = std::fopen(".dict", "rb");
    if (!dictionary ||
        !RunCompression(true, input_path, temp_path, output_path, dictionary,
            &input_bytes, &output_bytes, ".r1_payload_lex_side", true)) {
      return Help();
    }
    std::fclose(dictionary);
    dictionary = nullptr;

    cat(".decomp_bin", ".dict.comp", "dec1");
    HeaderInfo header = {};
    if (!read("test.dat", header)) return Help();
    header.decomp_input_size = static_cast<int>(GetFileSize(output_path));
    if (header.decomp_input_size <= 0 ||
        header.transformer6m_weights_size <= 0 ||
        static_cast<size_t>(header.transformer6m_weights_size) !=
            GetFileSize(".tfweights") ||
        !write("header4archive.dat", header)) {
      std::fprintf(stderr, "invalid packaged transformer or payload\n");
      return Help();
    }
    cat("dec1", ".tfweights", "dec1t");
    cat("dec1t", output_path.c_str(), "dec2");
    cat("dec2", "header4archive.dat", "archive9");
    if (::chmod("archive9", 0755) != 0) return Help();
  } else if (argv[1][1] == 'h') {
    HeaderInfo header = {};
    header.dict_size = std::atoi(argv[2]);
    header.new_article_order_size = std::atoi(argv[3]);
    header.decomp_input_size = std::atoi(argv[4]);
    header.transformer6m_weights_size = std::atoi(argv[5]);
    if (header.dict_size <= 0 || header.new_article_order_size <= 0 ||
        header.decomp_input_size < 0 ||
        header.transformer6m_weights_size <= 0 ||
        !write("header.dat", header)) {
      return Help();
    }
    goto exit;
  } else if (argv[1][1] == 'x') {
    dictionary = std::fopen(".dict", "rb");
    if (!dictionary ||
        !RunDecompression(input_path, temp_path, output_path, dictionary,
            &input_bytes, &output_bytes, nullptr, true)) {
      return Help();
    }
    std::fclose(dictionary);
    goto print_end_message;
  } else {
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

  if (dictionary) std::fclose(dictionary);

print_end_message:
  std::printf("\r%llu bytes -> %llu bytes in %.2f s.\n",
      input_bytes, output_bytes,
      (static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC));

exit:
  return 0;
}
