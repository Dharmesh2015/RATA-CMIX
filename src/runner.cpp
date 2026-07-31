#include <fstream>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <malloc.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"
#include "donor_plan.h"
#include "donor_fork_discovery.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/self_extract.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/misc.h"
#include "r1_reorder_transform.h"
#include "fx4_config.h"

#include <algorithm>
#include <cstdint>

namespace {
const int kMinVocabFileSize = 10000;

bool CopyResearchStream(const std::string& source, const char* destination) {
  if (!destination || !*destination) return true;
  if (source == destination) return false;
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(destination,
      std::ios::binary | std::ios::out | std::ios::trunc);
  if (!input.is_open() || !output.is_open()) return false;
  std::vector<char> buffer(1u << 20);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) output.write(buffer.data(), count);
  }
  return input.eof() && output.good();
}

bool EnvironmentEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0;
}
}

int Help() {
  printf("cmix-lex\n");
  printf("Compress:\n");
  printf("    to compress enwik9: cmix -e enwik9 [output]\n");
  printf("    to create a header for hutter prize: cmix -h comp_dict_size comp_new_order_size decomp_input_size\n");
  printf("    with dictionary:    cmix -c [dictionary] [input] [output]\n");
    printf("    without dictionary: cmix -c [input] [output]\n");
    printf("    no preprocessing:   cmix -n [input] [output]\n");
    printf("    only preprocessing: cmix -s [dictionary] [input] [output]\n");
    printf("                        cmix -s [input] [output]\n");
  printf("Decompress:\n");
    printf("    with dictionary:    cmix -d [dictionary] [input] [output]\n");
    printf("    without dictionary: cmix -d [input] [output]\n");
  return -1;
}

size_t getFileSize(const std::string& path) {
  // // get the size of the output file
  FILE *f = fopen(path.c_str(), "rb");
  if (f == NULL) {
    printf("can't open file for measuring its size");
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteHeader(unsigned long long length, const std::vector<bool>& vocab,
    bool dictionary_used, bool donor_plan_used, std::ofstream* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x3F;
      if (dictionary_used) c |= 0x80;
      if (donor_plan_used) c |= 0x40;
    }
    os->put(c);
  }
  if (length < kMinVocabFileSize) return;
  for (int i = 0; i < 32; ++i) {
    unsigned char c = 0;
    for (int j = 0; j < 8; ++j) {
      if (vocab[i * 8 + j]) c += 1<<j;
    }
    os->put(c);
  }
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

void ReadHeader(std::ifstream* is, unsigned long long* length,
    bool* dictionary_used, bool* donor_plan_used,
    std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    if (i == 0) {
      if (c&0x80) *dictionary_used = true;
      else *dictionary_used = false;
      *donor_plan_used = (c & 0x40) != 0;
      c &= 0x3F;
    }
    *length += c;
  }
  if (*length == 0) return;
  if (*length < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
    return;
  }
  for (int i = 0; i < 32; ++i) {
    unsigned char c = is->get();
    for (int j = 0; j < 8; ++j) {
      if (c & (1<<j)) (*vocab)[i * 8 + j] = true;
    }
  }
}

void ExtractVocab(unsigned long long num_bytes, std::ifstream* is,
    std::vector<bool>* vocab) {
  for (size_t pos = 0; pos < num_bytes; ++pos) {
    unsigned char c = is->get();
    (*vocab)[c] = true;
  }
  assert(num_bytes >= 2);
  std::valarray<int> byte_map(0, 256);
  uint16_t offset = 0;
  for (int i = 0; i < 256; ++i) {
    byte_map[i] = offset;
    if ((*vocab)[i]) ++offset;
  }
}

void ClearOutput() {
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\r                     \r");
  fflush(stderr);
#endif
}
bool Compress(unsigned long long input_bytes, std::ifstream* is,
    std::ofstream* os, unsigned long long* output_bytes, Predictor* p,
    DonorPlan* donor_plan) {
  Encoder e(os, p);
#if FX4_PROGRESS_LOG
  FILE* progress = fopen("./progress.log", "w");
#endif
  const unsigned long long progress_step = 1 + (input_bytes / FX4_PROGRESS_STEPS);
  unsigned long long next_progress = 0;
  ClearOutput();
  std::vector<char> buffer(FX4_IO_BUFFER_BYTES);
  unsigned long long pos = 0;
#if FX4_DONOR_PLAN
  const char* stats_path = std::getenv("FX4_REGION_STATS");
  std::ofstream stats;
  if (stats_path && *stats_path) {
    stats.open(stats_path, std::ios::out | std::ios::trunc);
    if (!stats.is_open()) return false;
    stats << "region,donor_seed_offset,cumulative_before,cumulative_after,"
             "payload_bytes\n";
  }
  size_t region_start = e.OutputSize();
  uint32_t current_region = 0;
  const char* stop_text = std::getenv("FX4_DONOR_STOP_AFTER_REGIONS");
  const uint32_t stop_after_regions = stop_text && *stop_text
      ? static_cast<uint32_t>(std::strtoul(stop_text, nullptr, 10))
      : 0;
#endif
  while (pos < input_bytes) {
    const size_t chunk = static_cast<size_t>(
        std::min<unsigned long long>(buffer.size(), input_bytes - pos));
    is->read(buffer.data(), chunk);
    const size_t got = static_cast<size_t>(is->gcount());
    if (got == 0) break;
    for (size_t i = 0; i < got; ++i, ++pos) {
#if FX4_DONOR_PLAN
      if (pos != 0 && pos % DonorPlan::kChunkSize == 0) {
        const size_t region_end = e.OutputSize();
        if (stats.is_open()) {
          stats << current_region << ','
                << (donor_plan
                    ? donor_plan->DonorOffset(current_region)
                    : DonorPlan::kNoDonor)
                << ',' << region_start << ',' << region_end << ','
                << (region_end - region_start) << '\n';
        }
        ++current_region;
        region_start = region_end;
        if (stop_after_regions != 0 &&
            current_region >= stop_after_regions) {
          if (stats.is_open()) stats.flush();
          fprintf(stderr, "FX4 donor staged run stopped after %u regions\n",
              current_region);
          return false;
        }
      }
      if (donor_plan && !donor_plan->ReplayAt(pos, p)) return false;
#endif
      unsigned char c = static_cast<unsigned char>(buffer[i]);
      for (int j = 7; j >= 0; --j) {
        e.Encode((c >> j) & 1);
      }
#if FX4_DONOR_PLAN
      if (donor_plan) donor_plan->CaptureByte(pos, c);
#endif
      if (pos >= next_progress) {
        double frac = 100.0 * pos / input_bytes;
#if FX4_STDERR_PROGRESS
        fprintf(stderr, "\rprogress: %.2f%%", frac);
        fflush(stderr);
#endif
#if FX4_PROGRESS_LOG
        if (progress) fprintf(progress, "%.2f %zu\n", frac, e.OutputSize());
#endif
        next_progress += progress_step;
      }
    }
  }
  e.Flush();
#if FX4_DONOR_PLAN
  if (stats.is_open()) {
    const size_t region_end = e.OutputSize();
    stats << current_region << ','
          << (donor_plan
              ? donor_plan->DonorOffset(current_region)
              : DonorPlan::kNoDonor)
          << ',' << region_start << ',' << region_end << ','
          << (region_end - region_start) << '\n';
  }
#endif
  *output_bytes = os->tellp();
#if FX4_PROGRESS_LOG
  if (progress) fclose(progress);
#endif
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rprogress: 100.00%%");
  fflush(stderr);
#endif
  return pos == input_bytes && os->good();
}
bool Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p, DonorPlan* donor_plan) {
  Decoder d(is, p);
  const unsigned long long progress_step = 1 + (output_length / FX4_PROGRESS_STEPS);
  unsigned long long next_progress = 0;
  std::vector<char> output;
  output.reserve(FX4_IO_BUFFER_BYTES);
  ClearOutput();
  for (unsigned long long pos = 0; pos < output_length; ++pos) {
#if FX4_DONOR_PLAN
    if (donor_plan && !donor_plan->ReplayAt(pos, p)) return false;
#endif
    int byte = 1;
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    output.push_back(static_cast<char>(byte));
#if FX4_DONOR_PLAN
    if (donor_plan) {
      donor_plan->CaptureByte(pos, static_cast<uint8_t>(byte));
    }
#endif
    if (output.size() >= FX4_IO_BUFFER_BYTES) {
      os->write(output.data(), static_cast<std::streamsize>(output.size()));
      output.clear();
    }
    if (pos >= next_progress) {
      double frac = 100.0 * pos / output_length;
#if FX4_STDERR_PROGRESS
      fprintf(stderr, "\rprogress: %.2f%%", frac);
      fflush(stderr);
#endif
      next_progress += progress_step;
    }
  }
  if (!output.empty()) {
    os->write(output.data(), static_cast<std::streamsize>(output.size()));
  }
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rprogress: 100.00%%");
  fflush(stderr);
#endif
  return os->good();
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
#if FX4_STDERR_PROGRESS
  fprintf(stderr, "\rpreprocessing...");
#endif
  fflush(stderr);
  preprocessor::Encode(data_in, data_out, *input_bytes, temp_path, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(bool enable_preprocess, const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  {
    FILE* data_in = fopen(input_path.c_str(), "rb");
    if (!data_in) return false;
    FILE* temp_out = fopen(temp_path.c_str(), "wb");
    if (!temp_out) {
      fclose(data_in);
      return false;
    }

    fseek(data_in, 0L, SEEK_END);
    *input_bytes = ftell(data_in);
    fseek(data_in, 0L, SEEK_SET);

    if (enable_preprocess) {
  #if FX4_STDERR_PROGRESS
      fprintf(stderr, "\rpreprocessing...");
#endif
      fflush(stderr);
      preprocessor::Encode(data_in, temp_out, *input_bytes, temp_path,
          dictionary);
    } else {
      preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
    }
    fclose(data_in);
    fclose(temp_out);
  }

  if (!CopyResearchStream(temp_path, std::getenv("FX4_DUMP_POST_WRT"))) {
    fprintf(stderr, "cannot dump exact post-WRT stream\n");
    return false;
  }

  if (post_wrt_side_path &&
      !r1_reorder::ReorderEncodedTailFile(temp_path, post_wrt_side_path)) {
    fprintf(stderr, "payload_lex encoded-tail reorder failed\n");
    return false;
  }

  const char* post_r1_dump_path = std::getenv("FX4_DUMP_POST_R1");
  if (!CopyResearchStream(temp_path, post_r1_dump_path)) {
    fprintf(stderr, "cannot dump exact post-R1 predictor stream\n");
    return false;
  }

  if (EnvironmentEnabled("FX4_STOP_AFTER_POST_R1")) {
    if (!post_wrt_side_path) {
      fprintf(stderr,
          "FX4_STOP_AFTER_POST_R1 is valid only for the R1-enabled -e path\n");
      return false;
    }
    if (!post_r1_dump_path || !*post_r1_dump_path) {
      fprintf(stderr,
          "FX4_STOP_AFTER_POST_R1 requires FX4_DUMP_POST_R1=<output-file>\n");
      return false;
    }
    struct stat stream_info;
    if (stat(temp_path.c_str(), &stream_info) != 0) {
      fprintf(stderr, "cannot measure exact post-R1 predictor stream\n");
      return false;
    }
    *output_bytes = static_cast<unsigned long long>(stream_info.st_size);
    fprintf(stderr, "post-R1 predictor stream dumped: %s (%llu bytes)\n",
        post_r1_dump_path, *output_bytes);
    remove(temp_path.c_str());
    return true;
  }

  std::ifstream temp_in(temp_path, std::ios::in | std::ios::binary);
  if (!temp_in.is_open()) return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  temp_in.seekg(0, std::ios::end);
  unsigned long long temp_bytes = temp_in.tellg();
  temp_in.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
  if (temp_bytes < kMinVocabFileSize) {
    std::fill(vocab.begin(), vocab.end(), true);
  } else {
    ExtractVocab(temp_bytes, &temp_in, &vocab);
    temp_in.seekg(0, std::ios::beg);
  }


  DonorPlan* active_donor_plan = nullptr;
#if FX4_DONOR_PLAN
  DonorPlan donor_plan;
  const char* donor_plan_path = std::getenv("FX4_DONOR_PLAN");
  if (donor_plan_path && *donor_plan_path) {
    if (!donor_plan.LoadExternal(donor_plan_path, temp_bytes)) {
      fprintf(stderr, "invalid FX4_DONOR_PLAN: %s\n", donor_plan_path);
      return false;
    }
    if (!donor_plan.empty()) active_donor_plan = &donor_plan;
  }
#endif

#if FX4_DONOR_FORK_DISCOVERY
  const char* discovery_results =
      std::getenv("FX4_DONOR_DISCOVERY_RESULTS");
  if (discovery_results && *discovery_results && active_donor_plan) {
    temp_in.close();
    data_out.close();
    remove(output_path.c_str());
    uint64_t discovery_output_bytes = 0;
    const bool discovery_ok = RunDonorForkDiscovery(
        temp_path, output_path, temp_bytes, vocab, dictionary,
        enable_preprocess, active_donor_plan, &discovery_output_bytes);
    *output_bytes = discovery_output_bytes;
    remove(output_path.c_str());
    remove(temp_path.c_str());
    return discovery_ok;
  }
#endif

  WriteHeader(temp_bytes, vocab, dictionary != NULL,
      active_donor_plan != nullptr, &data_out);
#if FX4_DONOR_PLAN
  if (active_donor_plan && !active_donor_plan->WriteArchive(&data_out)) {
    fprintf(stderr, "cannot write FX4 donor plan\n");
    return false;
  }
#endif
  Predictor p(vocab);
  if (enable_preprocess) preprocessor::Pretrain(&p, dictionary);
  if (!Compress(temp_bytes, &temp_in, &data_out, output_bytes, &p,
      active_donor_plan)) {
    fprintf(stderr, "FX4 entropy compression failed\n");
    return false;
  }
  temp_in.close();
  data_out.close();
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes,
    const char* post_wrt_side_path = nullptr) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  std::vector<bool> vocab(256, false);
  bool dictionary_used = false;
  bool donor_plan_used = false;
  ReadHeader(&data_in, output_bytes, &dictionary_used, &donor_plan_used,
      &vocab);
  if (!dictionary_used && dictionary != NULL) return false;
  if (dictionary_used && dictionary == NULL) return false;

  if (*output_bytes == 0) {  // undo store
    data_in.close();
    FILE* in = fopen(input_path.c_str(), "rb");
    if (!in) return false;
    FILE* data_out = fopen(output_path.c_str(), "wb");
    if (!data_out) return false;
    fseek(in, 5L, SEEK_SET);
    fprintf(stderr, "\rdecoding...");
    fflush(stderr);
    preprocessor::Decode(in, data_out, dictionary);
    fseek(data_out, 0L, SEEK_END);
    *output_bytes = ftell(data_out);
    fclose(in);
    fclose(data_out);
    return true;
  }
  {
    DonorPlan* active_donor_plan = nullptr;
#if FX4_DONOR_PLAN
    DonorPlan donor_plan;
    if (donor_plan_used) {
      if (!donor_plan.ReadArchive(&data_in, *output_bytes)) {
        fprintf(stderr, "invalid FX4 donor plan in archive\n");
        return false;
      }
      active_donor_plan = &donor_plan;
    }
#else
    if (donor_plan_used) {
      fprintf(stderr, "archive requires donor-enabled FX4 build\n");
      return false;
    }
#endif

    Predictor p(vocab);
    if (dictionary_used) preprocessor::Pretrain(&p, dictionary);

    std::ofstream temp_out(temp_path, std::ios::out | std::ios::binary);
    if (!temp_out.is_open()) return false;
    if (!Decompress(*output_bytes, &data_in, &temp_out, &p,
        active_donor_plan)) {
      fprintf(stderr, "FX4 entropy decompression failed\n");
      return false;
    }
    temp_out.close();
    p.FreeFxcmMemory();
    data_in.close();
  }
  malloc_trim(0);
  if (post_wrt_side_path) {
    if (!r1_reorder::ExtractSideFromFile(temp_path, post_wrt_side_path) ||
        !r1_reorder::RestoreEncodedTailFile(temp_path, post_wrt_side_path)) {
      fprintf(stderr, "payload_lex encoded-tail restore failed\n");
      return false;
    }
  }


  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

int main(int argc, char** argv) {
if ((argc != 1) && (argv[1][1] != 'h') && (argc < 4 || argc > 5 || strlen(argv[1]) != 2 || argv[1][0] != '-' ||
      (argv[1][1] != 'c' && argv[1][1] != 'd' && argv[1][1] != 'x' && argv[1][1] != 's' &&
      argv[1][1] != 'n' && argv[1][1] != 'e' ))) {
    return Help();
  }
   srand(SEED);

  clock_t start = clock();

  bool enable_preprocess = true;
  std::string input_path ;
  std::string output_path;
  FILE* dictionary = NULL;


  if ((argc > 1) && (argv[1][1] != 'h'))  {
    if (argv[1][1] == 'n') enable_preprocess = false;
    input_path = argv[2];
    output_path = argv[3];
    if (argc == 5) {
      if (argv[1][1] == 'n') return Help();
      dictionary = fopen(argv[2], "rb");
      if (!dictionary) return Help();
      input_path = argv[3];
      output_path = argv[4];
    }
  }

  std::string temp_path = output_path + ".cmix.temp";

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (argc == 1) {
    //Decompress enwik9
    // unpack a) header b) cmix dictionary, c) new order of articles, d) actual cmix binary
    if (selfextract_decomp() != 0) {
      return Help();
    }

    // run compression
    std::cout << "Running cmix decompression..." << std::endl;
    input_path = ".ready4cmix_decomp";
    output_path = ".input_decomp" ;
    dictionary = fopen(".dict", "rb");//_decomp

    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes, ".r1_payload_lex_side_decomp")) {
      return Help();
    }
    std::cout << "Cmix decompression finished" << std::endl;

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();

    // change the order of articles in the input
    sort();

    // merge all input parts after preprocessing
    cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d");
    cat("un1_d", ".coda_decomp", "enwik9_uncompressed");

    goto print_end_message;
  }

  if (argv[1][1] == 's') {
    if (!Store(input_path, temp_path, output_path, dictionary, &input_bytes,
        &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'c' || argv[1][1] == 'n') {
      remove(".dict");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (argv[1][1] == 'e') {
    // Compress enwik9
    input_path = argv[2];
    output_path = argv[3]; //name of a compressor output

    if (selfextract_comp() != 0) {
      return Help();
    }

    // Preparing enwik9 for reordering
    split4Comp(input_path.c_str());

    // change the order of articles in the input
    reorder();

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    cat(".main_phda9prepr", ".intro", "un1");
    cat("un1", ".coda", ".ready4cmix");

    // run compression
    input_path = ".ready4cmix";
    dictionary = fopen(".dict", "rb");
    if (!RunCompression(enable_preprocess, input_path, temp_path, output_path,
        dictionary, &input_bytes, &output_bytes, ".r1_payload_lex_side")) {
      return Help();
    }
    if (EnvironmentEnabled("FX4_STOP_AFTER_POST_R1")) return 0;
#if FX4_DONOR_FORK_DISCOVERY
    if (DonorForkDiscoveryCompleted()) return 0;
#endif

    // construct a selfextracting decompressor binary
    // archive9 = decomp_binary(upxed) + comp_dict + cmix_output + header.dat
    cat(".decomp_bin", ".dict.comp", "dec1");

    // get the size of the output file
    size_t output_size = getFileSize(output_path);

    HeaderInfo header;
    read("test.dat", header);
    header.decomp_input_size = output_size;
    write("header4archive.dat", header);

    cat("dec1", output_path.c_str(), "dec2");
    cat("dec2", "header4archive.dat", "archive9");

    // make the decompressor binary executable
    char mode[] = "0777";
    char buf[100] = "archive9";
    int i = strtol(mode, 0, 8);
    chmod(buf, i);

  } else if (argv[1][1] == 'h') {
    if (argc < 5) return Help();
    HeaderInfo header;
    header.dict_size = atoi(argv[2]);
    header.new_article_order_size = atoi(argv[3]);
    header.decomp_input_size = atoi(argv[4]);
    write("header.dat", header);
    goto exit;
  }  else if (argv[1][1] == 'x') {
    // run compression
    input_path = argv[2];
    output_path = argv[3];
    dictionary = fopen(".dict", "rb");
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
    goto print_end_message;
  }
  else {
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  }

print_end_message:
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      input_bytes, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);

exit:
  return 0;
}
