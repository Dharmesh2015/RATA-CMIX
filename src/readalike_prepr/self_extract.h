#ifndef SELF_EXTRACT_H
#define SELF_EXTRACT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>

// Native little-endian x86-64 trailer shared by S1 and archive9.
struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int decomp_input_size;
  int transformer6m_weights_size;
};

inline bool write(const std::string& file_name, const HeaderInfo& data) {
  FILE* output = std::fopen(file_name.c_str(), "wb");
  if (!output) return false;
  const bool wrote = std::fwrite(&data, sizeof(data), 1, output) == 1;
  const bool closed = std::fclose(output) == 0;
  return wrote && closed;
}

inline bool read(const std::string& file_name, HeaderInfo& data) {
  FILE* input = std::fopen(file_name.c_str(), "rb");
  if (!input) return false;
  const bool loaded = std::fread(&data, sizeof(data), 1, input) == 1;
  const bool closed = std::fclose(input) == 0;
  return loaded && closed;
}

namespace self_extract_internal {

inline bool WriteSlice(const char* path, const unsigned char* data,
    size_t size) {
  FILE* output = std::fopen(path, "wb");
  if (!output) return false;
  const bool wrote =
      size == 0 || std::fwrite(data, size, 1, output) == 1;
  const bool closed = std::fclose(output) == 0;
  return wrote && closed;
}

inline unsigned char* ReadWholeFile(const char* path, size_t* size) {
  *size = 0;
  FILE* input = std::fopen(path, "rb");
  if (!input) return nullptr;
  if (std::fseek(input, 0, SEEK_END) != 0) {
    std::fclose(input);
    return nullptr;
  }
  const long measured = std::ftell(input);
  if (measured < 0 || std::fseek(input, 0, SEEK_SET) != 0) {
    std::fclose(input);
    return nullptr;
  }
  *size = static_cast<size_t>(measured);
  unsigned char* data =
      static_cast<unsigned char*>(std::malloc(*size == 0 ? 1 : *size));
  if (!data) {
    std::fclose(input);
    return nullptr;
  }
  const bool loaded =
      *size == 0 || std::fread(data, *size, 1, input) == 1;
  const bool closed = std::fclose(input) == 0;
  const bool ok = loaded && closed;
  if (!ok) {
    std::free(data);
    return nullptr;
  }
  return data;
}

inline bool ValidNonnegative(const HeaderInfo& header) {
  return header.dict_size >= 0 &&
      header.new_article_order_size >= 0 &&
      header.decomp_input_size >= 0 &&
      header.transformer6m_weights_size > 0;
}

}  // namespace self_extract_internal

// Split S1 into the executable core, dictionary, article order and frozen
// transformer. The helper streams deliberately use the classical fallback
// predictor; the transformer is enabled only for the canonical post-R1 stream.
inline int selfextract_comp() {
  size_t file_size = 0;
  unsigned char* data =
      self_extract_internal::ReadWholeFile("cmix", &file_size);
  if (!data || file_size < sizeof(HeaderInfo)) {
    std::fprintf(stderr, "selfextract failed to read ./cmix\n");
    std::free(data);
    return 1;
  }

  HeaderInfo header = {};
  std::memcpy(&header, data + file_size - sizeof(header), sizeof(header));
  if (!self_extract_internal::ValidNonnegative(header)) {
    std::fprintf(stderr, "selfextract found an invalid S1 trailer\n");
    std::free(data);
    return 1;
  }
  const size_t dictionary_size = static_cast<size_t>(header.dict_size);
  const size_t order_size =
      static_cast<size_t>(header.new_article_order_size);
  const size_t transformer_size =
      static_cast<size_t>(header.transformer6m_weights_size);
  const size_t trailer_and_assets =
      sizeof(header) + dictionary_size + order_size + transformer_size;
  if (trailer_and_assets > file_size) {
    std::fprintf(stderr, "selfextract S1 assets exceed file size\n");
    std::free(data);
    return 1;
  }
  const size_t core_size = file_size - trailer_and_assets;
  const size_t dictionary_offset = core_size;
  const size_t order_offset = dictionary_offset + dictionary_size;
  const size_t transformer_offset = order_offset + order_size;

  std::remove(".dict");
  const bool extracted =
      write("test.dat", header) &&
      self_extract_internal::WriteSlice(
          ".decomp_bin", data, core_size) &&
      self_extract_internal::WriteSlice(
          ".dict.comp", data + dictionary_offset, dictionary_size) &&
      self_extract_internal::WriteSlice(
          ".new_article_order.comp", data + order_offset, order_size) &&
      self_extract_internal::WriteSlice(
          ".tfweights", data + transformer_offset, transformer_size);
  std::free(data);
  if (!extracted) {
    std::fprintf(stderr, "selfextract failed to write S1 assets\n");
    return 1;
  }

  int status =
      std::system("./cmix -d .new_article_order.comp .new_article_order");
  if (status != 0) {
    std::fprintf(stderr,
        "selfextract failed: article order decode status=%d\n", status);
    return 1;
  }
  status = std::system("./cmix -d .dict.comp .dict");
  if (status != 0) {
    std::fprintf(stderr,
        "selfextract failed: dictionary decode status=%d\n", status);
    return 1;
  }
  malloc_trim(0);
  return 0;
}

// Split archive9 into the decoder core, dictionary, frozen transformer and
// entropy payload. archive9 itself is the only executable needed to restore
// enwik9.
inline int selfextract_decomp() {
  size_t file_size = 0;
  unsigned char* data =
      self_extract_internal::ReadWholeFile("archive9", &file_size);
  if (!data || file_size < sizeof(HeaderInfo)) {
    std::fprintf(stderr, "selfextract failed to read ./archive9\n");
    std::free(data);
    return 1;
  }

  HeaderInfo header = {};
  std::memcpy(&header, data + file_size - sizeof(header), sizeof(header));
  if (!self_extract_internal::ValidNonnegative(header)) {
    std::fprintf(stderr, "selfextract found an invalid archive trailer\n");
    std::free(data);
    return 1;
  }
  const size_t dictionary_size = static_cast<size_t>(header.dict_size);
  const size_t transformer_size =
      static_cast<size_t>(header.transformer6m_weights_size);
  const size_t payload_size = static_cast<size_t>(header.decomp_input_size);
  const size_t trailer_and_assets =
      sizeof(header) + dictionary_size + transformer_size + payload_size;
  if (trailer_and_assets > file_size) {
    std::fprintf(stderr, "selfextract archive assets exceed file size\n");
    std::free(data);
    return 1;
  }
  const size_t core_size = file_size - trailer_and_assets;
  const size_t dictionary_offset = core_size;
  const size_t transformer_offset = dictionary_offset + dictionary_size;
  const size_t payload_offset = transformer_offset + transformer_size;

  std::remove(".dict");
  const bool extracted =
      write("test.dat", header) &&
      self_extract_internal::WriteSlice(
          ".tfweights", data + transformer_offset, transformer_size) &&
      self_extract_internal::WriteSlice(
          ".dict.comp_decomp", data + dictionary_offset, dictionary_size);
  if (!extracted) {
    std::fprintf(stderr, "selfextract failed to write archive assets\n");
    std::free(data);
    return 1;
  }

  const int status =
      std::system("./archive9 -d .dict.comp_decomp .dict");
  if (status != 0) {
    std::fprintf(stderr,
        "selfextract failed: dictionary decode status=%d\n", status);
    std::free(data);
    return 1;
  }
  const bool payload_ok = self_extract_internal::WriteSlice(
      ".ready4cmix_decomp", data + payload_offset, payload_size);
  std::free(data);
  if (!payload_ok) {
    std::fprintf(stderr, "selfextract failed to write entropy payload\n");
    return 1;
  }
  malloc_trim(0);
  return 0;
}

#endif  // SELF_EXTRACT_H
