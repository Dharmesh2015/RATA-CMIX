#include "altxs_transform.h"

#include "third_party/altxs/m3_densify.h"
#include "third_party/altxs/m5_payload_sim.h"
#include "third_party/altxs/product_seal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <sys/stat.h>
#include <vector>

namespace altxs {
namespace {

bool FileSize(const std::string& path, std::uint64_t* size) {
  struct stat info {};
  if (stat(path.c_str(), &info) != 0 || info.st_size < 0) return false;
  *size = static_cast<std::uint64_t>(info.st_size);
  return true;
}

bool ReadAll(const std::string& path, std::vector<std::uint8_t>* bytes) {
  std::uint64_t size = 0;
  if (!FileSize(path, &size) || size > static_cast<std::uint64_t>(SIZE_MAX)) {
    return false;
  }
  FILE* input = std::fopen(path.c_str(), "rb");
  if (!input) return false;
  bytes->resize(static_cast<std::size_t>(size));
  const bool ok = bytes->empty() ||
      std::fread(bytes->data(), 1, bytes->size(), input) == bytes->size();
  std::fclose(input);
  return ok;
}

bool WriteAll(const std::string& path, const std::uint8_t* bytes,
    std::size_t size) {
  FILE* output = std::fopen(path.c_str(), "wb");
  if (!output) return false;
  const bool ok = size == 0 || std::fwrite(bytes, 1, size, output) == size;
  const bool closed = std::fclose(output) == 0;
  return ok && closed;
}

bool ReplaceFile(const std::string& replacement, const std::string& target) {
  const std::string previous = target + ".altxs.previous";
  std::remove(previous.c_str());
  if (std::rename(target.c_str(), previous.c_str()) != 0) return false;
  if (std::rename(replacement.c_str(), target.c_str()) != 0) {
    std::rename(previous.c_str(), target.c_str());
    return false;
  }
  std::remove(previous.c_str());
  return true;
}

bool MoveGeneratedFile(const std::string& source, const std::string& target) {
  if (source == target) return true;
  std::remove(target.c_str());
  return std::rename(source.c_str(), target.c_str()) == 0;
}

bool CopyRange(FILE* input, std::uint64_t offset, std::uint64_t length,
    const std::string& output_path) {
  if (offset > static_cast<std::uint64_t>(LONG_MAX) ||
      std::fseek(input, static_cast<long>(offset), SEEK_SET) != 0) {
    return false;
  }
  FILE* output = std::fopen(output_path.c_str(), "wb");
  if (!output) return false;
  std::vector<std::uint8_t> buffer(1u << 20);
  std::uint64_t remaining = length;
  bool ok = true;
  while (remaining != 0) {
    const std::size_t want = static_cast<std::size_t>(
        remaining < buffer.size() ? remaining : buffer.size());
    if (std::fread(buffer.data(), 1, want, input) != want ||
        std::fwrite(buffer.data(), 1, want, output) != want) {
      ok = false;
      break;
    }
    remaining -= want;
  }
  if (std::fclose(output) != 0) ok = false;
  return ok;
}

}  // namespace

bool DensifyPhda9File(const std::string& path,
    const std::string& side_path, std::uint64_t* before_bytes,
    std::uint64_t* after_bytes, std::uint64_t* side_bytes) {
  if (!FileSize(path, before_bytes) || m3_densify_file(path.c_str()) != 0) {
    return false;
  }
  const std::string dense_path = path + ".dense";
  const std::string generated_side = dense_path + ".side";
  if (!FileSize(dense_path, after_bytes) ||
      !FileSize(generated_side, side_bytes) ||
      !MoveGeneratedFile(generated_side, side_path) ||
      !ReplaceFile(dense_path, path)) {
    return false;
  }
  return true;
}

bool EncodeWrtProductFile(const std::string& path,
    const std::string& m3_side_path, const ProductMeta& meta,
    std::uint64_t* before_bytes, std::uint64_t* after_bytes) {
  if (!FileSize(path, before_bytes) || m5_payload_sim_file(path.c_str()) != 0) {
    return false;
  }
  BlsmcProductMeta sealed {};
  sealed.version = BLSMC_META_VERSION;
  sealed.intro_bytes = meta.intro_bytes;
  sealed.coda_bytes = meta.coda_bytes;
  sealed.main_bytes = meta.main_bytes;
  if (product_seal_append(path.c_str(), m3_side_path.c_str(), &sealed) != 0) {
    return false;
  }
  return FileSize(path, after_bytes);
}

bool DecodeWrtProductFile(const std::string& path,
    const std::string& m3_side_path, ProductMeta* meta,
    std::uint64_t* restored_bytes) {
  const std::string m5_path = path + ".altxs.m5";
  BlsmcProductMeta sealed {};
  if (product_seal_split_file(path.c_str(), m5_path.c_str(),
      m3_side_path.c_str(), &sealed) != 1 ||
      sealed.version != BLSMC_META_VERSION) {
    return false;
  }

  std::vector<std::uint8_t> encoded;
  if (!ReadAll(m5_path, &encoded)) return false;
  std::uint8_t* restored = nullptr;
  std::size_t restored_size = 0;
  if (m5_payload_sim_restore(encoded.data(), encoded.size(), &restored,
      &restored_size) != 0) {
    return false;
  }
  const std::string restored_path = path + ".altxs.restored";
  const bool wrote = WriteAll(restored_path, restored, restored_size);
  std::free(restored);
  if (!wrote || !ReplaceFile(restored_path, path)) return false;
  std::remove(m5_path.c_str());

  meta->intro_bytes = sealed.intro_bytes;
  meta->coda_bytes = sealed.coda_bytes;
  meta->main_bytes = sealed.main_bytes;
  *restored_bytes = restored_size;
  return true;
}

bool SplitDenseReadyFile(const std::string& ready_path,
    const ProductMeta& meta, const std::string& dense_main_path,
    const std::string& intro_path, const std::string& coda_path) {
  std::uint64_t ready_size = 0;
  if (!FileSize(ready_path, &ready_size) ||
      meta.intro_bytes + meta.coda_bytes > ready_size) {
    return false;
  }
  const std::uint64_t main_size =
      ready_size - meta.intro_bytes - meta.coda_bytes;
  FILE* input = std::fopen(ready_path.c_str(), "rb");
  if (!input) return false;
  const bool ok = CopyRange(input, 0, main_size, dense_main_path) &&
      CopyRange(input, main_size, meta.intro_bytes, intro_path) &&
      CopyRange(input, main_size + meta.intro_bytes, meta.coda_bytes,
          coda_path);
  std::fclose(input);
  return ok;
}

bool RestorePhda9File(const std::string& dense_main_path,
    const std::string& m3_side_path, const std::string& raw_main_path) {
  return m3_undensify_file(dense_main_path.c_str(), m3_side_path.c_str(),
      raw_main_path.c_str()) == 0;
}

}  // namespace altxs
