#ifndef FX4_ALTXS_TRANSFORM_H
#define FX4_ALTXS_TRANSFORM_H

#include <cstdint>
#include <string>

namespace altxs {

// The transform-version byte stored behind FX4's existing transform flag.
// Versions 1 and 2 are SCR2 and the selective post-R1 portfolio.
constexpr std::uint8_t kArchiveVersion = 3;

struct ProductMeta {
  std::uint64_t intro_bytes = 0;
  std::uint64_t coda_bytes = 0;
  std::uint64_t main_bytes = 0;
};

// M3 runs on the PHDA9 main stream before WRT. The returned side file is
// sealed into the M5 product and therefore needs no external decoder asset.
bool DensifyPhda9File(const std::string& path,
    const std::string& side_path, std::uint64_t* before_bytes,
    std::uint64_t* after_bytes, std::uint64_t* side_bytes);

// M5 replaces payload_lex/R1 for this mode. It reorders complete WRT blocks
// and appends both its own inverse permutation and the M3 inverse side data.
bool EncodeWrtProductFile(const std::string& path,
    const std::string& m3_side_path, const ProductMeta& meta,
    std::uint64_t* before_bytes, std::uint64_t* after_bytes);

// Strip the product seal, restore M5 to the exact WRT stream, and retain the
// M3 side data and split sizes for the outer PHDA9 inverse.
bool DecodeWrtProductFile(const std::string& path,
    const std::string& m3_side_path, ProductMeta* meta,
    std::uint64_t* restored_bytes);

// WRT decode returns dense-PHDA9 || intro || coda. Split by the sizes sealed
// by the encoder, then restore the exact PHDA9 main stream.
bool SplitDenseReadyFile(const std::string& ready_path,
    const ProductMeta& meta, const std::string& dense_main_path,
    const std::string& intro_path, const std::string& coda_path);
bool RestorePhda9File(const std::string& dense_main_path,
    const std::string& m3_side_path, const std::string& raw_main_path);

}  // namespace altxs

#endif
