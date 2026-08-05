#ifndef SCR2_TRANSFORM_H
#define SCR2_TRANSFORM_H

#include <cstdint>
#include <string>

namespace scr2 {

constexpr std::uint8_t kArchiveVersion = 1;

bool EncodeFile(const std::string& path, std::uint64_t* input_size,
    std::uint64_t* output_size);
bool DecodeFile(const std::string& path, std::uint64_t* output_size);

}  // namespace scr2

#endif
