#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: byte_vocab FILE\n";
    return 2;
  }

  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 1;
  }

  std::array<std::uint64_t, 256> counts{};
  std::vector<unsigned char> buffer(1u << 20);
  std::uint64_t total = 0;
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize size = input.gcount();
    for (std::streamsize i = 0; i < size; ++i) ++counts[buffer[i]];
    total += static_cast<std::uint64_t>(size);
  }
  if (!input.eof()) {
    std::cerr << "read failed for " << argv[1] << "\n";
    return 1;
  }

  unsigned int vocabulary_size = 0;
  for (std::uint64_t count : counts) vocabulary_size += count != 0;
  std::cout << "file=" << argv[1] << "\n"
            << "bytes=" << total << "\n"
            << "vocabulary_size=" << vocabulary_size << "\n"
            << "present_hex=";
  for (unsigned int byte = 0; byte < counts.size(); ++byte) {
    if (counts[byte] == 0) continue;
    std::cout << (byte == 0 ? "" : " ") << std::hex << std::setw(2)
              << std::setfill('0') << byte;
  }
  std::cout << std::dec << "\nabsent_hex=";
  bool first = true;
  for (unsigned int byte = 0; byte < counts.size(); ++byte) {
    if (counts[byte] != 0) continue;
    if (!first) std::cout << ' ';
    std::cout << std::hex << std::setw(2) << std::setfill('0') << byte;
    first = false;
  }
  std::cout << std::dec << '\n';
  return 0;
}
