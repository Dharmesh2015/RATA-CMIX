#include "../src/postr1_transform.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr,
        "usage: postr1_transform_tool encode <file> <plan.f4tx>\n"
        "       postr1_transform_tool decode <file>\n");
    return 2;
  }
  const std::string command = argv[1];
  std::uint64_t before = 0;
  std::uint64_t after = 0;
  bool ok = false;
  if (command == "encode" && argc == 4) {
    ok = postr1::EncodeFile(argv[2], argv[3], &before, &after);
  } else if (command == "decode" && argc == 3) {
    ok = postr1::DecodeFile(argv[2], &after);
  }
  if (!ok) {
    std::fprintf(stderr, "post-R1 transform failed\n");
    return 1;
  }
  std::printf("%llu -> %llu bytes\n",
      static_cast<unsigned long long>(before),
      static_cast<unsigned long long>(after));
  return 0;
}
