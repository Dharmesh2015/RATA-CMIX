#include "../src/r1_reorder_transform.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: emit_r1_map POST_WRT_STREAM SIDE_OUTPUT MAP_OUTPUT\n";
    return 2;
  }
  if (setenv("FX4_R1_MAP", argv[3], 1) != 0) {
    std::cerr << "failed to set FX4_R1_MAP\n";
    return 1;
  }
  return r1_reorder::ReorderEncodedTailFile(argv[1], argv[2]) ? 0 : 1;
}

