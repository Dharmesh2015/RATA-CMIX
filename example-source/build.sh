#!/bin/sh
set -eu

cc \
  -std=c11 -O2 -pipe \
  -Wall -Wextra -Werror \
  -march=x86-64 -mtune=generic \
  -ffunction-sections -fdata-sections -fno-ident \
  -static -Wl,--gc-sections -Wl,--build-id=none -s \
  /entry/example-codec.c -lzstd \
  -o comp9
chmod 0555 comp9
