#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

readonly DICTIONARY="$ROOT_DIR/dictionary/english.dic"
readonly ARTICLE_ORDER="$ROOT_DIR/src/readalike_prepr/data/new_article_order"
readonly TRANSFORMER="$ROOT_DIR/models/transformer6m/6m-q4-fp32.tfwc2"

test -s "$DICTIONARY"
test -s "$ARTICLE_ORDER"
test -s "$TRANSFORMER"
command -v clang++-17 >/dev/null

# One connected production candidate. It deliberately keeps the canonical
# cmix-lex post-R1 stream; see docs/TARGET93_CPU_PIPELINE.md.
make clean
make target93 -j"$(nproc)" OUT=cmix

if command -v llvm-strip-17 >/dev/null 2>&1; then
  llvm-strip-17 --strip-all cmix
else
  strip --strip-all cmix
fi
if command -v objcopy >/dev/null 2>&1; then
  objcopy --remove-section=.comment --remove-section=.note.gnu.property \
    --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag \
    cmix 2>/dev/null || true
fi
if command -v upx >/dev/null 2>&1; then
  upx --ultra-brute cmix >/dev/null
  upx -t cmix >/dev/null
elif [[ -x tools/upx ]]; then
  tools/upx --ultra-brute cmix >/dev/null
  tools/upx -t cmix >/dev/null
fi

rm -rf run
mkdir -p run
cp cmix run/cmix_orig
cp "$TRANSFORMER" run/transformer6m.weights
chmod 0755 run/cmix_orig

cd run
export FX4_TRANSFORMER_WEIGHTS="$PWD/transformer6m.weights"
rm -f comp_dict comp_order header.dat ppm.temp verify_dict verify_order

./cmix_orig -c "$DICTIONARY" comp_dict
rm -f ppm.temp
./cmix_orig -c "$ARTICLE_ORDER" comp_order
rm -f ppm.temp
./cmix_orig -d comp_dict verify_dict
cmp -s "$DICTIONARY" verify_dict
rm -f ppm.temp verify_dict
./cmix_orig -d comp_order verify_order
cmp -s "$ARTICLE_ORDER" verify_order
rm -f ppm.temp verify_order

dict_size="$(stat -c%s comp_dict)"
order_size="$(stat -c%s comp_order)"
transformer_size="$(stat -c%s transformer6m.weights)"
./cmix_orig -h "$dict_size" "$order_size" 0 "$transformer_size"

# S1 contains the exact frozen model needed by both compression and
# standalone decompression. No external Python, plan, or model file is used.
cat cmix_orig comp_dict comp_order transformer6m.weights header.dat > cmix
chmod 0755 cmix
cp cmix "$ROOT_DIR/cmix"

core_size="$(stat -c%s cmix_orig)"
s1_size="$(stat -c%s cmix)"
printf 'Packed core:          %s bytes\n' "$core_size"
printf 'Embedded dictionary: %s bytes\n' "$dict_size"
printf 'Embedded order:      %s bytes\n' "$order_size"
printf 'Transformer model:   %s bytes\n' "$transformer_size"
printf 'Hutter S1:           %s bytes\n' "$s1_size"
printf 'S1 SHA-256:          %s\n' "$(sha256sum cmix | awk '{print $1}')"
