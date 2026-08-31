#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

readonly DICTIONARY="$ROOT_DIR/dictionary/english.dic"
readonly ARTICLE_ORDER="$ROOT_DIR/src/readalike_prepr/data/new_article_order"
readonly TRANSFORMER="$ROOT_DIR/models/transformer6m/6m-q4-fp32.tfwc2"
readonly BITLSTM32="$ROOT_DIR/models/bitlstm32/refit_golden256_fp16.blob"
readonly TOKEN_NGRAM="${TOKEN_NGRAM:-0}"

test -s "$DICTIONARY"
test -s "$ARTICLE_ORDER"
test -s "$TRANSFORMER"
test -s "$BITLSTM32"
command -v clang++-17 >/dev/null

make clean
make target93 -j"$(nproc)" OUT=cmix TOKEN_NGRAM="$TOKEN_NGRAM"

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
cp "$BITLSTM32" run/bitlstm32.blob
chmod 0755 run/cmix_orig

cd run
export FX4_TRANSFORMER_WEIGHTS="$PWD/transformer6m.weights"
export KH_BITLSTM32="$PWD/bitlstm32.blob"
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
bitlstm32_size="$(stat -c%s bitlstm32.blob)"
./cmix_orig -h "$dict_size" "$order_size" 0 \
  "$transformer_size" "$bitlstm32_size"

# S1 is a single self-contained executable. selfextract_comp() restores both
# model files before it decodes helper streams or constructs the main encoder.
cat cmix_orig comp_dict comp_order transformer6m.weights bitlstm32.blob \
  header.dat > cmix
chmod 0755 cmix
cp cmix "$ROOT_DIR/cmix"

core_size="$(stat -c%s cmix_orig)"
s1_size="$(stat -c%s cmix)"
printf 'Packed target93 core: %s bytes\n' "$core_size"
printf 'Embedded dictionary: %s bytes\n' "$dict_size"
printf 'Embedded order:      %s bytes\n' "$order_size"
printf 'Transformer model:   %s bytes\n' "$transformer_size"
printf 'BitLSTM32 model:     %s bytes\n' "$bitlstm32_size"
printf 'Hutter S1:           %s bytes\n' "$s1_size"
printf 'S1 SHA-256:          %s\n' "$(sha256sum cmix | awk '{print $1}')"

