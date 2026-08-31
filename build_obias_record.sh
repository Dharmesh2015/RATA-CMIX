#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

readonly HEAD_BLOB="$ROOT_DIR/models/bitlstm32/refit_golden256_fp16.blob"
readonly DICTIONARY="$ROOT_DIR/dictionary/english.dic"
readonly ARTICLE_ORDER="$ROOT_DIR/src/readalike_prepr/data/new_article_order"
readonly PROFILE_INPUT="${PROFILE_INPUT:-}"
readonly BUILD_SELECTIVE="${BUILD_SELECTIVE:-0}"
readonly RESIDUAL_LSTM96_BLOB="${RESIDUAL_LSTM96_BLOB:-}"

test -s "$HEAD_BLOB"
test -s "$DICTIONARY"
test -s "$ARTICLE_ORDER"
command -v clang++-17 >/dev/null
command -v llvm-strip-17 >/dev/null
export KH_BITLSTM32="$HEAD_BLOB"

make_args=(CMIX_OBIAS_RECORD=1 OUT=cmix)
if [[ -n "$RESIDUAL_LSTM96_BLOB" ]]; then
  test -s "$RESIDUAL_LSTM96_BLOB"
  export KH_RESIDUAL_LSTM96="$RESIDUAL_LSTM96_BLOB"
  make_args+=(RESIDUAL_LSTM96=1)
fi
if [[ "$BUILD_SELECTIVE" == 1 ]]; then
  make_args+=(DONOR=1 POSTR1=1 MINI_CMIX=1 VIRTUAL_REPLAY=1 \
    DONOR_DISCOVERY=1 POSTR1_TRANSFORM=1)
fi

rm -rf pgo_data
if [[ -n "$PROFILE_INPUT" ]]; then
  test -s "$PROFILE_INPUT"
  make prof_gen "${make_args[@]}"
  rm -f .pgo_sample.fx4 .pgo_sample.restored ppm.temp
  ./cmix -c "$DICTIONARY" "$PROFILE_INPUT" .pgo_sample.fx4
  rm -f ppm.temp
  ./cmix -d "$DICTIONARY" .pgo_sample.fx4 .pgo_sample.restored
  cmp -s "$PROFILE_INPUT" .pgo_sample.restored
  rm -f .pgo_sample.fx4 .pgo_sample.restored ppm.temp
  make prof_use "${make_args[@]}"
else
  make clean
  if [[ "$BUILD_SELECTIVE" == 1 ]]; then
    make selective "${make_args[@]}"
  else
    make record "${make_args[@]}"
  fi
fi

llvm-strip-17 --strip-all cmix
if command -v objcopy >/dev/null 2>&1; then
  objcopy --remove-section=.comment --remove-section=.note.gnu.property \
    --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag \
    cmix 2>/dev/null || true
fi
if command -v upx >/dev/null 2>&1; then
  upx --ultra-brute cmix
  upx -t cmix
fi

rm -rf run
mkdir -p run
cp cmix run/cmix_orig
cp "$HEAD_BLOB" run/bitlstm32.blob
if [[ -n "$RESIDUAL_LSTM96_BLOB" ]]; then
  cp "$RESIDUAL_LSTM96_BLOB" run/residual_lstm96.blob
fi
chmod 0755 run/cmix_orig

cd run
export KH_BITLSTM32="$PWD/bitlstm32.blob"
if [[ -s residual_lstm96.blob ]]; then
  export KH_RESIDUAL_LSTM96="$PWD/residual_lstm96.blob"
fi
rm -f comp_dict comp_order header.dat ppm.temp
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
./cmix_orig -h "$dict_size" "$order_size" 0
cat cmix_orig comp_dict comp_order header.dat > cmix
chmod 0755 cmix

core_size="$(stat -c%s cmix_orig)"
wrapper_size="$(stat -c%s cmix)"
blob_size="$(stat -c%s bitlstm32.blob)"
residual_lstm96_size=0
if [[ -s residual_lstm96.blob ]]; then
  residual_lstm96_size="$(stat -c%s residual_lstm96.blob)"
fi
s1_size="$((wrapper_size + blob_size + residual_lstm96_size))"
printf 'Packed core:           %s bytes\n' "$core_size"
printf 'S1 wrapper:            %s bytes\n' "$wrapper_size"
printf 'BitLSTM32 asset:       %s bytes\n' "$blob_size"
printf 'Residual LSTM96 asset: %s bytes\n' "$residual_lstm96_size"
printf 'Hutter S1 total:       %s bytes\n' "$s1_size"
printf 'S1 wrapper SHA-256:    %s\n' "$(sha256sum cmix | awk '{print $1}')"
printf 'BitLSTM32 SHA-256:     %s\n' "$(sha256sum bitlstm32.blob | awk '{print $1}')"
if [[ -s residual_lstm96.blob ]]; then
  printf 'Residual96 SHA-256:    %s\n' "$(sha256sum residual_lstm96.blob | awk '{print $1}')"
fi
