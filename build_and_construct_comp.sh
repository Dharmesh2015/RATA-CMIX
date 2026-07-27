#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

CC_BIN=clang++-17
UPX_BIN="$ROOT_DIR/tools/upx"
DICTIONARY="$ROOT_DIR/dictionary/english.dic"
ARTICLE_ORDER="$ROOT_DIR/src/readalike_prepr/data/new_article_order"
SEED="$(printenv SEED 2>/dev/null || printf 923)"
UPDATE_LIMIT="$(printenv UPDATE_LIMIT 2>/dev/null || printf 3000)"
DONOR="$(printenv DONOR 2>/dev/null || printf 0)"
DONOR_DISCOVERY="$(printenv DONOR_DISCOVERY 2>/dev/null || printf 0)"
if [[ "$DONOR_DISCOVERY" == 1 ]]; then
  DONOR=1
fi
CFLAGS_DEFINES="-DSEED=$SEED -DUPDATE_LIMIT=$UPDATE_LIMIT -DNDEBUG -DFX4_LSTM_MID_BRIDGE=2"
if [[ "$DONOR" == 1 ]]; then
  CFLAGS_DEFINES="$CFLAGS_DEFINES -DFX4_DONOR_PLAN=1"
fi
if [[ "$DONOR_DISCOVERY" == 1 ]]; then
  CFLAGS_DEFINES="$CFLAGS_DEFINES -DFX4_DONOR_FORK_DISCOVERY=1"
fi

command -v "$CC_BIN" >/dev/null
command -v llvm-strip-17 >/dev/null
test -x "$UPX_BIN"
test -s "$DICTIONARY"
test -s "$ARTICLE_ORDER"

echo "Building accepted ratio release..."
rm -f ppm.temp
make clean
make DONOR="$DONOR" DONOR_DISCOVERY="$DONOR_DISCOVERY" CFLAGS_DEFINES="$CFLAGS_DEFINES" cmix -j3

test -x cmix
llvm-strip-17 --strip-all cmix
if command -v objcopy >/dev/null 2>&1; then
  objcopy --remove-section=.comment \
    --remove-section=.note.gnu.property \
    --remove-section=.note.gnu.build-id \
    --remove-section=.note.ABI-tag cmix 2>/dev/null || true
fi
"$UPX_BIN" --ultra-brute cmix
"$UPX_BIN" -t cmix

RUN_DIR="$ROOT_DIR/run"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp cmix "$RUN_DIR/cmix_orig"
chmod 0755 "$RUN_DIR/cmix_orig"

cd "$RUN_DIR"
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

DICT_SIZE="$(stat -c%s comp_dict)"
ORDER_SIZE="$(stat -c%s comp_order)"
./cmix_orig -h "$DICT_SIZE" "$ORDER_SIZE" 0
cat cmix_orig comp_dict comp_order header.dat >cmix
chmod 0755 cmix

CORE_SIZE="$(stat -c%s cmix_orig)"
S1_SIZE="$(stat -c%s cmix)"
CORE_HASH="$(sha256sum cmix_orig | awk '{print $1}')"
S1_HASH="$(sha256sum cmix | awk '{print $1}')"

echo
echo "Build mode:            accepted"
echo "Packed core:           $CORE_SIZE bytes"
echo "Embedded dictionary:   $DICT_SIZE bytes"
echo "Embedded article order:$ORDER_SIZE bytes"
echo "Hutter S1:             $S1_SIZE bytes"
echo "Core SHA-256:          $CORE_HASH"
echo "S1 SHA-256:            $S1_HASH"
