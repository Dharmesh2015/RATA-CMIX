#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly run_root=/root/fx4run
readonly input_path="$1"
readonly export_path="$2"
readonly rebuild_s1="$(printenv REBUILD_S1 2>/dev/null || printf 0)"

test -f "$input_path"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "Hutter -e requires exact 1,000,000,000-byte enwik9." >&2
  exit 2
fi

resolved="$(readlink -m "$run_root")"
if [[ "$resolved" != /root/fx4run ]]; then
  echo "Refusing to clean unexpected path: $resolved" >&2
  exit 2
fi

pkill -9 -x cmix 2>/dev/null || true
rm -rf -- "$resolved"
mkdir -p "$run_root"
cp "$input_path" "$run_root/enwik9"

if [[ "$rebuild_s1" == 1 ]]; then
  cp -a "$source_root/src" "$run_root/src"
  cp -a "$source_root/dictionary" "$run_root/dictionary"
  cp -a "$source_root/install_tools" "$run_root/install_tools"
  cp -a "$source_root/tools" "$run_root/tools"

  cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh" \
     "$source_root/LICENSE" "$run_root/"
  chmod +x "$run_root/build_and_construct_comp.sh" "$run_root/tools/upx"
  cd "$run_root"
  ./build_and_construct_comp.sh
  cp run/cmix cmix
else
  test -x "$source_root/run/cmix"
  cp "$source_root/run/cmix" "$run_root/cmix"
fi

cd "$run_root"
chmod 0755 cmix
cpu=0
if (( $(nproc) > 7 )); then cpu=7; fi
cooldown="$(printenv FX4_COOLDOWN_SECONDS 2>/dev/null || printf 120)"

echo "Hutter S1: $(stat -c%s cmix) bytes"
echo "Compression CPU: $cpu"
echo "Runtime directory: $run_root"
echo "Cooling down for $cooldown seconds..."
sleep "$cooldown"

rm -f archive9 cmix_payload compress.log compress.time.txt ppm.temp
/usr/bin/time -v -o compress.time.txt \
  nice -n -20 taskset -c "$cpu" ./cmix -e enwik9 cmix_payload \
  2>&1 | tee compress.log

test -s archive9
archive_size="$(stat -c%s archive9)"
archive_hash="$(sha256sum archive9 | awk '{print $1}')"
echo "archive9 bytes: $archive_size"
echo "archive9 SHA-256: $archive_hash"

mkdir -p "$(dirname "$export_path")"
cp archive9 "$export_path"
cp compress.time.txt "$export_path.compress.time.txt"
cp compress.log "$export_path.compress.log"

rm -f ppm.temp cmix_payload cmix_payload.cmix.temp
rm -f .coda .decomp_bin .dict .dict.comp .intro .main_phda9prepr
rm -f .new_article_order .new_article_order.comp
rm -f .r1_payload_lex_side .ready4cmix
rm -f dec1 dec2 header4archive.dat test.dat un1

echo "Exported archive: $export_path"