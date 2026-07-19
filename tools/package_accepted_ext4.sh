#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ACCEPTED_CORE SOURCE_ROOT" >&2
  exit 2
fi

readonly accepted_core="$1"
readonly source_root="$2"
readonly work=/root/fx4run
cd "$work"

test -x "$accepted_core"
test -s dictionary/english.dic
test -s src/readalike_prepr/data/new_article_order
test -x tools/upx

rm -rf final_run
mkdir final_run
cp "$accepted_core" final_run/cmix_orig
chmod +x final_run/cmix_orig

if ! tools/upx -t final_run/cmix_orig >/dev/null 2>&1; then
  tools/upx --ultra-brute final_run/cmix_orig
fi
tools/upx -t final_run/cmix_orig

cd final_run
rm -f comp_dict comp_order header.dat cmix ppm.temp
./cmix_orig -c ../dictionary/english.dic comp_dict
rm -f ppm.temp
./cmix_orig -c ../src/readalike_prepr/data/new_article_order comp_order
rm -f ppm.temp

./cmix_orig -d comp_dict verify_dict
cmp -s ../dictionary/english.dic verify_dict
rm -f ppm.temp verify_dict
./cmix_orig -d comp_order verify_order
cmp -s ../src/readalike_prepr/data/new_article_order verify_order
rm -f ppm.temp verify_order

dict_size="$(stat -c%s comp_dict)"
order_size="$(stat -c%s comp_order)"
./cmix_orig -h "$dict_size" "$order_size" 0
cat cmix_orig comp_dict comp_order header.dat > cmix
chmod 0755 cmix

core_size="$(stat -c%s cmix_orig)"
s1_size="$(stat -c%s cmix)"
core_hash="$(sha256sum cmix_orig | awk '{print $1}')"
s1_hash="$(sha256sum cmix | awk '{print $1}')"

echo "Packed accepted core: $core_size bytes"
echo "Embedded dictionary:  $dict_size bytes"
echo "Embedded order:       $order_size bytes"
echo "Final Hutter S1:      $s1_size bytes"
echo "Core SHA-256:         $core_hash"
echo "S1 SHA-256:           $s1_hash"

mkdir -p "$source_root/run"
cp -f cmix "$source_root/cmix"
cp -f cmix_orig "$source_root/run/cmix_orig"
cp -f cmix "$source_root/run/cmix"
{
  echo "Packed accepted core: $core_size bytes"
  echo "Embedded dictionary:  $dict_size bytes"
  echo "Embedded order:       $order_size bytes"
  echo "Final Hutter S1:      $s1_size bytes"
  echo "Core SHA-256:         $core_hash"
  echo "S1 SHA-256:           $s1_hash"
} > "$source_root/tools/last_package.txt"

rm -f ppm.temp