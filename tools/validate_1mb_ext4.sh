#!/usr/bin/env bash
set -euo pipefail

echo "NOTICE: validate_1mb_ext4.sh is a legacy raw-helper regression." >&2
echo "It does not activate or validate the canonical-stream transformer." >&2
echo "Do not use its 99,345-byte gate to accept target93." >&2

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 SOURCE_ROOT ENWIK9" >&2
  exit 2
fi

readonly source_root="$1"
readonly enwik9="$2"
readonly run_root=/root/fx4run
readonly sample_offset_mib=421
readonly sample_bytes=1048576
readonly archive_gate=99345

test -d "$source_root/src"
test -f "$enwik9"
if [[ "$(stat -c%s "$enwik9")" -lt 1000000000 ]]; then
  echo "Expected enwik9 (1,000,000,000 bytes): $enwik9" >&2
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
cp -a "$source_root/src" "$run_root/src"
cp -a "$source_root/dictionary" "$run_root/dictionary"
cp -a "$source_root/install_tools" "$run_root/install_tools"
cp -a "$source_root/tools" "$run_root/tools"

cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh" \
   "$source_root/LICENSE" "$run_root/"
chmod +x "$run_root/build_and_construct_comp.sh" "$run_root/tools/upx"

cd "$run_root"
dd if="$enwik9" of=benchmark_421mib_1mb.in \
  bs=1048576 skip="$sample_offset_mib" count=1 status=none
test "$(stat -c%s benchmark_421mib_1mb.in)" -eq "$sample_bytes"

echo "Static call-site audit..."
if grep -R -n -E \
  'EncodeString|EncodeWordString|EncodeSubstringString|AppendEncodedByte' \
  src; then
  echo "Dead-path audit failed." >&2
  exit 1
fi
test ! -e src/models/indirect.cpp

echo "Clang static analysis on changed standalone units..."
clang++-17 --analyze -std=c++17 -DNDEBUG -DSEED=923 -DUPDATE_LIMIT=3000 \
  -Isrc src/preprocess/dictionary.cpp -o /dev/null
clang++-17 --analyze -std=c++17 -DNDEBUG -DSEED=923 -DUPDATE_LIMIT=3000 \
  -Isrc src/mixer/mixer.cpp -o /dev/null

echo "Fresh accepted release+UPX build on native WSL ext4..."
./build_and_construct_comp.sh 2>&1 | tee build.log

test -x run/cmix_orig
test -x run/cmix
if ! tools/upx -t run/cmix_orig >/dev/null; then
  echo "UPX integrity test failed." >&2
  exit 1
fi

cpu=0
if (( $(nproc) > 7 )); then cpu=7; fi

rm -f benchmark.fx4 benchmark.restored ppm.temp \
      validate_compress.time validate_decompress.time \
      validate_compress.log validate_decompress.log

echo "Compressing exact accepted 421 MiB sample on CPU $cpu..."
/usr/bin/time -v -o validate_compress.time \
  taskset -c "$cpu" ./run/cmix_orig -c dictionary/english.dic \
  benchmark_421mib_1mb.in benchmark.fx4 \
  >validate_compress.log 2>&1

rm -f ppm.temp

echo "Decompressing and checking exact roundtrip..."
/usr/bin/time -v -o validate_decompress.time \
  taskset -c "$cpu" ./run/cmix_orig -d dictionary/english.dic \
  benchmark.fx4 benchmark.restored \
  >validate_decompress.log 2>&1

archive_bytes="$(stat -c%s benchmark.fx4)"
input_hash="$(sha256sum benchmark_421mib_1mb.in | awk '{print $1}')"
output_hash="$(sha256sum benchmark.restored | awk '{print $1}')"
bpb="$(awk -v a="$archive_bytes" -v n="$sample_bytes" 'BEGIN { printf "%.9f", 8*a/n }')"
core_bytes="$(stat -c%s run/cmix_orig)"
wrapper_bytes="$(stat -c%s run/cmix)"

archive_status=PASS
if (( archive_bytes > archive_gate )); then archive_status=FAIL; fi
roundtrip_status=PASS
if [[ "$input_hash" != "$output_hash" ]]; then roundtrip_status=FAIL; fi

{
  echo "Input offset:       441450496"
  echo "Input bytes:        $sample_bytes"
  echo "Archive bytes:      $archive_bytes"
  echo "Bits/byte:          $bpb"
  echo "Archive gate:       $archive_status (<= $archive_gate)"
  echo "Roundtrip:          $roundtrip_status"
  echo "Input SHA-256:      $input_hash"
  echo "Output SHA-256:     $output_hash"
  echo "Packed core bytes:  $core_bytes"
  echo "Hutter S1 bytes:    $wrapper_bytes"
  echo
  echo "Compression program output:"
  cat validate_compress.log
  echo
  echo "Compression resources:"
  grep -E 'Elapsed|User time|System time|Maximum resident|Major.*page faults|Minor.*page faults' validate_compress.time
  echo
  echo "Decompression program output:"
  cat validate_decompress.log
  echo
  echo "Decompression resources:"
  grep -E 'Elapsed|User time|System time|Maximum resident|Major.*page faults|Minor.*page faults' validate_decompress.time
} | tee validation_summary.txt

if [[ "$archive_status" != PASS || "$roundtrip_status" != PASS ]]; then
  echo "Release validation failed." >&2
  exit 1
fi

mkdir -p "$source_root/run"
cp -f run/cmix "$source_root/cmix"
cp -f run/cmix "$source_root/run/cmix"
cp -f run/cmix_orig "$source_root/run/cmix_orig"
cp -f validation_summary.txt "$source_root/tools/last_validation.txt"

rm -f ppm.temp benchmark.fx4 benchmark.restored benchmark_421mib_1mb.in

echo "Validated artifacts copied back to $source_root"
