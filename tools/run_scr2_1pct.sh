#!/usr/bin/env bash
# Does SCR2 pay, on THIS stream, once the coder is asked rather than the raw
# byte count?
#
# The transform's own accounting says yes by a wide margin: 5,871,388 bytes of
# our 1% prefix become 5,117,435 plus 1,232 of metadata, a 12.82% raw saving.
# That number counts bytes removed. It does not count what the coder was
# already paying for them, which for a pattern appearing 3,835 times is a
# fraction of a bit rather than eight bytes.
#
# So this codes the transformed stream with the same binary, same weights, same
# flags as the baseline, and compares TOTAL cost for reconstructing the same
# 5,871,388 original bytes:
#
#   baseline   archive(5,871,388 original)                 = 673,813
#   scr2       archive(5,117,435 transformed) + metadata
#
# Produce the inputs first with postr1_structural_codec.py; see SCR2_INPUT.
# RESULT: THIS CANNOT RUN, and the reason is architectural rather than a
# tuning problem. SCR2 substitutes patterns with marker and escape bytes
# drawn from the values the stream does not use. Our 1% prefix uses 203 of
# 256; the transform takes ALL 51 that are free, and the coder stops with
#
#   cmix error: the transformer was trained on the enwik9 vocabulary of 205
#   bytes, but this input has a vocabulary of 254 bytes
#
# The frozen 6M transformer has a fixed 205-symbol vocabulary. A transform
# that expands the alphabet to 254 cannot be placed in front of it at any
# setting -- there is no version of SCR2-as-a-transform that fits this
# architecture without retraining the transformer on the new alphabet,
# which is 8 GPUs for 26 hours to test an idea whose best previous result
# was a 46-byte archive LOSS.
#
# Kept because the derivation and the round-trip check are still valid, and
# because the next person to propose a stream transform in front of a frozen
# model should find this here first.
#
set -uo pipefail

repo=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
branch="${BRANCH:-exp/gch-optimization}"
scr2_dir="${SCR2_DIR:-/root/scr2_ours}"
scr2_stream="$scr2_dir/ours1pct.bin"
scr2_meta="$scr2_dir/ours1pct.meta"
cpu="${CPU:-7}"
ckpt="${CKPT:-65536}"
weights=models/6m-q4-fp32.tfwc5

# The baseline this is judged against: same branch, same build flags, the
# untransformed 1% prefix.
baseline_archive=673813
original_bytes=5871388

die () { printf '\nrun_scr2: %s\n' "$*" >&2; exit 1; }

command -v clang++-17 >/dev/null || die "clang++-17 not installed"
[ -f "$scr2_stream" ] || die "transformed stream missing: $scr2_stream"
[ -f "$scr2_meta" ] || die "metadata missing: $scr2_meta"
pgrep -x prefix_bench >/dev/null 2>&1 &&
  die "another prefix_bench is running; it would compete for cpu $cpu"

limit=$(stat -c%s "$scr2_stream")
meta=$(stat -c%s "$scr2_meta")

dirty=$(git -C "$repo" -c core.autocrlf=true status --porcelain -- src makefile tools)
[ -z "$dirty" ] || [ "${ALLOW_DIRTY:-0}" = 1 ] ||
  die "uncommitted changes in src/makefile/tools; commit them or ALLOW_DIRTY=1"

commit=$(git -C "$repo" rev-parse --short "$branch") || die "unknown ref: $branch"
root="/root/scr2_1pct_$(date -u +%Y%m%dT%H%M%SZ)"
tree="$root/tree"
mkdir -p "$tree" || die "cannot create $root"

printf '=== SCR2 on our stream: %s @ %s ===\n' "$branch" "$commit"
printf '  transformed : %s bytes\n' "$limit"
printf '  metadata    : %s bytes\n' "$meta"
printf '  reconstructs: %s original bytes\n' "$original_bytes"
printf '  baseline    : %s (untransformed, same build)\n' "$baseline_archive"
printf '  lto/pgo/f16 : %s / %s / %s\n' "${BENCH_LTO:-1}" "${BENCH_PGO:-1}" \
  "${BENCH_F16:-1}"

git -C "$repo" archive "$branch" | tar -x -C "$tree" || die "git archive failed"
cd "$tree" || die "cannot enter $tree"
cp dictionary/english.dic .dict || die "dictionary/english.dic missing"

printf '\n=== building ===\n'
make -f tools/compare_prefix_gch.mk prefix-bench -j4 \
  BENCH_LTO="${BENCH_LTO:-1}" BENCH_PGO="${BENCH_PGO:-1}" \
  BENCH_F16="${BENCH_F16:-1}" > "$root/build.log" 2>&1 ||
  { tail -20 "$root/build.log" >&2; die "build failed, see $root/build.log"; }
printf 'prefix_bench: %s bytes\n\n' "$(stat -c%s prefix_bench)"

printf '=== coding %s transformed bytes on cpu %s ===\n' "$limit" "$cpu"
/usr/bin/time -v -o "$root/compress.time.txt" \
  env FX2_CKPT="$ckpt" \
  taskset -c "$cpu" ./prefix_bench "$scr2_stream" "$limit" "$weights" \
  "$root/archive.bin" 2>&1 | tee "$root/compress.log"

a=$(stat -c%s "$root/archive.bin" 2>/dev/null) || die "no archive produced"
[ "$a" -gt 1000 ] || die "archive is $a bytes; the run did not finish"

awk -v a="$a" -v m="$meta" -v b="$baseline_archive" -v n="$original_bytes" \
    -v t="$limit" 'BEGIN {
  total = a + m
  printf "\nFINAL scr2 archive=%d + metadata=%d = %d\n", a, m, total
  printf "  baseline (no transform)          %d\n", b
  printf "  SCR2 total                       %d   %+d\n", total, total - b
  printf "  bpb over the original %d bytes  %.9f  (baseline %.9f)\n", \
    n, total * 8 / n, b * 8 / n
  print ""
  if (total < b)
    printf "  SCR2 WINS by %d bytes on this stream.\n", b - total
  else
    printf "  SCR2 LOSES by %d bytes. The %d raw bytes it removed were\n" \
           "  already costing the coder less than the transform saves.\n", \
           total - b, n - t - m
}'
printf 'results: %s\n' "$root"
