#!/usr/bin/env bash
# Encode a 1%-scale prefix, decode it back, and confirm the result matches --
# a real round trip, not just an archive-size measurement.
#
# WHAT THIS IS AND IS NOT. S1 (the compressor binary) does not scale with
# how much of enwik9 you compress -- it is a fixed artifact, build it once
# with build_and_construct_comp.sh and its size is its size. What scales is
# S2 (the archive), and that is what this measures: the coder's own output
# on a prefix of the post-WRT stream, via the same Predictor/Encoder/Decoder
# classes the real codec uses -- not the self-extracting production package.
#
# A genuine S1-based `-e`/self-extract round trip cannot run on a prefix:
# split4Comp()/reorder() need the real, structurally complete enwik9 article
# set. Confirmed directly -- a 2 MB raw slice produced both a real error and
# a multi-day runtime projection. That is a full-enwik9-only test
# (tools/run_google_cloud_hutter.sh already does it); this is the fast,
# WSL-sized alternative for everything short of that.
#
#   BRANCH=<ref>   what to build      (default main)
#   LIMIT=<n>      bytes to round-trip (default 5871388, the full 1%)
#   CPU=<n>        core to pin to     (default 7)
#   CKPT=<n>       checkpoint stride  (default 65536)
#   BENCH_LTO=0    build WITHOUT -flto=thin (default is on, as shipped)
#   BENCH_PGO=0    build WITHOUT the committed profile (default on)
#   BENCH_F16=0    build WITHOUT the LSTM's fp16 shadow dot products
#
# Always builds from `git archive`: an uncommitted edit is refused, not
# silently measured.
set -uo pipefail

repo=/mnt/d/mywork/myideas/latestcompressor/RATA-CMIX
branch="${BRANCH:-main}"
stream="${STREAM:-/root/fx4donor/ready.pre_r1.bin}"
limit="${LIMIT:-5871388}"
cpu="${CPU:-7}"
ckpt="${CKPT:-65536}"
weights=models/6m-q4-fp32.tfwc5

die () { printf '\nrun_1pct_roundtrip: %s\n' "$*" >&2; exit 1; }

command -v clang++-17 >/dev/null || die "clang++-17 not installed"
[ -f "$stream" ] || die "stream not found: $stream"
s=$(stat -c%s "$stream")
[ "$s" = 586459321 ] || die "stream is $s bytes, expected 586459321"
pgrep -x prefix_bench >/dev/null 2>&1 &&
  die "a prefix_bench is already running; it would compete for cpu $cpu"
pgrep -x prefix_bench_decode >/dev/null 2>&1 &&
  die "a prefix_bench_decode is already running; it would compete for cpu $cpu"

dirty=$(git -C "$repo" -c core.autocrlf=true status --porcelain -- src makefile tools)
if [ -n "$dirty" ]; then
  printf 'uncommitted changes; this builds from git archive %s, so these are NOT measured:\n' "$branch" >&2
  printf '%s\n' "$dirty" | sed 's/^/    /' >&2
  [ "${ALLOW_DIRTY:-0}" = 1 ] || die "commit them first"
fi

commit=$(git -C "$repo" rev-parse --short "$branch") || die "unknown ref: $branch"
root="/root/gch_roundtrip_$(date -u +%Y%m%dT%H%M%SZ)"
tree="$root/tree"
mkdir -p "$tree" || die "cannot create $root"

printf '=== %s @ %s ===\n' "$branch" "$commit"
printf 'weights   : %s\n' "$weights"
printf 'limit     : %s bytes\n' "$limit"
printf 'lto / pgo : %s / %s   f16: %s\n' \
  "${BENCH_LTO:-1}" "${BENCH_PGO:-1}" "${BENCH_F16:-1}"

git -C "$repo" archive "$branch" | tar -x -C "$tree" || die "git archive failed"
cd "$tree" || die "cannot enter $tree"
[ -f "$weights" ] || die "weights missing from the archived tree: $weights"
cp dictionary/english.dic .dict || die "dictionary/english.dic missing"

printf '\n=== building (encode + decode) ===\n'
make -f tools/compare_prefix_gch.mk prefix-bench-roundtrip -j4 \
  BENCH_LTO="${BENCH_LTO:-1}" BENCH_PGO="${BENCH_PGO:-1}" \
  BENCH_F16="${BENCH_F16:-1}" \
  > "$root/build.log" 2>&1 ||
  { tail -20 "$root/build.log" >&2; die "build failed, see $root/build.log"; }
printf 'prefix_bench       : %s bytes\n' "$(stat -c%s prefix_bench)"
printf 'prefix_bench_decode: %s bytes\n\n' "$(stat -c%s prefix_bench_decode)"

printf '=== S2: encoding %s bytes on cpu %s ===\n' "$limit" "$cpu"
env FX2_CKPT="$ckpt" taskset -c "$cpu" ./prefix_bench \
  "$stream" "$limit" "$weights" "$root/archive.bin" 2>&1 |
  tee "$root/encode.log"
a=$(stat -c%s "$root/archive.bin" 2>/dev/null) || die "no archive produced"
[ "$a" -gt 1000 ] || die "archive is $a bytes; encode did not finish"

printf '\n=== round trip: decoding %s back ===\n' "$root/archive.bin"
env FX2_CKPT="$ckpt" taskset -c "$cpu" ./prefix_bench_decode \
  "$stream" "$weights" "$root/archive.bin" "$root/decoded.bin" 2>&1 |
  tee "$root/decode.log"
status=$?

printf '\n=== S1 (unrelated to the above -- fixed size, not "at 1%%") ===\n'
printf 'To see it: build_and_construct_comp.sh produces the real self-\n'
printf 'extracting cmix; its byte count is S1, the same at any input size.\n'

printf '\n=== summary ===\n'
printf '  S2 (archive at this limit) : %s bytes\n' "$a"
grep -m1 '^ROUNDTRIP=' "$root/decode.log" | sed 's/^/  /'
printf 'results: %s\n' "$root"
exit "$status"
