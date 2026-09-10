#!/usr/bin/env bash
# One complete 1% run of this branch's single configuration.
#
# There are no arms. This branch compiles one codec, so the only thing to
# measure is that codec. What it prints at the end is an archive size for
# 5,871,388 bytes of the post-WRT stream, against the two completed reference
# numbers this project has.
#
#   BRANCH=<ref>   what to build      (default main)
#   LIMIT=<n>      bytes to code      (default 5871388, the full 1%)
#   CPU=<n>        core to pin to     (default 7)
#   CKPT=<n>       checkpoint stride  (default 65536)
#   BENCH_LTO=0    build WITHOUT -flto=thin (default is on, as shipped)
#   BENCH_PGO=0    build WITHOUT the committed profile (default on)
#   BENCH_F16=0    build WITHOUT the LSTM's fp16 shadow dot products
#
# Always builds from `git archive`: an uncommitted edit is reported and
# refused rather than silently measured, or silently not measured. Commit
# first, then run this -- there is no mode that tests uncommitted work.
set -uo pipefail

repo=/mnt/d/mywork/myideas/latestcompressor/RATA-CMIX
branch="${BRANCH:-main}"
stream="${STREAM:-/root/fx4donor/ready.pre_r1.bin}"
limit="${LIMIT:-5871388}"
cpu="${CPU:-7}"
ckpt="${CKPT:-65536}"
weights=models/6m-q4-fp32.tfwc5

# Completed 1% runs this can be read against.
#
#   673,799  fx2-cmix-transformer-v521 default stack, 6020.1 s of entropy
#   673,813  this branch at f6df4cc with BENCH_F16=0 LTO=0 PGO=0 and no
#            Scr2Match, 4543.5 s -- +14 bytes for 32.5% less time
#   673,793  the same plus Scr2Match as it first landed, silencing every
#            tie. Superseded: agreement-aware ties measured 3 bytes better
#            at 1 MB, and that is the baseline now, so this run is a
#            reference rather than a target.
#   674,399  pristine fxcm v22
#
# Scr2Match is unconditional, so there is nothing to switch. The 1 MB
# ablation that settled its shape, against the same stack without it:
#
#   silencing every tie            -10     the version that measured 673,793
#   speaking when ties agree       -13     <- kept
#   + KMP failure links            -10     different archive, same length
#   + confidence by bit position    -4     actively harmful
#
# Agreement was the whole win. KMP cost a prefix table and kBlobBytes +
# kPatterns of RAM for no byte movement; splitting confidence by bit
# position diluted cells that only about 1% of bytes ever reach. Both gone.
#
# The +14 is the optimization tier itself, not the compiler and not fp16:
# that run had all three off. The transformer's own loss is 0.462182 in
# both, so the drift is in the LSTM/mixer rework, not the transformer.
v521_stack=673799     # fx2-cmix-transformer-v521 default stack
gch_tier=673813       # this branch, optimization tier, no Scr2Match
gch_scr2=673793       # the superseded tie-silencing Scr2Match
v22_control=674399    # pristine fxcm v22

die () { printf '\nrun_1pct: %s\n' "$*" >&2; exit 1; }

command -v clang++-17 >/dev/null || die "clang++-17 not installed"
[ -f "$stream" ] || die "stream not found: $stream"
s=$(stat -c%s "$stream")
[ "$s" = 586459321 ] || die "stream is $s bytes, expected 586459321"
pgrep -x prefix_bench >/dev/null 2>&1 &&
  die "another prefix_bench is running; it would compete for cpu $cpu"

# WSL's git does not inherit core.autocrlf from the Windows side, so without
# it every CRLF file reads as modified against an LF index.
dirty=$(git -C "$repo" -c core.autocrlf=true status --porcelain -- src makefile tools)
if [ -n "$dirty" ]; then
  printf 'uncommitted changes; this builds from git archive %s, so these are NOT measured:\n' "$branch" >&2
  printf '%s\n' "$dirty" | sed 's/^/    /' >&2
  [ "${ALLOW_DIRTY:-0}" = 1 ] || die "commit them first"
fi

commit=$(git -C "$repo" rev-parse --short "$branch") || die "unknown ref: $branch"
root="/root/gch_1pct_$(date -u +%Y%m%dT%H%M%SZ)"
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

printf 'models    : %s\n' "$(grep -c . /dev/null 2>/dev/null; stat -c%s "$weights") bytes of weights"
printf '\n=== building ===\n'
make -f tools/compare_prefix_gch.mk prefix-bench -j4 \
  BENCH_LTO="${BENCH_LTO:-1}" BENCH_PGO="${BENCH_PGO:-1}" \
  BENCH_F16="${BENCH_F16:-1}" \
  > "$root/build.log" 2>&1 ||
  { tail -20 "$root/build.log" >&2; die "build failed, see $root/build.log"; }
printf 'prefix_bench: %s bytes\n\n' "$(stat -c%s prefix_bench)"

printf '=== coding %s bytes on cpu %s ===\n' "$limit" "$cpu"
/usr/bin/time -v -o "$root/compress.time.txt" \
  env FX2_CKPT="$ckpt" \
  taskset -c "$cpu" ./prefix_bench "$stream" "$limit" "$weights" \
  "$root/archive.bin" 2>&1 | tee "$root/compress.log"

a=$(stat -c%s "$root/archive.bin" 2>/dev/null) || die "no archive produced"
[ "$a" -gt 1000 ] || die "archive is $a bytes; the run did not finish"

awk -v a="$a" -v n="$limit" -v c="$commit" -v s="$v521_stack" -v v="$v22_control" \
    -v lto="${BENCH_LTO:-1}" -v pgo="${BENCH_PGO:-1}" 'BEGIN {
  printf "\nFINAL commit=%s input_bytes=%d archive_bytes=%d bpb=%.9f\n", c, n, a, a*8/n
  if (n != 5871388) {
    print "short prefix: comparable to other runs at this LIMIT, not to the numbers below"
    exit
  }
  printf "  vs v521 default stack (%d) : %+d\n", s, a - s
  if (lto == 1 || pgo == 1)
    print "    (that reference was measured WITHOUT lto/pgo; both change\n     float association, so a small delta here is the build, not the model.\n     BENCH_LTO=0 BENCH_PGO=0 for a like-for-like comparison)"
  printf "  vs pristine v22 control (%d) : %+d\n", v, a - v
  if (a == s) print "  identical to the v521 stack -- the port preserved the archive exactly"
}'
printf 'results: %s\n' "$root"
