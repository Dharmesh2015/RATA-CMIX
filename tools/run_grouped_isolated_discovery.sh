#!/usr/bin/env bash
# Fast isolated donor discovery over page-aligned grouped post-R1 recipients.
#
# Mirrors the measurement that produced the accepted region-421 result: each
# recipient is sliced out and encoded standalone, so cost is independent of the
# recipient's offset in the stream. No causal replay from byte 0.
#
# Conditional payload of a region given a donor prefix is measured exactly as
#   assisted = |encode(donor ++ region)| - |encode(donor)|
# and compared against
#   baseline = |encode(region)|
#
# This is a SCREEN. Any group with gain > metadata must still pass full-prefix
# causal validation before promotion (acceptance rule 7).
set -euo pipefail

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
  echo "usage: $0 POSTR1_STREAM RECIPIENTS_CSV CANDIDATES_CSV START_GROUP END_GROUP [TOP_N]" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly stream="$(readlink -f "$1")"
readonly recipients_csv="$(readlink -f "$2")"
readonly candidates_csv="$(readlink -f "$3")"
readonly start_group="$4"
readonly end_group="$5"
readonly top_n="${6:-3}"
readonly cpu="${FX4_DONOR_CPU:-7}"
readonly result_root="${FX4_ISO_RESULT_ROOT:-$source_root/results/grouped_isolated_discovery}"
readonly work_root="${FX4_ISO_WORK_ROOT:-/root/fx4iso_cpu$cpu}"
readonly ledger="$result_root/isolated_gains.csv"
readonly donor_cache="$work_root/donorcache"

test -f "$stream"
test -f "$recipients_csv"
test -f "$candidates_csv"
if [[ "$(stat -c%s "$stream")" -ne 587138826 ]]; then
  echo "expected the canonical 587,138,826-byte post-R1 stream" >&2
  exit 2
fi
[[ "$start_group" =~ ^[0-9]+$ && "$end_group" =~ ^[0-9]+$ && "$top_n" =~ ^[0-9]+$ ]]
if (( start_group < 1 )); then
  echo "group 0 has no causal predecessor and therefore no donors; start at 1" >&2
  exit 2
fi

mkdir -p "$result_root" "$work_root" "$donor_cache"
exec 9>"$work_root/run.lock"
if ! flock -n 9; then
  echo "another isolated run owns $work_root" >&2
  exit 3
fi

readonly binary="${FX4_ISO_BINARY:-/root/fx4_page_donor_v3/cmix_page_donor}"
if [[ ! -x "$binary" ]]; then
  echo "missing $binary -- run the page donor specialist once to build it" >&2
  exit 2
fi
readonly dict="$source_root/dictionary/english.dic"
test -f "$dict"

if [[ ! -f "$ledger" ]]; then
  echo "group,recipient_offset,recipient_length,candidate_rank,donor_offset,donor_length,baseline_bytes,donor_alone_bytes,warm_total_bytes,assisted_bytes,gain_bytes,phrase_score" \
    > "$ledger"
fi

# Encode a file standalone and echo its payload size.
encode_bytes() {
  local in="$1" tag="$2"
  local out="$work_root/out.$tag"
  rm -f "$out" "$work_root/ppm.temp" ppm.temp
  FX4_RAW_ENTROPY_INPUT=1 taskset -c "$cpu" "$binary" -n "$dict" "$in" "$out" >/dev/null 2>&1
  local n; n="$(stat -c%s "$out" 2>/dev/null || echo 0)"
  rm -f "$out" "$work_root/ppm.temp" ppm.temp
  # A failed encode exits 0 and leaves a zero-byte file. Never let that be
  # scored as a gain -- abort loudly instead.
  if (( n <= 0 )); then
    echo "ENCODE FAILED (empty output) for $in [$tag]" >&2
    exit 4
  fi
  echo "$n"
}

cd "$work_root"

for (( g = start_group; g <= end_group; g++ )); do
  if awk -F, -v g="$g" 'NR>1 && $1==g { found=1 } END { exit !found }' "$ledger"; then
    echo "group $g already in ledger, skipping"
    continue
  fi

  rline="$(awk -F, -v g="$g" '$1==g { print; exit }' "$recipients_csv")"
  [[ -n "$rline" ]] || { echo "group $g not in recipients csv"; continue; }
  roff="$(cut -d, -f4 <<<"$rline")"
  rlen="$(cut -d, -f6 <<<"$rline")"

  mapfile -t cands < <(awk -F, -v g="$g" -v n="$top_n" \
    'NR>1 && $1==g { print $4","$5","$6","$7; c++; if (c>=n) exit }' "$candidates_csv")
  if (( ${#cands[@]} == 0 )); then
    echo "group $g has no causal candidates, skipping"
    continue
  fi

  dd if="$stream" of=region.bin bs=4M skip="$roff" count="$rlen" \
     iflag=skip_bytes,count_bytes status=none
  baseline="$(encode_bytes region.bin region)"
  echo "group $g off=$roff len=$rlen baseline=$baseline"

  for c in "${cands[@]}"; do
    IFS=, read -r rank doff dlen score <<<"$c"

    donor_file="$donor_cache/d_${doff}_${dlen}.bin"
    donor_size_file="$donor_cache/d_${doff}_${dlen}.size"
    if [[ ! -f "$donor_file" ]]; then
      dd if="$stream" of="$donor_file" bs=4M skip="$doff" count="$dlen" \
         iflag=skip_bytes,count_bytes status=none
    fi
    if [[ -f "$donor_size_file" ]]; then
      donor_alone="$(cat "$donor_size_file")"
    else
      donor_alone="$(encode_bytes "$donor_file" donor)"
      echo "$donor_alone" > "$donor_size_file"
    fi

    # Control: the dlen bytes immediately preceding the region. A donor is only
    # interesting if it beats the recipient's own adjacent context, which the
    # real causal pipeline already holds in model state for free. Without this,
    # a cold baseline makes every nearby donor look like a winner.
    ctl_off=$(( roff - dlen ))
    if (( ctl_off < 0 )); then ctl_off=0; fi
    ctl_file="$donor_cache/c_${ctl_off}_${dlen}.bin"
    ctl_size_file="$donor_cache/c_${ctl_off}_${dlen}.size"
    ctl_pair_file="$donor_cache/cp_${roff}_${dlen}.size"
    if [[ ! -f "$ctl_file" ]]; then
      dd if="$stream" of="$ctl_file" bs=4M skip="$ctl_off" count="$dlen" \
         iflag=skip_bytes,count_bytes status=none
    fi
    if [[ -f "$ctl_size_file" ]]; then
      ctl_alone="$(cat "$ctl_size_file")"
    else
      ctl_alone="$(encode_bytes "$ctl_file" ctl)"
      echo "$ctl_alone" > "$ctl_size_file"
    fi
    if [[ -f "$ctl_pair_file" ]]; then
      ctl_assisted="$(cat "$ctl_pair_file")"
    else
      cat "$ctl_file" region.bin > ctl.bin
      ctl_total="$(encode_bytes ctl.bin ctlpair)"
      ctl_assisted=$(( ctl_total - ctl_alone ))
      echo "$ctl_assisted" > "$ctl_pair_file"
      rm -f ctl.bin
    fi

    cat "$donor_file" region.bin > warm.bin
    warm_total="$(encode_bytes warm.bin warm)"
    assisted=$(( warm_total - donor_alone ))
    gain=$(( ctl_assisted - assisted ))

    distance=$(( roff - doff - dlen ))
    printf '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n' \
      "$g" "$roff" "$rlen" "$rank" "$doff" "$dlen" \
      "$baseline" "$ctl_assisted" "$assisted" "$gain" "$distance" \
      "$donor_alone" "$warm_total" "$score" \
      >> "$ledger"

    if (( gain > 0 )); then
      echo "  *** rank=$rank donor=$doff/$dlen dist=$distance NET_GAIN=$gain vs adjacent control"
    else
      echo "      rank=$rank donor=$doff/$dlen dist=$distance net=$gain"
    fi
    rm -f warm.bin
  done
  rm -f region.bin
done

echo "done: groups $start_group..$end_group -> $ledger"
