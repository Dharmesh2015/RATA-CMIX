#!/usr/bin/env bash
# Stage 2: measure what donor experts add ON TOP OF PPMd+LSTM+FXCM+mixer.
#
#   Run A: isolated region, ensemble only
#   Run B: isolated region, ensemble + donor_profile expert (oracle page spans)
#   gain = A - B
#
# Both runs are cold on the identical region, so the cold-start handicap
# cancels and the difference isolates the expert's contribution. The expert
# emits a bounded logit correction over immutable continuation tables and does
# not mutate PPMd/LSTM/FXCM/mixer state.
#
# If B == A exactly the plan was a no-op (silently ignored) -- reported as
# PLAN_NOOP rather than as a real zero.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 START_GROUP END_GROUP" >&2
  exit 2
fi

readonly root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly stream=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin
readonly grouped="$root/results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv"
readonly pagemap="$root/results/donor_multiday_seed/enwik9.page_recipients.csv"
readonly binary="${FX4_ISO_BINARY:-/root/fx4_page_donor_v3/cmix_page_donor}"
readonly dict="$root/dictionary/english.dic"
readonly out="$root/results/oracle_tail"
readonly cpu="${FX4_DONOR_CPU:-4}"
readonly work="${FX4_AB_WORK:-/root/fx4ab_cpu$cpu}"
readonly ledger="$out/oracle_ab.csv"

test -x "$binary"; test -f "$dict"; test -f "$grouped"; test -f "$pagemap"
mkdir -p "$out" "$work"
exec 9>"$work/ab.lock"; flock -n 9 || { echo "another A/B run owns $work" >&2; exit 3; }

[[ -f "$ledger" ]] || echo "group,offset,length,spans,baseline_A,assisted_B,gain_bytes,bpb_A,bpb_B,status" > "$ledger"

enc() { # file tag [planfile]
  local in="$1" tag="$2" plan="${3:-}"
  local o="$work/o.$tag"
  rm -f "$o" "$work/ppm.temp" ppm.temp
  if [[ -n "$plan" ]]; then
    FX4_RAW_ENTROPY_INPUT=1 FX4_DONOR_PLAN="$plan" FX4_SELECTIVE_POSTR1=1 \
      taskset -c "$cpu" "$binary" -n "$dict" "$in" "$o" >/dev/null 2>&1
  else
    FX4_RAW_ENTROPY_INPUT=1 \
      taskset -c "$cpu" "$binary" -n "$dict" "$in" "$o" >/dev/null 2>&1
  fi
  local n; n="$(stat -c%s "$o" 2>/dev/null || echo 0)"
  # A rejected plan exits 0 and leaves a zero-byte file. Never score that.
  if (( n <= 0 )); then
    echo "ENCODE FAILED (empty output) [$tag] -- plan likely rejected" >&2
    exit 4
  fi
  echo "$n"
  rm -f "$o" "$work/ppm.temp" ppm.temp
}

cd "$work"
for (( g = $1; g <= $2; g++ )); do
  if awk -F, -v g="$g" 'NR>1 && $1==g{f=1} END{exit !f}' "$ledger"; then
    echo "group $g done, skipping"; continue
  fi
  row="$(awk -F, -v g="$g" '$1==g{print;exit}' "$grouped")"
  [[ -n "$row" ]] || { echo "group $g missing"; continue; }
  off="$(cut -d, -f4 <<<"$row")"; len="$(cut -d, -f6 <<<"$row")"

  plan="$out/pack${g}_oracle.csv"
  if [[ ! -f "$plan" ]]; then
    python3 "$root/tools/build_postr1_page_oracle.py" "$pagemap" "$plan" \
      --start "$off" --length "$len" --experts 'oracle+donor_profile' >/dev/null
  fi
  spans=$(( $(wc -l < "$plan") - 1 ))

  dd if="$stream" of=region.bin bs=4M skip="$off" count="$len" \
     iflag=skip_bytes,count_bytes status=none

  a="$(enc region.bin A)"
  b="$(enc region.bin B "$plan")"
  gain=$(( a - b ))
  status=OK
  (( a == b )) && status=PLAN_NOOP
  bpa="$(awk -v x="$a" -v n="$len" 'BEGIN{printf "%.6f", 8*x/n}')"
  bpb="$(awk -v x="$b" -v n="$len" 'BEGIN{printf "%.6f", 8*x/n}')"

  printf '%d,%d,%d,%d,%d,%d,%d,%s,%s,%s\n' \
    "$g" "$off" "$len" "$spans" "$a" "$b" "$gain" "$bpa" "$bpb" "$status" >> "$ledger"
  echo "group $g spans=$spans A=$a B=$b gain=$gain bpb $bpa -> $bpb [$status]"
  rm -f region.bin
done
echo "done -> $ledger"
