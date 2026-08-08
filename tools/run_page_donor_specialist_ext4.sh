#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 POST_R1_STREAM PAGE_RECIPIENTS_CSV F4CD_ENVELOPE [STOP_MIB]" >&2
  exit 2
fi

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly input_path="$(readlink -f "$1")"
readonly packs_path="$(readlink -f "$2")"
readonly plan_path="$(readlink -f "$3")"
readonly stop_mib="${4:-6}"
readonly stop_offset="$((stop_mib * 1048576))"
readonly cpu="${FX4_DONOR_CPU:-7}"
readonly work_root=/root/fx4_page_donor_v3
readonly result_root="${FX4_DONOR_RESULT_ROOT:-$source_root/research/page_donor_pages_20260808}"
readonly ledger="$result_root/page"
readonly candidate_path="${FX4_WINNER_PAGE_CANDIDATES_CSV:-$result_root/page_candidates.csv}"
readonly recipient_path="$candidate_path.recipients.csv"
readonly use_ranked="${FX4_WINNER_USE_RANKED:-1}"

test -f "$input_path"
test -f "$packs_path"
test -f "$plan_path"
[[ "$(stat -c%s "$input_path")" -eq 587138826 ]]
grep -q '^pack_id,first_page,page_count,post_r1_start,post_r1_end,post_r1_length,decoded_length,stream_class,class_name,feature_mask$' "$packs_path"
[[ "$stop_mib" =~ ^[1-9][0-9]*$ ]]

mkdir -p "$work_root" "$result_root"
exec 9>"$work_root/run.lock"
if ! flock -n 9; then
  echo "another page donor run owns $work_root" >&2
  exit 3
fi

rm -rf "$work_root/source"
mkdir -p "$work_root/source"
cp -a "$source_root/src" "$source_root/dictionary"   "$source_root/tools" "$work_root/source/"
cp "$source_root/makefile" "$source_root/LICENSE" "$work_root/source/"
cd "$work_root/source"
make clean
make page_donor_ranker -j"$(nproc)"
mv postr1_page_donor_ranker "$work_root/postr1_page_donor_ranker"

if [[ "$use_ranked" == 1 && (! -s "$candidate_path" || ! -s "$recipient_path" ||
      "${FX4_WINNER_RERANK:-0}" == 1) ]]; then
  rm -f "$candidate_path" "$recipient_path"
  "$work_root/postr1_page_donor_ranker" \
    "$input_path" "$packs_path" "$candidate_path" \
    "${FX4_WINNER_PROXY_CANDIDATES:-12}" "$stop_offset"
fi

make cmix -j"$(nproc)" DONOR_DISCOVERY=1 OUT=cmix_page_donor
mv cmix_page_donor "$work_root/cmix_page_donor"

readonly metadata="$result_root/run.meta"
readonly metadata_new="$result_root/run.meta.new"
{
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "packs_sha256=$(sha256sum "$packs_path" | cut -d' ' -f1)"
  echo "plan_sha256=$(sha256sum "$plan_path" | cut -d' ' -f1)"
  if [[ "$use_ranked" == 1 ]]; then
    echo "candidates_sha256=$(sha256sum "$candidate_path" | cut -d' ' -f1)"
  else
    echo "candidates_sha256=disabled"
  fi
  echo "stop_offset=$stop_offset"
  echo "binary_sha256=$(sha256sum "$work_root/cmix_page_donor" | cut -d' ' -f1)"
} >"$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing ledger belongs to a different stream, pack map, plan, or binary" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

cd "$work_root"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp
rank_env=()
if [[ "$use_ranked" == 1 ]]; then
  rank_env+=(
    "FX4_WINNER_PAGE_CANDIDATES_CSV=$candidate_path"
  )
fi
set +e
/usr/bin/time -v -o "$result_root/latest.time.txt" \
  env \
    "${rank_env[@]}" \
    FX4_RAW_ENTROPY_INPUT=1 \
    FX4_DONOR_PLAN="$plan_path" \
    FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
    FX4_DONOR_WINNER_SEARCH=1 \
    FX4_WINNER_PACKS_CSV="$packs_path" \
    FX4_WINNER_STOP_OFFSET="$stop_offset" \
    FX4_WINNER_PLANNED_ONLY="${FX4_WINNER_PLANNED_ONLY:-1}" \
    FX4_WINNER_PLANNED_PREFIXES="${FX4_WINNER_PLANNED_PREFIXES:-1}" \
    FX4_WINNER_PLANNED_DONORS="${FX4_WINNER_PLANNED_DONORS:-7}" \
    FX4_WINNER_START_REGION="${FX4_WINNER_START_REGION:-0}" \
    FX4_WINNER_MAX_REGIONS=0 \
    FX4_WINNER_CANDIDATES="${FX4_WINNER_CANDIDATES:-12}" \
    FX4_DONOR_MAX_NEW_TRIALS="${FX4_DONOR_MAX_NEW_TRIALS:-0}" \
    FX4_DONOR_TRIAL_CPU="$cpu" \
  taskset -c "$cpu" ./cmix_page_donor -n \
    "$work_root/source/dictionary/english.dic" "$input_path" discovery_payload \
    2>&1 | tee -a "$result_root/discovery.log"
status=${PIPESTATUS[0]}
set -e

rm -f discovery_payload discovery_payload.cmix.temp ppm.temp
echo "exit_status=$status"
for suffix in winner_selected.csv winner_positive.csv winner_near.csv     winner.status winner.paused winner.complete; do
  path="$ledger.$suffix"
  if [[ -f "$path" ]]; then
    echo "==== $path"
    tail -n 20 "$path"
  fi
done
if [[ -s "$ledger.winner_selected.csv" ]]; then
  bash "$work_root/source/tools/summarize_page_donor_results.sh" \
    "$packs_path" "$ledger.winner_selected.csv" "$result_root" "$stop_mib"
  cat "$result_root/region_summary.csv"
  cat "$result_root/class_summary.csv"
fi
exit "$status"

