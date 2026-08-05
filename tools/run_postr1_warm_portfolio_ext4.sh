#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 POST_R1_STREAM [COMPLETE_MIB_REGIONS]" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly input_path="$(readlink -f "$1")"
readonly regions="${2:-6}"
readonly chunk_size=1048576
readonly full_stream_bytes=587138826
readonly pilot_bytes=$((regions == 559 ? full_stream_bytes : regions * chunk_size))
readonly work_root=/root/fx4warm_postr1
readonly result_root="${FX4_WARM_RESULT_ROOT:-$source_root/research/postr1_warm_1pct_v1}"
readonly ledger="$result_root/warm"
readonly cpu="${FX4_DONOR_CPU:-7}"

test -f "$input_path"
if [[ "$(stat -c%s "$input_path")" -ne 587138826 ]]; then
  echo "expected the canonical 587,138,826-byte post-R1 stream" >&2
  exit 2
fi
if (( regions < 2 || regions > 559 )); then
  echo "COMPLETE_MIB_REGIONS must be between 2 and 559" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
exec 9>"$work_root/run.lock"
if ! flock -n 9; then
  echo "another warm portfolio run owns $work_root" >&2
  exit 3
fi

rm -rf -- "$work_root/source"
mkdir -p "$work_root/source"
cp -a "$source_root/src" "$source_root/dictionary" \
  "$source_root/tools" "$work_root/source/"
cp "$source_root/makefile" "$source_root/LICENSE" "$work_root/source/"
sed -i 's/\r$//' "$work_root/source/makefile" \
  "$work_root/source/tools/"*.sh

cd "$work_root/source"
rm -f -- ./*.o cmix_discovery cmix_production
make cmix -j"$(nproc)" DONOR_DISCOVERY=1 OUT=cmix_discovery
mv cmix_discovery "$work_root/cmix_discovery"
rm -f -- ./*.o
make cmix -j"$(nproc)" DONOR=1 OUT=cmix_production
mv cmix_production "$work_root/cmix_production"

cd "$work_root"
if [[ ! -f postr1_pilot.bin ]] || \
   [[ "$(stat -c%s postr1_pilot.bin)" -ne "$pilot_bytes" ]]; then
  if (( regions == 559 )); then
    cp "$input_path" postr1_pilot.bin
  else
    dd if="$input_path" of=postr1_pilot.bin bs=1048576 count="$regions" \
      status=none
  fi
fi
python3 "$work_root/source/tools/build_planned_donor_profiles.py" \
  postr1_pilot.bin candidates.f4cd \
  --candidates \
  "$source_root/research/donor_graph_post_r1_20260727/exact_mesh_all_candidates.csv" \
  --donors-per-region 7 \
  --pin-region-421 \
  > "$result_root/candidate_plan.json"
cp -f candidates.csv "$result_root/candidate_plan.csv"

readonly metadata="$result_root/run.meta"
readonly metadata_new="$result_root/run.meta.new"
{
  echo "input_sha256=$(sha256sum postr1_pilot.bin | cut -d' ' -f1)"
  echo "candidate_plan_sha256=$(sha256sum candidates.f4cd | cut -d' ' -f1)"
  echo "discovery_binary_sha256=$(sha256sum cmix_discovery | cut -d' ' -f1)"
  echo "production_binary_sha256=$(sha256sum cmix_production | cut -d' ' -f1)"
  echo "regions=$regions"
} > "$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing ledger belongs to another input, plan, binary, or region count" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

echo "Warm post-R1 donor portfolio pilot"
echo "  input bytes: $pilot_bytes ($regions complete MiB regions)"
echo "  tested recipients: 1 through $((regions - 1))"
echo "  candidates: one ordered seven-donor profile per recipient"
echo "  fallback: exact no-donor branch"
echo "  CPU: $cpu"
echo "  durable results: $result_root"

if [[ ! -f "$ledger.winner.complete" ]]; then
  rm -f discovery_payload discovery_payload.cmix.temp ppm.temp
  /usr/bin/time -v -o "$result_root/discovery.time.txt" \
    env FX4_RAW_ENTROPY_INPUT=1 \
        FX4_DONOR_PLAN="$work_root/candidates.f4cd" \
        FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
        FX4_DONOR_WINNER_SEARCH=1 \
        FX4_WINNER_PLANNED_ONLY=1 \
        FX4_WINNER_START_REGION="${FX4_WINNER_START_REGION:-1}" \
        FX4_WINNER_MAX_REGIONS="${FX4_WINNER_MAX_REGIONS:-0}" \
        FX4_DONOR_TRIAL_CPU="$cpu" \
    ionice -c2 -n0 nice -n -10 taskset -c "$cpu" \
    ./cmix_discovery -n "$work_root/source/dictionary/english.dic" \
      postr1_pilot.bin discovery_payload \
    2>&1 | tee -a "$result_root/discovery.log"
fi

python3 "$work_root/source/tools/build_winner_plan.py" \
  candidates.f4cd "$ledger.winner_selected.csv" selected.f4cp \
  | tee "$result_root/selected_plan.json"
cp -f selected.f4cp "$result_root/selected.f4cp"

python3 - "$ledger.winner_selected.csv" \
  > "$result_root/region_summary.json" <<'PY'
import csv
import json
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    rows = list(csv.DictReader(source))
latest = {int(row["recipient_region"]): row for row in rows}
values = list(latest.values())
print(json.dumps({
    "tested_regions": len(values),
    "winning_regions": sum(int(row["net_gain_bytes"]) > 0 for row in values),
    "gross_payload_gain_bytes": sum(int(row["gain_bytes"]) for row in values),
    "net_gain_after_plan_bytes": sum(max(0, int(row["net_gain_bytes"])) for row in values),
    "regions": latest,
}, indent=2))
PY

if [[ ! -f "$ledger.winner.complete" ]]; then
  echo "Warm discovery batch paused cleanly; rerun the same command to resume."
  echo "Current production plan: $result_root/selected.f4cp"
  rm -f discovery_payload discovery_payload.cmix.temp ppm.temp
  exit 0
fi

rm -f selected.fx4 restored.postr1 ppm.temp
/usr/bin/time -v -o "$result_root/compress.time.txt" \
  env FX4_RAW_ENTROPY_INPUT=1 \
      FX4_DONOR_PLAN="$work_root/selected.f4cp" \
  ionice -c2 -n0 nice -n -10 taskset -c "$cpu" \
  ./cmix_production -n "$work_root/source/dictionary/english.dic" \
    postr1_pilot.bin selected.fx4 \
  2>&1 | tee "$result_root/compress.log"
/usr/bin/time -v -o "$result_root/decompress.time.txt" \
  env FX4_RAW_ENTROPY_OUTPUT=1 \
  ionice -c2 -n0 nice -n -10 taskset -c "$cpu" \
  ./cmix_production -d "$work_root/source/dictionary/english.dic" \
    selected.fx4 restored.postr1 \
  2>&1 | tee "$result_root/decompress.log"
cmp postr1_pilot.bin restored.postr1

{
  stat -c 'archive_bytes=%s' selected.fx4
  stat -c 'production_binary_bytes=%s' cmix_production
  echo "input_sha256=$(sha256sum postr1_pilot.bin | cut -d' ' -f1)"
  echo "output_sha256=$(sha256sum restored.postr1 | cut -d' ' -f1)"
  echo "roundtrip=exact"
} | tee "$result_root/final.txt"
cp -f selected.fx4 "$result_root/selected.fx4"
sha256sum "$result_root"/* > "$result_root/SHA256SUMS"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

echo "Warm portfolio pilot completed: $result_root"
