#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENWIK9 CAUSAL_PLAN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4donor_discovery
readonly input_path="$(readlink -f "$1")"
readonly plan_path="$(readlink -f "$2")"
readonly result_root="$source_root/research/donor_full_actual_20260727"
readonly cpu="${FX4_CPU:-7}"

test -f "$input_path"
test -f "$plan_path"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "expected exact 1,000,000,000-byte enwik9" >&2
  exit 2
fi

resolved="$(readlink -m "$work_root")"
if [[ "$resolved" != /root/fx4donor_discovery ]]; then
  echo "refusing unexpected work path: $resolved" >&2
  exit 2
fi

pkill -9 -x cmix 2>/dev/null || true
rm -rf -- "$work_root"
mkdir -p "$work_root/source" "$work_root/baseline" "$work_root/assisted"
mkdir -p "$result_root"

cp -a "$source_root/src" "$work_root/source/src"
cp -a "$source_root/dictionary" "$work_root/source/dictionary"
cp -a "$source_root/install_tools" "$work_root/source/install_tools"
cp -a "$source_root/tools" "$work_root/source/tools"
cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh" \
   "$source_root/LICENSE" "$work_root/source/"
if [[ ! -x "$work_root/source/tools/upx" ]]; then
  cp "$(command -v upx)" "$work_root/source/tools/upx"
fi
sed -i 's/\r$//' "$work_root/source/makefile" \
    "$work_root/source/build_and_construct_comp.sh" \
    "$work_root/source/tools/"*.sh
chmod +x "$work_root/source/build_and_construct_comp.sh" \
    "$work_root/source/tools/upx"

cd "$work_root/source"
DONOR=1 ./build_and_construct_comp.sh
cp run/cmix "$work_root/cmix"
cp "$input_path" "$work_root/enwik9"
cp "$plan_path" "$work_root/trial.f4cp"

run_one() {
  local name="$1"
  local plan="$2"
  local directory="$work_root/$name"
  cp "$work_root/cmix" "$directory/cmix"
  cp "$work_root/enwik9" "$directory/enwik9"
  cd "$directory"
  rm -f archive9 payload payload.cmix.temp ppm.temp regions.csv run.log time.txt
  if [[ -n "$plan" ]]; then
    /usr/bin/time -v -o time.txt \
      env FX4_DONOR_PLAN="$plan" FX4_REGION_STATS=regions.csv \
      nice -n -20 taskset -c "$cpu" ./cmix -e enwik9 payload \
      2>&1 | tee run.log
  else
    /usr/bin/time -v -o time.txt \
      env FX4_REGION_STATS=regions.csv \
      nice -n -20 taskset -c "$cpu" ./cmix -e enwik9 payload \
      2>&1 | tee run.log
  fi
  test -s archive9
  test -s payload
  test -s regions.csv
  cp archive9 "$result_root/$name.archive9"
  cp payload "$result_root/$name.payload"
  cp regions.csv "$result_root/$name.regions.csv"
  cp run.log "$result_root/$name.log"
  cp time.txt "$result_root/$name.time.txt"
  rm -f ppm.temp payload.cmix.temp
}

run_one baseline ""
run_one assisted "$work_root/trial.f4cp"

python3 "$work_root/source/tools/select_actual_donor_winners.py" \
  "$work_root/trial.f4cp" \
  "$result_root/baseline.regions.csv" \
  "$result_root/assisted.regions.csv" \
  "$result_root/actual_winners.f4cp" \
  --winners-csv "$result_root/actual_winners.csv" \
  --minimum-gain 6 \
  | tee "$result_root/winner_summary.json"

baseline_payload="$(stat -c%s "$result_root/baseline.payload")"
assisted_payload="$(stat -c%s "$result_root/assisted.payload")"
baseline_archive="$(stat -c%s "$result_root/baseline.archive9")"
assisted_archive="$(stat -c%s "$result_root/assisted.archive9")"
s1_bytes="$(stat -c%s "$work_root/cmix")"

{
  echo "s1_bytes=$s1_bytes"
  echo "baseline_payload_bytes=$baseline_payload"
  echo "assisted_payload_bytes=$assisted_payload"
  echo "payload_gain_bytes=$((baseline_payload-assisted_payload))"
  echo "baseline_archive9_bytes=$baseline_archive"
  echo "assisted_archive9_bytes=$assisted_archive"
  echo "archive9_gain_bytes=$((baseline_archive-assisted_archive))"
  echo "trial_plan_bytes=$(stat -c%s "$work_root/trial.f4cp")"
  echo "winner_plan_bytes=$(stat -c%s "$result_root/actual_winners.f4cp")"
  echo "validation=paired_full_stream_pathwise_measurement"
  echo "next_step=rerun_actual_winners_plan_to_remove_interaction_false_positives"
} | tee "$result_root/overall.txt"

sha256sum "$result_root"/*.archive9 "$result_root"/*.payload \
  "$result_root"/*.f4cp > "$result_root/SHA256SUMS"

echo "Donor discovery results: $result_root"
