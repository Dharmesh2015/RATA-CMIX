#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 ENWIK9 CAUSAL_PLAN [REGIONS]" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4donor_stage
readonly input_path="$(readlink -f "$1")"
readonly plan_path="$(readlink -f "$2")"
readonly regions="${3:-16}"
readonly result_root="$source_root/research/donor_stage_actual_20260727"
readonly cpu="${FX4_CPU:-7}"

test -f "$input_path"
test -f "$plan_path"
[[ "$(stat -c%s "$input_path")" -eq 1000000000 ]]
[[ "$regions" =~ ^[1-9][0-9]*$ ]]

resolved="$(readlink -m "$work_root")"
[[ "$resolved" == /root/fx4donor_stage ]]
pkill -9 -x cmix 2>/dev/null || true
rm -rf -- "$work_root"
mkdir -p "$work_root/source" "$work_root/baseline" \
    "$work_root/assisted" "$result_root"

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

run_stage() {
  local name="$1"
  local plan="$2"
  local directory="$work_root/$name"
  cp "$work_root/cmix" "$directory/cmix"
  cp "$work_root/enwik9" "$directory/enwik9"
  cd "$directory"
  rm -f payload payload.cmix.temp ppm.temp regions.csv run.log time.txt
  set +e
  if [[ -n "$plan" ]]; then
    /usr/bin/time -v -o time.txt \
      env FX4_DONOR_PLAN="$plan" FX4_REGION_STATS=regions.csv \
          FX4_DONOR_STOP_AFTER_REGIONS="$regions" \
      nice -n -20 taskset -c "$cpu" ./cmix -e enwik9 payload \
      >run.log 2>&1
  else
    /usr/bin/time -v -o time.txt \
      env FX4_REGION_STATS=regions.csv \
          FX4_DONOR_STOP_AFTER_REGIONS="$regions" \
      nice -n -20 taskset -c "$cpu" ./cmix -e enwik9 payload \
      >run.log 2>&1
  fi
  status=$?
  set -e
  if ! grep -q "stopped after $regions regions" run.log; then
    cat run.log >&2
    echo "unexpected staged-run status: $status" >&2
    exit 3
  fi
  test -s regions.csv
  cp regions.csv "$result_root/$name.regions.csv"
  cp run.log "$result_root/$name.log"
  cp time.txt "$result_root/$name.time.txt"
  rm -f ppm.temp payload payload.cmix.temp
}

run_stage baseline ""
run_stage assisted "$work_root/trial.f4cp"

python3 "$work_root/source/tools/select_actual_donor_winners.py" \
  "$work_root/trial.f4cp" \
  "$result_root/baseline.regions.csv" \
  "$result_root/assisted.regions.csv" \
  "$result_root/stage_winners.f4cp" \
  --winners-csv "$result_root/stage_winners.csv" \
  --minimum-gain 6 \
  | tee "$result_root/stage_winner_summary.json"

python3 - "$result_root" "$regions" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
regions = int(sys.argv[2])

def load(name):
    with (root / name).open(newline="") as source:
        return {int(row["region"]): row for row in csv.DictReader(source)}

baseline = load("baseline.regions.csv")
assisted = load("assisted.regions.csv")
rows = []
for region in sorted(set(baseline) & set(assisted)):
    if region >= regions:
        continue
    base = int(baseline[region]["payload_bytes"])
    trial = int(assisted[region]["payload_bytes"])
    rows.append(
        {
            "region": region,
            "donor_seed_offset": int(
                assisted[region]["donor_seed_offset"]
            ),
            "baseline_payload_bytes": base,
            "assisted_payload_bytes": trial,
            "measured_path_gain_bytes": base - trial,
        }
    )
with (root / "stage_comparison.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "regions": regions,
    "baseline_payload_bytes": sum(
        row["baseline_payload_bytes"] for row in rows
    ),
    "assisted_payload_bytes": sum(
        row["assisted_payload_bytes"] for row in rows
    ),
    "path_gain_bytes": sum(
        row["measured_path_gain_bytes"] for row in rows
    ),
    "positive_assigned_regions": sum(
        row["donor_seed_offset"] != 0xFFFFFFFF
        and row["measured_path_gain_bytes"] > 6
        for row in rows
    ),
    "negative_assigned_regions": sum(
        row["donor_seed_offset"] != 0xFFFFFFFF
        and row["measured_path_gain_bytes"] <= 0
        for row in rows
    ),
}
(root / "stage_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
PY

sha256sum "$result_root"/*.f4cp > "$result_root/SHA256SUMS"
echo "Staged donor results: $result_root"
