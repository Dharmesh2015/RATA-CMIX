#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENWIK9 F4CD_CANDIDATE_PLAN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4donor_mesh
readonly input_path="$(readlink -f "$1")"
readonly plan_path="$(readlink -f "$2")"
readonly result_root="${FX4_DONOR_RESULT_ROOT:-$source_root/research/donor_exact_mesh_20260729}"
readonly cpu="${FX4_DONOR_CPU:-7}"
readonly trial_cpu="${FX4_DONOR_TRIAL_CPU:-7}"
readonly max_new_trials="${FX4_DONOR_MAX_NEW_TRIALS:-0}"

test -f "$input_path"
test -f "$plan_path"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "expected exact 1,000,000,000-byte enwik9" >&2
  exit 2
fi

resolved="$(readlink -m "$work_root")"
if [[ "$resolved" != /root/fx4donor_mesh ]]; then
  echo "refusing unexpected work path: $resolved" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
rm -rf -- "$work_root/source"
mkdir -p "$work_root/source"
cp -a "$source_root/src" "$work_root/source/src"
cp -a "$source_root/dictionary" "$work_root/source/dictionary"
cp -a "$source_root/install_tools" "$work_root/source/install_tools"
cp -a "$source_root/tools" "$work_root/source/tools"
cp "$source_root/Makefile" "$source_root/build_and_construct_comp.sh" \
   "$source_root/LICENSE" "$work_root/source/"
if [[ ! -x "$work_root/source/tools/upx" ]]; then
  cp "$(command -v upx)" "$work_root/source/tools/upx"
fi
sed -i 's/\r$//' "$work_root/source/Makefile" \
    "$work_root/source/build_and_construct_comp.sh" \
    "$work_root/source/tools/"*.sh
cd "$work_root/source"
DONOR_DISCOVERY=1 ./build_and_construct_comp.sh
cp run/cmix "$work_root/cmix"

if [[ ! -f "$work_root/enwik9" ]] || \
   [[ "$(stat -c%s "$work_root/enwik9")" -ne 1000000000 ]]; then
  cp "$input_path" "$work_root/enwik9"
fi
cp -f "$plan_path" "$work_root/candidates.f4cd"

readonly ledger="$result_root/edges.csv"
readonly selected="$ledger.selected.csv"
readonly log="$result_root/discovery.log"
readonly timing="$result_root/discovery.time.txt"
readonly metadata="$result_root/run.meta"
readonly metadata_new="$result_root/run.meta.new"

{
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "candidate_plan_sha256=$(sha256sum "$plan_path" | cut -d' ' -f1)"
  echo "discovery_binary_sha256=$(sha256sum "$work_root/cmix" | cut -d' ' -f1)"
} > "$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "Existing ledger belongs to a different input, plan, or binary:" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

cd "$work_root"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

echo "Exact donor mesh discovery"
echo "  input: $input_path"
echo "  candidates: $plan_path"
echo "  durable edge ledger: $ledger"
echo "  every positive edge: $ledger.winners.csv"
echo "  advancing selections: $selected"
echo "  maximum new trials this invocation: $max_new_trials (0 = unlimited)"
echo "  compression CPU: $cpu"
echo "  trial CPU: $trial_cpu"

set +e
/usr/bin/time -v -o "$timing" \
  env FX4_DONOR_PLAN="$work_root/candidates.f4cd" \
      FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
      FX4_DONOR_TRIAL_CPU="$trial_cpu" \
      FX4_DONOR_MAX_NEW_TRIALS="$max_new_trials" \
  nice -n -10 taskset -c "$cpu" \
  ./cmix -e enwik9 discovery_payload \
  2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e

if [[ -f "$selected" ]]; then
  python3 "$work_root/source/tools/build_selected_donor_plan.py" \
    "$work_root/candidates.f4cd" "$selected" \
    "$result_root/current_selected.f4cp" \
    | tee "$result_root/current_selected_summary.json"
fi

sha256sum "$ledger"* "$result_root/current_selected.f4cp" \
  2>/dev/null > "$result_root/SHA256SUMS" || true
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

if [[ "$status" -ne 0 ]]; then
  echo "Discovery process failed with status $status" >&2
  exit "$status"
fi
if [[ -f "$ledger.complete" ]]; then
  echo "Exact donor mesh completed: $result_root"
elif [[ -f "$ledger.paused" ]]; then
  echo "Exact donor mesh paused cleanly; rerun the same command to resume."
else
  echo "Discovery exited without a complete or paused marker." >&2
  exit 1
fi
