#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENWIK9 CAUSAL_TRIAL_PLAN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4donor_exact
readonly input_path="$(readlink -f "$1")"
readonly plan_path="$(readlink -f "$2")"
readonly result_root="$source_root/research/donor_exact_full_20260727"
readonly donor_cpu="${FX4_DONOR_CPU:-7}"
readonly baseline_cpu="${FX4_DONOR_BASELINE_CPU:-8}"

test -f "$input_path"
test -f "$plan_path"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "expected exact 1,000,000,000-byte enwik9" >&2
  exit 2
fi

resolved="$(readlink -m "$work_root")"
if [[ "$resolved" != /root/fx4donor_exact ]]; then
  echo "refusing unexpected work path: $resolved" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
if [[ ! -x "$work_root/cmix" ]]; then
  rm -rf -- "$work_root/source"
  mkdir -p "$work_root/source"
  cp -a "$source_root/." "$work_root/source/"
  sed -i 's/\r$//' "$work_root/source/Makefile" \
      "$work_root/source/build_and_construct_comp.sh" \
      "$work_root/source/tools/"*.sh
  cd "$work_root/source"
  DONOR_DISCOVERY=1 ./build_and_construct_comp.sh
  cp run/cmix "$work_root/cmix"
fi

cp -f "$input_path" "$work_root/enwik9"
cp -f "$plan_path" "$work_root/trial.f4cp"
cd "$work_root"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

readonly ledger="$result_root/trials.csv"
readonly log="$result_root/discovery.log"
readonly timing="$result_root/discovery.time.txt"

echo "Exact donor discovery"
echo "  input: $input_path"
echo "  trial plan: $plan_path"
echo "  all decisions: $ledger"
echo "  positive decisions: $ledger.winners.csv"
echo "  resume: existing decisions are replayed without retesting"
echo "  donor CPU: $donor_cpu"
echo "  baseline CPU: $baseline_cpu"

/usr/bin/time -v -o "$timing" \
  env FX4_DONOR_PLAN="$work_root/trial.f4cp" \
      FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
      FX4_DONOR_BASELINE_CPU="$baseline_cpu" \
  nice -n -10 taskset -c "$donor_cpu" \
  ./cmix -e enwik9 discovery_payload \
  2>&1 | tee -a "$log"

python3 "$work_root/source/tools/build_exact_donor_plan.py" \
  "$work_root/trial.f4cp" "$ledger.winners.csv" \
  "$result_root/exact_winners.f4cp" \
  | tee "$result_root/exact_winner_summary.txt"

sha256sum "$ledger" "$ledger.winners.csv" \
  "$result_root/exact_winners.f4cp" > "$result_root/SHA256SUMS"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp
echo "Completed: $result_root"
