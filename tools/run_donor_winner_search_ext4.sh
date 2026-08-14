#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENWIK9 F4CD_CANDIDATE_PLAN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4winner_search
readonly input_path="$(readlink -f "$1")"
readonly plan_path="$(readlink -f "$2")"
readonly result_root="${FX4_DONOR_RESULT_ROOT:-$source_root/research/donor_winner_search_20260731}"
readonly ledger="$result_root/search"
readonly cpu="${FX4_DONOR_CPU:-7}"
readonly trial_cpu="${FX4_DONOR_TRIAL_CPU:-7}"

test -f "$input_path"
test -f "$plan_path"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "expected exact 1,000,000,000-byte enwik9" >&2
  exit 2
fi
if [[ "$(readlink -m "$work_root")" != /root/fx4winner_search ]]; then
  echo "refusing unexpected work path" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
exec 9>"$work_root/search.lock"
if ! flock -n 9; then
  echo "another winner-search process owns $work_root" >&2
  exit 3
fi

rm -rf -- "$work_root/source"
mkdir -p "$work_root/source"
cp -a "$source_root/src" "$source_root/dictionary" \
  "$source_root/install_tools" "$source_root/tools" "$work_root/source/"
cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh" \
  "$source_root/LICENSE" "$work_root/source/"
if [[ ! -x "$work_root/source/tools/upx" ]]; then
  cp "$(command -v upx)" "$work_root/source/tools/upx"
fi
sed -i 's/\r$//' "$work_root/source/makefile" \
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

readonly metadata="$result_root/run.meta"
readonly metadata_new="$result_root/run.meta.new"
{
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "candidate_plan_sha256=$(sha256sum "$plan_path" | cut -d' ' -f1)"
  echo "discovery_binary_sha256=$(sha256sum "$work_root/cmix" | cut -d' ' -f1)"
} > "$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing winner ledger belongs to another input, plan, or binary" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

cd "$work_root"
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

echo "Loss-guided exact donor winner search"
echo "  input: $input_path"
echo "  candidates: $plan_path"
echo "  durable ledger base: $ledger"
echo "  compression CPU: $cpu"
echo "  trial CPU: $trial_cpu"
echo "  max new trials: ${FX4_DONOR_MAX_NEW_TRIALS:-0}"
echo "  max new regions: ${FX4_WINNER_MAX_REGIONS:-0}"
echo "  stop offset: ${FX4_WINNER_STOP_OFFSET:-full}"
echo "  grouped recipients: ${FX4_WINNER_PACKS_CSV:-default 1MiB}"
echo "  ranked candidates: ${FX4_WINNER_PAGE_CANDIDATES_CSV:-internal}"
echo "  recipient filter: ${FX4_WINNER_RECIPIENTS_CSV:-all}"
echo "  quick singles: ${FX4_WINNER_QUICK_SINGLES:-0}"
echo "  donor strength: ${FX4_WINNER_DONOR_STRENGTH:-1}"
echo "  context mixer: ${FX4_WINNER_CONTEXT_MIXER:-0}"
echo "  incremental metadata: ${FX4_WINNER_METADATA_BYTES:-29} bytes"

set +e
/usr/bin/time -v -o "$result_root/search.time.txt" \
  env FX4_DONOR_PLAN="$work_root/candidates.f4cd" \
      FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
      FX4_DONOR_WINNER_SEARCH=1 \
      FX4_DONOR_TRIAL_CPU="$trial_cpu" \
      FX4_DONOR_MAX_NEW_TRIALS="${FX4_DONOR_MAX_NEW_TRIALS:-0}" \
      FX4_WINNER_MAX_REGIONS="${FX4_WINNER_MAX_REGIONS:-0}" \
      FX4_WINNER_START_REGION="${FX4_WINNER_START_REGION:-1}" \
      FX4_WINNER_STOP_OFFSET="${FX4_WINNER_STOP_OFFSET:-18446744073709551615}" \
      FX4_WINNER_PACKS_CSV="${FX4_WINNER_PACKS_CSV:-}" \
      FX4_WINNER_PAGE_CANDIDATES_CSV="${FX4_WINNER_PAGE_CANDIDATES_CSV:-}" \
      FX4_WINNER_RECIPIENTS_CSV="${FX4_WINNER_RECIPIENTS_CSV:-}" \
      FX4_WINNER_RECIPIENT_LIMIT="${FX4_WINNER_RECIPIENT_LIMIT:-0}" \
      FX4_WINNER_TOP_SPANS="${FX4_WINNER_TOP_SPANS:-8}" \
      FX4_WINNER_CANDIDATES="${FX4_WINNER_CANDIDATES:-12}" \
      FX4_WINNER_REFINE_OFFSETS="${FX4_WINNER_REFINE_OFFSETS:-4}" \
      FX4_WINNER_LOCAL_SEEDS="${FX4_WINNER_LOCAL_SEEDS:-2}" \
      FX4_WINNER_LOCAL_RADIUS="${FX4_WINNER_LOCAL_RADIUS:-1024}" \
      FX4_WINNER_COMBO_CANDIDATES="${FX4_WINNER_COMBO_CANDIDATES:-8}" \
      FX4_WINNER_BEAM_WIDTH="${FX4_WINNER_BEAM_WIDTH:-4}" \
      FX4_WINNER_MAX_DEPTH="${FX4_WINNER_MAX_DEPTH:-8}" \
      FX4_WINNER_NEAR_BYTES="${FX4_WINNER_NEAR_BYTES:-64}" \
      FX4_WINNER_LEAVE_ONE_OUT="${FX4_WINNER_LEAVE_ONE_OUT:-1}" \
      FX4_WINNER_PLANNED_DONORS="${FX4_WINNER_PLANNED_DONORS:-7}" \
      FX4_WINNER_PLANNED_ONLY="${FX4_WINNER_PLANNED_ONLY:-0}" \
      FX4_WINNER_PLANNED_PREFIXES="${FX4_WINNER_PLANNED_PREFIXES:-1}" \
      FX4_WINNER_QUICK_SINGLES="${FX4_WINNER_QUICK_SINGLES:-0}" \
      FX4_WINNER_DONOR_STRENGTH="${FX4_WINNER_DONOR_STRENGTH:-1}" \
      FX4_WINNER_CONTEXT_MIXER="${FX4_WINNER_CONTEXT_MIXER:-0}" \
      FX4_WINNER_METADATA_BYTES="${FX4_WINNER_METADATA_BYTES:-29}" \
  nice -n -10 taskset -c "$cpu" \
  ./cmix -e enwik9 discovery_payload \
  2>&1 | tee -a "$result_root/search.log"
status=${PIPESTATUS[0]}
set -e

if [[ -f "$ledger.winner_selected.csv" ]]; then
  python3 "$work_root/source/tools/build_winner_plan.py" \
    "$work_root/candidates.f4cd" "$ledger.winner_selected.csv" \
    "$result_root/current_winners.f4cp" \
    | tee "$result_root/current_winners_summary.json"
fi
if [[ -f "$ledger.winner_marginals.csv" ]] && [[ $(wc -l < "$ledger.winner_marginals.csv") -gt 1 ]]; then
  bash "$work_root/source/tools/find_multi_recipient_donors.sh" \
    "$ledger.winner_marginals.csv" 0 2 \
    >"$result_root/multi_recipient_donors.csv"
fi

sha256sum "$ledger".winner* "$result_root/current_winners.f4cp" \
  2>/dev/null > "$result_root/SHA256SUMS" || true
rm -f discovery_payload discovery_payload.cmix.temp ppm.temp

if [[ "$status" -ne 0 ]]; then
  echo "winner search failed with status $status" >&2
  exit "$status"
fi
if [[ -f "$ledger.winner.complete" ]]; then
  echo "winner search completed: $result_root"
elif [[ -f "$ledger.winner.paused" ]]; then
  echo "winner search paused cleanly; rerun the same command to resume"
else
  echo "winner search exited without a completion marker" >&2
  exit 1
fi
