#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENWIK9 [RESULT_DIRECTORY]" >&2
  exit 2
fi

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly input_path="$(readlink -f "$1")"
readonly result_root="${2:-$source_root/results/selective_discovery_winner_first}"
readonly work_root=/root/fx4_selective_discovery
readonly candidate_plan="$source_root/results/donor_multiday_seed/all_ranked.f4cd"
readonly recipients_csv="$source_root/results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv"
readonly ranked_csv="$source_root/results/donor_multiday_seed/enwik9.grouped_1mib_candidates.csv"
readonly cpu="${FX4_DONOR_CPU:-7}"
readonly nice_value="${FX4_NICE:-0}"
readonly post_r1_stream="$work_root/enwik9.post_r1.bin"
readonly individual_ledger="$result_root/search.individual"
readonly donor_ledger="$result_root/search.donor_beam"
readonly combine_ledger="$result_root/search.combine"

test -f "$input_path"
test -f "$candidate_plan"
test -f "$recipients_csv"
test -f "$ranked_csv"
if [[ "$(stat -c%s "$input_path")" -ne 1000000000 ]]; then
  echo "expected exact 1,000,000,000-byte enwik9" >&2
  exit 2
fi
if [[ "$(readlink -m "$work_root")" != /root/fx4_selective_discovery ]]; then
  echo "refusing unexpected work path" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
exec 9>"$work_root/discovery.lock"
if ! flock -n 9; then
  echo "another selective discovery process owns $work_root" >&2
  exit 3
fi

source_hash="$(
  cd "$source_root"
  find src tools dictionary -type f -print0 |
    sort -z |
    xargs -0 sha256sum
  sha256sum makefile build_and_construct_comp.sh
)"
source_hash="$(printf '%s' "$source_hash" | sha256sum | cut -d' ' -f1)"

rebuild=1
if [[ -x "$work_root/cmix" && -f "$work_root/source.sha256" ]] &&
   [[ "$(cat "$work_root/source.sha256")" == "$source_hash" ]]; then
  rebuild=0
fi

if [[ "$rebuild" == 1 ]]; then
  rm -rf -- "$work_root/source"
  mkdir -p "$work_root/source"
  cp -a "$source_root/src" "$source_root/dictionary"     "$source_root/install_tools" "$source_root/tools" "$work_root/source/"
  cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh"     "$source_root/LICENSE" "$work_root/source/"
  if [[ ! -x "$work_root/source/tools/upx" ]]; then
    cp "$(command -v upx)" "$work_root/source/tools/upx"
  fi
  sed -i 's/$//' "$work_root/source/makefile"     "$work_root/source/build_and_construct_comp.sh"     "$work_root/source/tools/"*.sh
  cd "$work_root/source"
  DONOR=1 POSTR1=1 MINI_CMIX=1 SHADOW_LSTM200=1 VIRTUAL_REPLAY=1 TOPOLOGY=1 CAUSAL_CNN=1     DONOR_DISCOVERY=1 ./build_and_construct_comp.sh |
    tee "$result_root/build.log"
  cp run/cmix "$work_root/cmix"
  printf '%s
' "$source_hash" >"$work_root/source.sha256"
fi

cp "$work_root/cmix" "$result_root/cmix_selective_discovery_s1"
chmod 0755 "$result_root/cmix_selective_discovery_s1"

if [[ "${FX4_DISCOVERY_BUILD_ONLY:-0}" == 1 ]]; then
  echo "Selective discovery S1 built: $result_root/cmix_selective_discovery_s1"
  exit 0
fi

if [[ ! -f "$work_root/enwik9" ]] ||
   [[ "$(stat -c%s "$work_root/enwik9")" -ne 1000000000 ]]; then
  cp "$input_path" "$work_root/enwik9"
fi

metadata="$result_root/run.meta"
metadata_new="$result_root/run.meta.new"
readonly discovery_semantics="winner_first_v2_entropy_1"
{
  echo "architecture=winner_first_v2"
  echo "discovery_semantics=$discovery_semantics"
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "candidate_plan_sha256=$(sha256sum "$candidate_plan" | cut -d' ' -f1)"
  echo "recipients_sha256=$(sha256sum "$recipients_csv" | cut -d' ' -f1)"
  echo "ranked_sha256=$(sha256sum "$ranked_csv" | cut -d' ' -f1)"
  echo "planned_donors=${FX4_WINNER_PLANNED_DONORS:-7}"
  echo "beam_width=${FX4_WINNER_PORTFOLIO_BEAM_WIDTH:-8}"
  echo "donor_atoms=${FX4_WINNER_PORTFOLIO_DONOR_ATOMS:-24}"
  echo "max_candidates=${FX4_WINNER_PORTFOLIO_MAX_CANDIDATES:-48}"
  echo "max_depth=${FX4_WINNER_PORTFOLIO_MAX_DEPTH:-8}"
} >"$metadata_new"
if [[ -f "$metadata" ]] &&
   grep -qx 'architecture=winner_first_v2' "$metadata" &&
   ! grep -q '^discovery_semantics=' "$metadata"; then
  # One-time migration for the output-neutral replay-accounting repair.
  metadata_migrated="$result_root/run.meta.migrated"
  awk -v semantics="$discovery_semantics" '
    /^architecture=/ {
      print
      print "discovery_semantics=" semantics
      next
    }
    /^discovery_binary_sha256=/ { next }
    { print }
  ' "$metadata" >"$metadata_migrated"
  mv -f "$metadata_migrated" "$metadata"
fi
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing ledger belongs to a different input, build, or matrix" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

cd "$work_root"

common_environment=(
  "MALLOC_ARENA_MAX=1"
  "MALLOC_TRIM_THRESHOLD_=131072"
  "FX4_DONOR_PLAN=$candidate_plan"
  "FX4_DONOR_WINNER_SEARCH=1"
  "FX4_DONOR_TRIAL_CPU=$cpu"
  "FX4_WINNER_PACKS_CSV=$recipients_csv"
  "FX4_WINNER_PAGE_CANDIDATES_CSV=$ranked_csv"
  "FX4_WINNER_START_REGION=${FX4_WINNER_START_REGION:-0}"
  "FX4_WINNER_MAX_REGIONS=${FX4_WINNER_MAX_REGIONS:-0}"
  "FX4_DONOR_MAX_NEW_TRIALS=${FX4_DONOR_MAX_NEW_TRIALS:-0}"
  "FX4_WINNER_QUICK_SINGLES=1"
  "FX4_WINNER_PLANNED_DONORS=${FX4_WINNER_PLANNED_DONORS:-7}"
  "FX4_WINNER_PORTFOLIO_BEAM_WIDTH=${FX4_WINNER_PORTFOLIO_BEAM_WIDTH:-8}"
  "FX4_WINNER_PORTFOLIO_DONOR_ATOMS=${FX4_WINNER_PORTFOLIO_DONOR_ATOMS:-24}"
  "FX4_WINNER_PORTFOLIO_MAX_CANDIDATES=${FX4_WINNER_PORTFOLIO_MAX_CANDIDATES:-48}"
  "FX4_WINNER_PORTFOLIO_MAX_DEPTH=${FX4_WINNER_PORTFOLIO_MAX_DEPTH:-8}"
  "FX4_WINNER_CONTEXT_MIXER=0"
  "FX4_WINNER_NEAR_BYTES=${FX4_WINNER_NEAR_BYTES:-64}"
  "FX4_WINNER_SCR2_COST=1"
  "FX4_WINNER_SCR2_COST_ONLY=${FX4_WINNER_SCR2_COST_ONLY:-1}"
  "FX4_WINNER_TOPOLOGY=0"
  "FX4_WINNER_CAUSAL_CNN=0"
  "FX4_WINNER_LEAVE_ONE_OUT=0"
)

run_phase() {
  local phase="$1"
  local ledger="$2"
  shift 2

  if [[ -f "$ledger.winner.complete" ]]; then
    if [[ "$phase" != individual || -s "$post_r1_stream" ]]; then
      echo "phase already complete: $phase"
      return 0
    fi
    rm -f "$ledger.winner.complete"
  fi

  local payload="discovery_payload_$phase"
  rm -f "$payload" "$payload.cmix.temp" ppm.temp
  local -a phase_environment=(
    "${common_environment[@]}"
    "FX4_DONOR_DISCOVERY_RESULTS=$ledger"
    "FX4_WINNER_PORTFOLIO_PHASE=$phase"
  )
  while [[ $# -gt 0 ]]; do
    phase_environment+=("$1")
    shift
  done

  echo
  echo "FX4 exact selective discovery phase: $phase"
  echo "  GPU: not used"
  echo "  CPU: one pinned core ($cpu)"
  echo "  warm baseline: rebuilt from raw enwik9"
  echo "  durable ledger: $ledger"

  set +e
  /usr/bin/time -v -o "$result_root/$phase.time.txt"     env "${phase_environment[@]}"     nice -n "$nice_value" taskset -c "$cpu"     ./cmix -e enwik9 "$payload" 2>&1 |
    tee -a "$result_root/$phase.log"
  local status=${PIPESTATUS[0]}
  set -e
  rm -f "$payload" "$payload.cmix.temp" ppm.temp

  if [[ "$status" -ne 0 ]]; then
    echo "phase $phase stopped with status $status" >&2
    echo "completed recipient rows are durable; rerun this command" >&2
    exit "$status"
  fi
  if [[ ! -f "$ledger.winner.complete" ]]; then
    if [[ -f "$ledger.winner.paused" ]]; then
      echo "phase $phase paused cleanly; rerun this command"
      exit 0
    fi
    echo "phase $phase exited without a completion marker" >&2
    exit 1
  fi
}

echo "FX4 winner-first post-R1 discovery"
echo "  pass A: exact cost-positive SCR2 catch-up (default)"
echo "  optional pass B/C: legacy donor and combination beams when SCR2_COST_ONLY=0"
echo "  prior topology/CNN/LSTM-200/mini/URL trials are retained, not rerun"
echo "  trial rule: standalone side cost is recorded conservatively for every action"
echo "  rejected branches never mutate PPMd/LSTM-170/FXCM"
echo "  topology: one reusable causal profile; no embedded graph or donor bytes"
echo "  causal CNN: deterministic online state; no weights or checkpoints in side data"
echo "  final plan: profiles are shared and compact span costs are recalculated"
echo "  final S1: discovery engine and unselected experts are compile-time removed"
run_phase individual "$individual_ledger"   "FX4_VR_DUMP_INPUT=$post_r1_stream"

test -f "$individual_ledger.portfolio_trials.csv"
test -s "$post_r1_stream"
winner_ledger="$individual_ledger"
if [[ "${FX4_WINNER_SCR2_COST_ONLY:-1}" != 1 ]]; then
  run_phase donor_beam "$donor_ledger"     "FX4_WINNER_INDIVIDUAL_TRIALS=$individual_ledger.portfolio_trials.csv"
  test -f "$donor_ledger.portfolio_trials.csv"
  run_phase combine "$combine_ledger"     "FX4_WINNER_INDIVIDUAL_TRIALS=$individual_ledger.portfolio_trials.csv"     "FX4_WINNER_DONOR_TRIALS=$donor_ledger.portfolio_trials.csv"
  winner_ledger="$combine_ledger"
fi

baseline_s1=0
discovery_s1="$(stat -c%s "$work_root/cmix")"
candidate_s1=0
release_s1="$source_root/../fx4-cmix/run/cmix"
if [[ -f "$release_s1" ]]; then
  baseline_s1="$(stat -c%s "$release_s1")"
fi
if [[ -n "${FX4_SELECTED_RELEASE_S1:-}" ]]; then
  test -f "$FX4_SELECTED_RELEASE_S1"
  candidate_s1="$(stat -c%s "$FX4_SELECTED_RELEASE_S1")"
fi
trial_ledger_args=(
  --portfolio-trials-csv "$individual_ledger.portfolio_trials.csv"
)
if [[ -f "$donor_ledger.portfolio_trials.csv" ]]; then
  trial_ledger_args+=(
    --portfolio-trials-csv "$donor_ledger.portfolio_trials.csv"
  )
fi
if [[ -f "$combine_ledger.portfolio_trials.csv" ]]; then
  trial_ledger_args+=(
    --portfolio-trials-csv "$combine_ledger.portfolio_trials.csv"
  )
fi
python3 "$work_root/source/tools/build_selective_winner_plan.py" \
  "$candidate_plan" \
  "$winner_ledger.portfolio_selected.csv" \
  "$result_root/current_winners.f4cp" \
  --post-r1-stream "$post_r1_stream" \
  --output-vr-plan "$result_root/current_winners.f4vr" \
  --vr-events-csv "$individual_ledger.portfolio_vr_events.csv" \
  "${trial_ledger_args[@]}" \
  --discovery-s1-bytes "$discovery_s1" \
  --baseline-s1-bytes "$baseline_s1" \
  --candidate-s1-bytes "$candidate_s1" |
  tee "$result_root/current_winners_summary.json"

sha256sum   "$individual_ledger".portfolio_*.csv   "$donor_ledger".portfolio_*.csv   "$combine_ledger".portfolio_*.csv   "$result_root/current_winners.f4cp"   "$result_root/current_winners.f4vr"   "$result_root/cmix_selective_discovery_s1"   2>/dev/null >"$result_root/SHA256SUMS" || true

echo
echo "winner-first discovery completed: $result_root"
echo "Final production validation must use both current_winners.f4cp and"
echo "current_winners.f4vr when the summary reports virtual-replay events."
