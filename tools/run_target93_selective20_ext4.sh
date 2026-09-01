#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENWIK9 [RESULT_DIRECTORY]" >&2
  exit 2
fi

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly input_path="$(readlink -f "$1")"
readonly result_root="${2:-$source_root/results/target93_selective20}"
readonly work_root=/root/fx4_target93_selective20
readonly cpu="${FX4_TEST_CPU:-7}"
readonly jobs="${FX4_BUILD_JOBS:-3}"
readonly model_rel=models/transformer6m/6m-q4-fp32.tfwc2
readonly candidate_plan_rel=results/donor_multiday_seed/all_ranked.f4cd
readonly recipients_rel=results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv
readonly candidates_rel=results/donor_multiday_seed/enwik9.grouped_1mib_candidates.csv

test -f "$input_path"
test "$(stat -c%s "$input_path")" -eq 1000000000
test -s "$source_root/$model_rel"
test -s "$source_root/$candidate_plan_rel"
test -s "$source_root/$recipients_rel"
test -s "$source_root/$candidates_rel"
if [[ "$(readlink -m "$work_root")" != /root/fx4_target93_selective20 ]]; then
  echo "refusing unexpected work path" >&2
  exit 2
fi

mkdir -p "$work_root" "$result_root"
exec 9>"$work_root/run.lock"
flock -n 9 || {
  echo "another target93 selective-20 run owns $work_root" >&2
  exit 3
}

readonly first20_recipients="$work_root/first20.recipients.csv"
readonly first20_candidates="$work_root/first20.candidates.csv"
awk -F, 'NR == 1 || ($1 + 0 >= 0 && $1 + 0 < 20)' \
  "$source_root/$recipients_rel" >"$first20_recipients"
awk -F, 'NR == 1 || ($1 + 0 >= 0 && $1 + 0 < 20)' \
  "$source_root/$candidates_rel" >"$first20_candidates"
test "$(wc -l <"$first20_recipients")" -eq 21
readonly prefix_end="$(
  awk -F, 'NR > 1 && $1 == 19 {print $5; exit}' "$first20_recipients"
)"
test "$prefix_end" = 20990874

source_hash="$(
  cd "$source_root"
  {
    find src dictionary models/transformer6m -type f -print0 |
      sort -z | xargs -0 sha256sum
    sha256sum makefile "$candidate_plan_rel" "$recipients_rel" \
      "$candidates_rel"
  } | sha256sum | cut -d' ' -f1
)"

strip_binary() {
  local path="$1"
  if command -v llvm-strip-17 >/dev/null 2>&1; then
    llvm-strip-17 --strip-all "$path"
  else
    strip --strip-all "$path"
  fi
}

build_core() {
  local goal="$1"
  local output="$2"
  if [[ -x "$work_root/$output" ]]; then
    return
  fi
  cd "$work_root/source"
  make clean
  make "$goal" -j"$jobs" OUT="$output"
  strip_binary "$output"
  cp "$output" "$work_root/$output"
}

package_discovery() {
  local package_root="$work_root/package"
  mkdir -p "$package_root"
  cp "$work_root/cmix_discovery20_core" "$package_root/cmix_orig"
  cp "$work_root/source/dictionary/english.dic" "$package_root/english.dic"
  cp "$work_root/source/src/readalike_prepr/data/new_article_order" \
    "$package_root/article_order"
  cp "$work_root/source/$model_rel" "$package_root/transformer6m.weights"
  cd "$package_root"
  export FX4_TRANSFORMER_WEIGHTS="$package_root/transformer6m.weights"
  rm -f comp_dict comp_order header.dat ppm.temp cmix
  ./cmix_orig -c english.dic comp_dict
  rm -f ppm.temp
  ./cmix_orig -c article_order comp_order
  rm -f ppm.temp
  local dict_size order_size model_size
  dict_size="$(stat -c%s comp_dict)"
  order_size="$(stat -c%s comp_order)"
  model_size="$(stat -c%s transformer6m.weights)"
  ./cmix_orig -h "$dict_size" "$order_size" 0 "$model_size"
  cat cmix_orig comp_dict comp_order transformer6m.weights header.dat >cmix
  chmod 0755 cmix
  cp cmix "$work_root/cmix_discovery20"
}

cached_source_hash="$(cat "$work_root/source.sha256" 2>/dev/null || true)"
if [[ ! -x "$work_root/cmix_discovery20" ||
      ! -f "$work_root/source.sha256" ||
      "$cached_source_hash" != "$source_hash" ]]; then
  rm -rf -- "$work_root/source" "$work_root/package"
  rm -f -- "$work_root"/cmix_* "$work_root/source.sha256"
  mkdir -p "$work_root/source"
  cp -a "$source_root/src" "$source_root/dictionary" "$source_root/models" \
    "$source_root/makefile" "$work_root/source/"
  sed -i 's/\r$//' "$work_root/source/makefile"
  build_core selective20 cmix_discovery20_core
  package_discovery
  printf '%s\n' "$source_hash" >"$work_root/source.sha256"
fi

cached_input_size="$(stat -c%s "$work_root/enwik9" 2>/dev/null || true)"
if [[ ! -f "$work_root/enwik9" ||
      "$cached_input_size" != 1000000000 ]]; then
  cp "$input_path" "$work_root/enwik9"
fi
cp "$source_root/$candidate_plan_rel" "$work_root/candidates.f4cd"

metadata="$result_root/run.meta"
metadata_new="$result_root/run.meta.new"
{
  echo "architecture=target93_selective20_v1"
  echo "source_sha256=$source_hash"
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "binary_sha256=$(sha256sum "$work_root/cmix_discovery20" | cut -d' ' -f1)"
  echo "plan_sha256=$(sha256sum "$work_root/candidates.f4cd" | cut -d' ' -f1)"
  echo "recipients_sha256=$(sha256sum "$first20_recipients" | cut -d' ' -f1)"
  echo "candidates_sha256=$(sha256sum "$first20_candidates" | cut -d' ' -f1)"
  echo "prefix_end=$prefix_end"
} >"$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing result directory belongs to another build or input" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

run_phase() {
  local phase="$1"
  local ledger="$result_root/search.$phase"
  shift
  if [[ -f "$ledger.winner.complete" ]]; then
    echo "Reusing completed phase: $phase"
    return
  fi

  cd "$work_root"
  rm -f "payload_$phase" "payload_$phase.cmix.temp" ppm.temp
  echo
  echo "Target93 selective first-20 phase: $phase"
  echo "  exact warm prefix: 0..$prefix_end"
  echo "  transformer: frozen FX2 6M replacement"
  echo "  candidates: donor profiles and cost-positive SCR2 virtual replay"
  set +e
  /usr/bin/time -v -o "$result_root/$phase.time.txt" \
    env MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=131072 \
      FX4_TRANSFORMER_WEIGHTS="$work_root/package/transformer6m.weights" \
      FX4_DONOR_PLAN="$work_root/candidates.f4cd" \
      FX4_DONOR_DISCOVERY_RESULTS="$ledger" \
      FX4_DONOR_WINNER_SEARCH=1 \
      FX4_DONOR_TRIAL_CPU="$cpu" \
      FX4_WINNER_PACKS_CSV="$first20_recipients" \
      FX4_WINNER_PAGE_CANDIDATES_CSV="$first20_candidates" \
      FX4_WINNER_START_REGION=0 \
      FX4_WINNER_MAX_REGIONS=20 \
      FX4_WINNER_STOP_OFFSET="$prefix_end" \
      FX4_WINNER_QUICK_SINGLES=1 \
      FX4_WINNER_PLANNED_DONORS=7 \
      FX4_WINNER_PORTFOLIO_BEAM_WIDTH=8 \
      FX4_WINNER_PORTFOLIO_DONOR_ATOMS=24 \
      FX4_WINNER_PORTFOLIO_MAX_CANDIDATES=64 \
      FX4_WINNER_PORTFOLIO_MAX_DEPTH=7 \
      FX4_WINNER_CONTEXT_MIXER=0 \
      FX4_WINNER_NEAR_BYTES=64 \
      FX4_WINNER_SCR2_COST=1 \
      FX4_WINNER_SCR2_COST_ONLY=0 \
      FX4_WINNER_LEAVE_ONE_OUT=0 \
      FX4_WINNER_PORTFOLIO_PHASE="$phase" \
      "$@" \
      taskset -c "$cpu" ./cmix_discovery20 -e enwik9 "payload_$phase" \
      >"$result_root/$phase.log" 2>&1
  status=$?
  set -e
  rm -f "payload_$phase" "payload_$phase.cmix.temp" ppm.temp
  if [[ "$status" -ne 0 ]]; then
    echo "phase $phase failed with status $status" >&2
    echo "completed rows are durable; rerun the same command" >&2
    exit "$status"
  fi
  test -f "$ledger.winner.complete"
}

run_phase individual
run_phase donor_beam \
  "FX4_WINNER_INDIVIDUAL_TRIALS=$result_root/search.individual.portfolio_trials.csv"
run_phase combine \
  "FX4_WINNER_INDIVIDUAL_TRIALS=$result_root/search.individual.portfolio_trials.csv" \
  "FX4_WINNER_DONOR_TRIALS=$result_root/search.donor_beam.portfolio_trials.csv"

winner_csv="$result_root/selected_winners.csv"
awk -F, '
  FNR == 1 {
    if (!printed_header) {
      print
      printed_header = 1
    }
    next
  }
  {
    net = $17 + 0
    region = $1 + 0
    if (net > 0 && (!seen[region] || net > best[region])) {
      seen[region] = 1
      best[region] = net
      row[region] = $0
    }
  }
  END {
    for (region = 0; region < 20; ++region) {
      if (seen[region]) print row[region]
    }
  }
' "$result_root/search.individual.portfolio_selected.csv" \
  "$result_root/search.donor_beam.portfolio_selected.csv" \
  "$result_root/search.combine.portfolio_selected.csv" >"$winner_csv"

regional_net="$(awk -F, 'NR > 1 {sum += $17} END {print sum + 0}' "$winner_csv")"
winner_count="$(awk 'END {print NR > 0 ? NR - 1 : 0}' "$winner_csv")"
needs_donor="$(awk -F, 'NR > 1 && $9 + 0 > 0 {print 1; exit}' "$winner_csv")"
needs_scr2="$(awk -F, 'NR > 1 && $11 + 0 > 0 {print 1; exit}' "$winner_csv")"
needs_donor="${needs_donor:-0}"
needs_scr2="${needs_scr2:-0}"

build_core target93 cmix_target93_core
selected_core=cmix_target93_core
selected_goal=target93
if [[ "$needs_donor" == 1 && "$needs_scr2" == 1 ]]; then
  build_core target93-selective cmix_target93_selective_core
  selected_core=cmix_target93_selective_core
  selected_goal=target93-selective
elif [[ "$needs_donor" == 1 ]]; then
  build_core target93-donor cmix_target93_donor_core
  selected_core=cmix_target93_donor_core
  selected_goal=target93-donor
elif [[ "$needs_scr2" == 1 ]]; then
  build_core target93-scr2 cmix_target93_scr2_core
  selected_core=cmix_target93_scr2_core
  selected_goal=target93-scr2
fi

baseline_core_bytes="$(stat -c%s "$work_root/cmix_target93_core")"
selected_core_bytes="$(stat -c%s "$work_root/$selected_core")"
code_cost=$((selected_core_bytes - baseline_core_bytes))
if (( code_cost < 0 )); then code_cost=0; fi
hutter_net=$((regional_net - code_cost))
promotion=research_only
if (( winner_count > 0 && hutter_net > 0 )); then
  promotion=build_plan_and_roundtrip
fi

{
  echo "groups_tested=20"
  echo "post_r1_prefix_bytes=$prefix_end"
  echo "winner_groups=$winner_count"
  echo "regional_net_bytes=$regional_net"
  echo "selected_release_target=$selected_goal"
  echo "baseline_stripped_core_bytes=$baseline_core_bytes"
  echo "selected_stripped_core_bytes=$selected_core_bytes"
  echo "conservative_s1_code_cost_bytes=$code_cost"
  echo "conservative_hutter_net_bytes=$hutter_net"
  echo "promotion=$promotion"
  echo "note=regional net already includes each action plan side data"
} >"$result_root/summary.txt"

cat "$result_root/summary.txt"
echo "Exact per-group winners: $winner_csv"
