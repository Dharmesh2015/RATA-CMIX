#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENWIK9 [RESULT_DIRECTORY]" >&2
  exit 2
fi

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly enwik9="$(readlink -f "$1")"
readonly result_root="${2:-$source_root/results/intelligent_20groups_20260829}"
readonly work=/root/fx4_20group_intelligent
readonly cpu="${FX4_TEST_CPU:-7}"
readonly recipients="$source_root/results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv"
readonly full_postr1=/root/fx4_selective_discovery/enwik9.post_r1.bin

test -f "$enwik9"
test "$(stat -c%s "$enwik9")" -eq 1000000000
test -f "$recipients"
test -f "$full_postr1"
test "$(stat -c%s "$full_postr1")" -eq 587138826
mkdir -p "$work" "$result_root"
exec 9>"$work/run.lock"
flock -n 9 || { echo "another 20-group test owns $work" >&2; exit 3; }

prefix_end="$(awk -F, 'NR > 1 && $1 == 19 {print $5; exit}' "$recipients")"
test "$prefix_end" = 20990874

source_hash="$(
  cd "$source_root"
  find src tools dictionary -type f ! -name '*.pyc' \
    ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum
  sha256sum makefile build_and_construct_comp.sh
)"
source_hash="$(printf '%s' "$source_hash" | sha256sum | cut -d' ' -f1)"
if [[ ! -x "$work/cmix" || ! -f "$work/source.sha256" ||
      "$(cat "$work/source.sha256" 2>/dev/null || true)" != "$source_hash" ]]; then
  rm -rf -- "$work/source"
  mkdir -p "$work/source"
  cp -a "$source_root/src" "$source_root/tools" "$source_root/dictionary" \
    "$source_root/install_tools" "$work/source/"
  cp "$source_root/makefile" "$source_root/build_and_construct_comp.sh" \
    "$source_root/LICENSE" "$work/source/"
  sed -i 's/\r$//' "$work/source/makefile" \
    "$work/source/build_and_construct_comp.sh" "$work/source/tools/"*.sh
  cd "$work/source"
  make clean
  make selective -j3 OUT=cmix | tee "$result_root/build.log"
  cp cmix "$work/cmix"
  printf '%s\n' "$source_hash" >"$work/source.sha256"
fi

if [[ ! -f "$work/postr1_g0_g19.bin" ||
      "$(stat -c%s "$work/postr1_g0_g19.bin" 2>/dev/null || true)" != "$prefix_end" ]]; then
  dd if="$full_postr1" of="$work/postr1_g0_g19.bin" bs=4M \
    count="$prefix_end" iflag=count_bytes status=none
fi

readonly input="$work/postr1_g0_g19.bin"
readonly dictionary="$work/source/dictionary/english.dic"
readonly common_experts="oracle+structural+ppmd_escape_order+sparse_virtual_ppm+word_xml_ppm+residual_lstm+micro_diffusion+rare_residual+cts_skipcts+dmc+token_match+context_mixer+mini_cmix+url_structure+shadow_lstm200+topology_recurrence+causal_cnn"

{
  echo 'offset,length,experts,stream_class,profile,residual_gain,pack_id,status'
  awk -F, -v experts="$common_experts" \
    'NR > 1 && $1 < 20 {print $4 "," $6 "," experts ",mixed,0,0.25," $1 ",oracle_only"}' \
    "$recipients"
} >"$work/all_experts.csv"

{
  echo 'offset,length'
  awk -F, 'NR > 1 && $1 < 20 {print $4 "," $6}' "$recipients"
} >"$work/group_ranges.csv"

python3 "$work/source/tools/build_postr1_portfolio.py" "$input" \
  --prefix "$work/all_experts" --expert-csv "$work/all_experts.csv" \
  --entropy-size "$prefix_end" >"$result_root/oracle_plan.log"

if [[ "${FX4_20GROUP_BUILD_ONLY:-0}" == 1 ]]; then
  echo "20-group research binary and plans built successfully."
  exit 0
fi

rm -f -- "$result_root"/oracle_global*.bytes \
  "$result_root"/oracle_global*.time.txt "$result_root"/oracle_global*.log \
  "$result_root"/oracle_selective*.bytes \
  "$result_root"/oracle_selective*.time.txt \
  "$result_root"/oracle_selective*.log "$result_root"/scr2*.bytes \
  "$result_root"/scr2*.time.txt "$result_root"/scr2*.log \
  "$result_root"/oracle_analysis.json "$result_root"/final_summary.txt \
  "$result_root"/SHA256SUMS

run_case() {
  local name="$1"
  shift
  rm -f "$work/$name.fx4" "$work/$name.restored" "$work/ppm.temp"
  /usr/bin/time -v -o "$result_root/$name.time.txt" \
    env MALLOC_ARENA_MAX=1 FX4_RAW_ENTROPY_INPUT=1 FX4_FORCE_FULL_VOCAB=1 \
      "$@" taskset -c "$cpu" "$work/cmix" -n "$dictionary" \
      "$input" "$work/$name.fx4" >"$result_root/$name.log" 2>&1
  stat -c%s "$work/$name.fx4" >"$result_root/$name.bytes"
}

cmix_hash="$(sha256sum "$work/cmix" | cut -d' ' -f1)"
input_hash="$(sha256sum "$input" | cut -d' ' -f1)"
baseline_record="$result_root/baseline.completed"
baseline_valid=0
if [[ -f "$baseline_record" && -f "$result_root/baseline.bytes" ]] &&
   grep -qx "cmix_sha256=$cmix_hash" "$baseline_record" &&
   grep -qx "input_sha256=$input_hash" "$baseline_record" &&
   grep -qx "input_bytes=$prefix_end" "$baseline_record" &&
   grep -qx "archive_bytes=$(cat "$result_root/baseline.bytes")" \
     "$baseline_record"; then
  baseline_valid=1
fi
if [[ "$baseline_valid" == 1 ]]; then
  echo "Reusing exact completed baseline: $(cat "$result_root/baseline.bytes") bytes"
else
  run_case baseline
  {
    echo "cmix_sha256=$cmix_hash"
    echo "input_sha256=$input_hash"
    echo "input_bytes=$prefix_end"
    echo "archive_bytes=$(cat "$result_root/baseline.bytes")"
  } >"$baseline_record"
fi

oracle_record="$result_root/oracle.completed"
oracle_plan_hash="$(sha256sum "$work/all_experts.f4cp" | cut -d' ' -f1)"
oracle_valid=0
if [[ -f "$oracle_record" && -f "$result_root/oracle.bytes" &&
      -f "$result_root/oracle_spans.csv" ]] &&
   grep -qx "cmix_sha256=$cmix_hash" "$oracle_record" &&
   grep -qx "input_sha256=$input_hash" "$oracle_record" &&
   grep -qx "plan_sha256=$oracle_plan_hash" "$oracle_record" &&
   grep -qx "archive_bytes=$(cat "$result_root/oracle.bytes")" \
     "$oracle_record" &&
   [[ "$(wc -l <"$result_root/oracle_spans.csv")" == 21 ]]; then
  oracle_valid=1
fi
if [[ "$oracle_valid" == 1 ]]; then
  echo "Reusing exact completed 20-group oracle."
else
  rm -f -- "$result_root/oracle.bytes" "$result_root/oracle.time.txt" \
    "$result_root/oracle.log" "$result_root/oracle_spans.csv" \
    "$result_root/oracle.completed"
  run_case oracle \
    FX4_DONOR_PLAN="$work/all_experts.f4cp" \
    FX4_POSTR1_ORACLE_ONLY=1 \
    FX4_POSTR1_SPAN_ORACLE="$result_root/oracle_spans.csv"
  [[ "$(wc -l <"$result_root/oracle_spans.csv")" == 21 ]]
  {
    echo "cmix_sha256=$cmix_hash"
    echo "input_sha256=$input_hash"
    echo "plan_sha256=$oracle_plan_hash"
    echo "archive_bytes=$(cat "$result_root/oracle.bytes")"
  } >"$oracle_record"
fi

python3 "$work/source/tools/analyze_20group_oracle.py" \
  "$result_root/oracle_spans.csv" \
  --global-csv "$work/oracle_global.csv" \
  --selective-csv "$work/oracle_selective.csv" \
  --report "$result_root/oracle_analysis.json" \
  >"$result_root/oracle_analysis.log"

for mode in global selective; do
  csv="$work/oracle_${mode}.csv"
  if [[ "$(wc -l <"$csv")" -le 1 ]]; then
    continue
  fi
  python3 "$work/source/tools/build_postr1_portfolio.py" "$input" \
    --prefix "$work/oracle_${mode}_plan" --expert-csv "$csv" \
    --entropy-size "$prefix_end" >"$result_root/oracle_${mode}_plan.log"
  run_case "oracle_${mode}" \
    FX4_DONOR_PLAN="$work/oracle_${mode}_plan.f4cp"
done

run_case scr2_all \
  FX4_CAUSAL_SCR2="$work/group_ranges.csv" \
  FX4_CAUSAL_SCR2_PATTERNS=all \
  FX4_CAUSAL_SCR2_STATS="$result_root/scr2_all_stats.csv"

# These two candidates share one profile over the complete prefix. They pay
# only the seven-byte causal header and retain adaptation across group edges.
run_case scr2_continuous \
  FX4_CAUSAL_SCR2=all \
  FX4_CAUSAL_SCR2_PATTERNS=all
run_case scr2_positive \
  FX4_CAUSAL_SCR2=all \
  FX4_CAUSAL_SCR2_PATTERNS=1,3,7,11,15,38,76

python3 "$work/source/tools/select_causal_scr2_winners.py" \
  "$result_root/scr2_all_stats.csv" \
  --ranges "$work/scr2_selected_ranges.csv" \
  --patterns "$work/scr2_selected_patterns.txt" \
  --report "$result_root/scr2_selection.json" \
  >"$result_root/scr2_selection.log"

if [[ "$(wc -l <"$work/scr2_selected_ranges.csv")" -gt 1 &&
      -s "$work/scr2_selected_patterns.txt" ]]; then
  run_case scr2_selected \
    FX4_CAUSAL_SCR2="$work/scr2_selected_ranges.csv" \
    FX4_CAUSAL_SCR2_PATTERNS="$(tr -d '\r\n' <"$work/scr2_selected_patterns.txt")"
fi

baseline_bytes="$(cat "$result_root/baseline.bytes")"
best_name=baseline
best_bytes="$baseline_bytes"
for bytes_file in "$result_root"/*.bytes; do
  name="$(basename "$bytes_file" .bytes)"
  [[ "$name" == oracle ]] && continue
  size="$(cat "$bytes_file")"
  if (( size < best_bytes )); then
    best_name="$name"
    best_bytes="$size"
  fi
done

roundtrip=not_run
if [[ "$best_name" != baseline ]]; then
  rm -f "$work/$best_name.restored" "$work/ppm.temp"
  /usr/bin/time -v -o "$result_root/$best_name.decompress.time.txt" \
    env MALLOC_ARENA_MAX=1 FX4_RAW_ENTROPY_INPUT=1 \
    taskset -c "$cpu" "$work/cmix" -d "$dictionary" \
      "$work/$best_name.fx4" "$work/$best_name.restored" \
      >"$result_root/$best_name.decompress.log" 2>&1
  if cmp -s "$input" "$work/$best_name.restored"; then
    roundtrip=exact
  else
    roundtrip=failed
  fi
fi

awk -v size="$prefix_end" -v baseline="$baseline_bytes" \
  -v best="$best_bytes" -v name="$best_name" -v roundtrip="$roundtrip" \
  'BEGIN {
    print "input_bytes=" size;
    print "baseline_bytes=" baseline;
    printf "baseline_bpb=%.9f\n", baseline * 8.0 / size;
    print "winner=" name;
    print "winner_bytes=" best;
    printf "winner_bpb=%.9f\n", best * 8.0 / size;
    print "saving_bytes=" baseline - best;
    print "roundtrip=" roundtrip;
  }' >"$result_root/final_summary.txt"

cp "$work/all_experts.csv" "$work/group_ranges.csv" \
  "$work/oracle_global.csv" "$work/oracle_selective.csv" \
  "$work/scr2_selected_ranges.csv" "$work/scr2_selected_patterns.txt" \
  "$result_root/" 2>/dev/null || true
sha256sum "$input" "$work/$best_name.fx4" \
  "$work/$best_name.restored" 2>/dev/null >"$result_root/SHA256SUMS" || true
cat "$result_root/final_summary.txt"
