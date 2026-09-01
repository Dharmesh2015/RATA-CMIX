#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ENWIK9 OUTPUT_ARCHIVE9 [RESULT_DIRECTORY]" >&2
  exit 2
fi

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly input_path="$(readlink -f "$1")"
readonly export_path="$(readlink -m "$2")"
result_root="$source_root/results/target93_selective_full"
[[ $# -eq 3 ]] && result_root="$3"
readonly result_root
readonly work_root="$(printenv FX4_WORK_ROOT 2>/dev/null || \
  printf /root/fx4_target93_selective_full)"
readonly cpu="$(printenv FX4_TEST_CPU 2>/dev/null || printf 7)"
readonly jobs="$(printenv FX4_BUILD_JOBS 2>/dev/null || printf 3)"
readonly cooldown="$(printenv FX4_COOLDOWN_SECONDS 2>/dev/null || printf 120)"
readonly validate_roundtrip="$(printenv FX4_VALIDATE_ROUNDTRIP 2>/dev/null || printf 1)"
readonly min_free_bytes="$(printenv FX4_MIN_FREE_BYTES 2>/dev/null || printf 30000000000)"
readonly ppm_rss_mb="$(printenv FX4_PPM_RSS_MB 2>/dev/null || printf 8704)"
readonly model_rel=models/transformer6m/6m-q4-fp32.tfwc2
readonly plan_rel=results/donor_multiday_seed/all_ranked.f4cd
readonly recipients_rel=results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv
readonly candidates_rel=results/donor_multiday_seed/enwik9.grouped_1mib_candidates.csv

test -f "$input_path"
test "$(stat -c%s "$input_path")" -eq 1000000000
test -s "$source_root/$model_rel"
test -s "$source_root/$plan_rel"
test -s "$source_root/$recipients_rel"
test -s "$source_root/$candidates_rel"
[[ "$ppm_rss_mb" =~ ^[0-9]+$ ]] || {
  echo "FX4_PPM_RSS_MB must be an integer MiB value" >&2
  exit 2
}
case "$(readlink -m "$work_root")" in
  /root/fx4_target93_selective_full|/root/fx4_target93_selective_full_*) ;;
  *)
    echo "FX4_WORK_ROOT must remain under /root/fx4_target93_selective_full*" >&2
    exit 2
    ;;
esac
if pgrep -x cmix >/dev/null 2>&1; then
  echo "another cmix process is running; stop it before this full run" >&2
  exit 3
fi
available="$(df -PB1 /root | awk 'NR == 2 {print $4}')"
if [[ -z "$available" || "$available" -lt "$min_free_bytes" ]]; then
  echo "need at least $min_free_bytes bytes free on WSL ext4" >&2
  exit 2
fi

cat <<EOF
FX4 connected CPU-only selective run
  canonical stream: WRT -> payload_lex/R1
  primary models: PPMd order 25 / 14000 MiB mmap, FXCM v26,
                  frozen 6M transformer replacing the online byte LSTM
  retained CPU experts: DeepMix contexts, GrammarMatch, ESN/NLMS
  bridge: full FXCM byte-distribution input plus half-strength middle input
  selective search: donor profiles, 11-model mini-cmix, URL specialist,
                    arithmetic-cost SCR2 virtual replay
  PPM RSS purge trigger: $ppm_rss_mb MiB
  GPU: disabled
  ALTXS M3/M5: excluded because its changed stream is incompatible with the
               supplied transformer's canonical-stream weights
EOF

mkdir -p "$work_root" "$result_root" "$(dirname "$export_path")"
exec 9>"$work_root/run.lock"
flock -n 9 || {
  echo "another selective full run owns $work_root" >&2
  exit 3
}

state_file="$work_root/state.env"
write_state() {
  local phase="$1"
  local temporary="$state_file.new"
  {
    echo "phase=$phase"
    echo "work_root=$work_root"
    echo "result_root=$result_root"
    echo "cpu=$cpu"
    echo "ppm_rss_mb=$ppm_rss_mb"
    echo "updated_epoch=$(date +%s)"
    [[ -f "$work_root/selected_s1_bytes" ]] &&
      echo "s1_bytes=$(cat "$work_root/selected_s1_bytes")"
    [[ -f "$work_root/selected_architecture" ]] &&
      echo "architecture=$(cat "$work_root/selected_architecture")"
  } >"$temporary"
  mv -f "$temporary" "$state_file"
}

source_hash="$(
  cd "$source_root"
  {
    find src dictionary models/transformer6m -type f -print0 |
      sort -z | xargs -0 sha256sum
    sha256sum makefile tools/build_selective_plan.cpp \
      "$plan_rel" "$recipients_rel" "$candidates_rel"
  } | sha256sum | cut -d' ' -f1
)"
cached_hash="$(cat "$work_root/source.sha256" 2>/dev/null || true)"
if [[ "$cached_hash" != "$source_hash" ]]; then
  rm -rf -- "$work_root/source" "$work_root/bin" "$work_root/assets"
  rm -f -- "$work_root/source.sha256"
  mkdir -p "$work_root/source/tools" "$work_root/bin"
  cp -a "$source_root/src" "$source_root/dictionary" \
    "$source_root/models" "$work_root/source/"
  cp "$source_root/makefile" "$work_root/source/"
  cp "$source_root/tools/build_selective_plan.cpp" "$work_root/source/tools/"
  sed -i 's/\r$//' "$work_root/source/makefile"
  printf '%s\n' "$source_hash" >"$work_root/source.sha256"
fi

strip_pack() {
  local binary="$1"
  if command -v llvm-strip-17 >/dev/null 2>&1; then
    llvm-strip-17 --strip-all "$binary"
  else
    strip --strip-all "$binary"
  fi
  if command -v objcopy >/dev/null 2>&1; then
    objcopy --remove-section=.comment --remove-section=.note.gnu.property \
      --remove-section=.note.gnu.build-id --remove-section=.note.ABI-tag \
      "$binary" 2>/dev/null || true
  fi
  upx_enabled="$(printenv FX4_UPX 2>/dev/null || printf 1)"
  if [[ "$upx_enabled" == 1 ]]; then
    if command -v upx >/dev/null 2>&1; then
      upx --ultra-brute "$binary" >/dev/null
      upx -t "$binary" >/dev/null
    elif [[ -x "$source_root/tools/upx" ]]; then
      "$source_root/tools/upx" --ultra-brute "$binary" >/dev/null
      "$source_root/tools/upx" -t "$binary" >/dev/null
    fi
  fi
}

build_core() {
  local goal="$1"
  local name="$2"
  if [[ -x "$work_root/bin/$name" ]]; then return; fi
  write_state "build_$name"
  cd "$work_root/source"
  make clean
  make "$goal" -j"$jobs" OUT="$name"
  strip_pack "$name"
  cp "$name" "$work_root/bin/$name"
}

build_core target93 cmix_baseline_core
build_core selective20 cmix_discovery_core
if [[ ! -x "$work_root/bin/build_selective_plan" ]]; then
  cd "$work_root/source"
  make selective-plan-builder
  cp build_selective_plan "$work_root/bin/"
fi

build_assets() {
  if [[ -f "$work_root/assets/complete" ]]; then return; fi
  write_state package_assets
  rm -rf -- "$work_root/assets"
  mkdir -p "$work_root/assets"
  cp "$work_root/bin/cmix_baseline_core" "$work_root/assets/cmix_orig"
  cp "$work_root/source/dictionary/english.dic" "$work_root/assets/english.dic"
  cp "$work_root/source/src/readalike_prepr/data/new_article_order" \
    "$work_root/assets/article_order"
  cp "$work_root/source/$model_rel" "$work_root/assets/transformer6m.weights"
  cd "$work_root/assets"
  export FX4_TRANSFORMER_WEIGHTS="$PWD/transformer6m.weights"
  ./cmix_orig -c english.dic comp_dict
  rm -f ppm.temp
  ./cmix_orig -c article_order comp_order
  rm -f ppm.temp
  ./cmix_orig -d comp_dict verify_dict
  cmp -s english.dic verify_dict
  rm -f ppm.temp verify_dict
  ./cmix_orig -d comp_order verify_order
  cmp -s article_order verify_order
  rm -f ppm.temp verify_order
  ./cmix_orig -h "$(stat -c%s comp_dict)" "$(stat -c%s comp_order)" \
    0 "$(stat -c%s transformer6m.weights)"
  touch complete
}

assemble_s1() {
  local core="$1"
  local output="$2"
  cat "$work_root/bin/$core" "$work_root/assets/comp_dict" \
    "$work_root/assets/comp_order" "$work_root/assets/transformer6m.weights" \
    "$work_root/assets/header.dat" >"$output"
  chmod 0755 "$output"
}

build_assets
assemble_s1 cmix_baseline_core "$work_root/bin/cmix_baseline"
assemble_s1 cmix_discovery_core "$work_root/bin/cmix_discovery"

input_size="$(stat -c%s "$work_root/enwik9" 2>/dev/null || true)"
if [[ "$input_size" != 1000000000 ]]; then
  cp "$input_path" "$work_root/enwik9"
fi
cp "$source_root/$plan_rel" "$work_root/candidates.f4cd"
cp "$source_root/$recipients_rel" "$work_root/recipients.csv"
cp "$source_root/$candidates_rel" "$work_root/ranked_candidates.csv"

metadata="$result_root/run.meta"
metadata_new="$result_root/run.meta.new"
{
  echo "architecture=target93_selective_full_v1"
  echo "source_sha256=$source_hash"
  echo "input_sha256=$(sha256sum "$input_path" | cut -d' ' -f1)"
  echo "candidate_plan_sha256=$(sha256sum "$work_root/candidates.f4cd" | cut -d' ' -f1)"
  echo "planned_donors=$(printenv FX4_WINNER_PLANNED_DONORS 2>/dev/null || printf 7)"
  echo "beam_width=$(printenv FX4_WINNER_PORTFOLIO_BEAM_WIDTH 2>/dev/null || printf 8)"
  echo "donor_atoms=$(printenv FX4_WINNER_PORTFOLIO_DONOR_ATOMS 2>/dev/null || printf 24)"
  echo "max_candidates=$(printenv FX4_WINNER_PORTFOLIO_MAX_CANDIDATES 2>/dev/null || printf 64)"
    echo "max_depth=$(printenv FX4_WINNER_PORTFOLIO_MAX_DEPTH 2>/dev/null || printf 7)"
    echo "ppm_rss_mb=$ppm_rss_mb"
  } >"$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "result directory belongs to another source/input/configuration" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  exit 2
fi
mv -f "$metadata_new" "$metadata"

post_r1="$work_root/enwik9.post_r1.bin"
common_environment=(
  "MALLOC_ARENA_MAX=1"
  "MALLOC_TRIM_THRESHOLD_=131072"
  "CUDA_VISIBLE_DEVICES="
  "HIP_VISIBLE_DEVICES="
  "ROCR_VISIBLE_DEVICES="
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "CMIX_PPM_RSS_MB=$ppm_rss_mb"
  "FX4_ENABLE_TRANSFORMER6M=1"
  "FX4_DONOR_PLAN=$work_root/candidates.f4cd"
  "FX4_DONOR_WINNER_SEARCH=1"
  "FX4_DONOR_TRIAL_CPU=$cpu"
  "FX4_WINNER_PACKS_CSV=$work_root/recipients.csv"
  "FX4_WINNER_PAGE_CANDIDATES_CSV=$work_root/ranked_candidates.csv"
  "FX4_WINNER_START_REGION=$(printenv FX4_WINNER_START_REGION 2>/dev/null || printf 0)"
  "FX4_WINNER_MAX_REGIONS=$(printenv FX4_WINNER_MAX_REGIONS 2>/dev/null || printf 0)"
  "FX4_DONOR_MAX_NEW_TRIALS=$(printenv FX4_DONOR_MAX_NEW_TRIALS 2>/dev/null || printf 0)"
  "FX4_WINNER_QUICK_SINGLES=1"
  "FX4_WINNER_PLANNED_DONORS=$(printenv FX4_WINNER_PLANNED_DONORS 2>/dev/null || printf 7)"
  "FX4_WINNER_PORTFOLIO_BEAM_WIDTH=$(printenv FX4_WINNER_PORTFOLIO_BEAM_WIDTH 2>/dev/null || printf 8)"
  "FX4_WINNER_PORTFOLIO_DONOR_ATOMS=$(printenv FX4_WINNER_PORTFOLIO_DONOR_ATOMS 2>/dev/null || printf 24)"
  "FX4_WINNER_PORTFOLIO_MAX_CANDIDATES=$(printenv FX4_WINNER_PORTFOLIO_MAX_CANDIDATES 2>/dev/null || printf 64)"
  "FX4_WINNER_PORTFOLIO_MAX_DEPTH=$(printenv FX4_WINNER_PORTFOLIO_MAX_DEPTH 2>/dev/null || printf 7)"
  "FX4_WINNER_CONTEXT_MIXER=0"
  "FX4_WINNER_NEAR_BYTES=$(printenv FX4_WINNER_NEAR_BYTES 2>/dev/null || printf 64)"
  "FX4_WINNER_SCR2_COST=1"
  "FX4_WINNER_SCR2_COST_ONLY=0"
  "FX4_WINNER_LEAVE_ONE_OUT=0"
)

run_phase() {
  local phase="$1"
  shift
  local ledger="$result_root/search.$phase"
  if [[ -f "$ledger.winner.complete" ]]; then
    if [[ "$phase" != individual || -s "$post_r1" ]]; then
      echo "Reusing completed phase: $phase"
      return
    fi
    rm -f "$ledger.winner.complete"
  fi
  write_state "discovery_$phase"
  cd "$work_root"
  rm -f "payload_$phase" "payload_$phase.cmix.temp" ppm.temp
  local -a phase_environment=(
    "${common_environment[@]}"
    "FX4_DONOR_DISCOVERY_RESULTS=$ledger"
    "FX4_WINNER_PORTFOLIO_PHASE=$phase"
  )
  while [[ $# -gt 0 ]]; do
    phase_environment+=("$1")
    shift
  done
  echo "Selective discovery phase: $phase"
  set +e
  /usr/bin/time -v -o "$result_root/$phase.time.txt" \
    env "${phase_environment[@]}" \
      nice -n "$(printenv FX4_NICE 2>/dev/null || printf 5)" \
      taskset -c "$cpu" ./bin/cmix_discovery \
      -e enwik9 "payload_$phase" 2>&1 | tee -a "$result_root/$phase.log"
  status=${PIPESTATUS[0]}
  set -e
  rm -f "payload_$phase" "payload_$phase.cmix.temp" ppm.temp
  if [[ "$status" -ne 0 ]]; then
    echo "phase $phase stopped with status $status" >&2
    echo "completed recipient rows are durable; rerun this script" >&2
    exit "$status"
  fi
  test -f "$ledger.winner.complete"
}

run_phase individual "FX4_VR_DUMP_INPUT=$post_r1"
run_phase donor_beam \
  "FX4_WINNER_INDIVIDUAL_TRIALS=$result_root/search.individual.portfolio_trials.csv"
run_phase combine \
  "FX4_WINNER_INDIVIDUAL_TRIALS=$result_root/search.individual.portfolio_trials.csv" \
  "FX4_WINNER_DONOR_TRIALS=$result_root/search.donor_beam.portfolio_trials.csv"
test -s "$post_r1"

build_core target93-donor cmix_f4cp_core
build_core target93-scr2 cmix_f4vr_core
build_core target93-selective cmix_combined_core
assemble_s1 cmix_f4cp_core "$work_root/bin/cmix_f4cp"
assemble_s1 cmix_f4vr_core "$work_root/bin/cmix_f4vr"
assemble_s1 cmix_combined_core "$work_root/bin/cmix_combined"

builder_common=(
  --candidate-plan "$work_root/candidates.f4cd"
  --post-r1 "$post_r1"
  --selected "$result_root/search.individual.portfolio_selected.csv"
  --selected "$result_root/search.donor_beam.portfolio_selected.csv"
  --selected "$result_root/search.combine.portfolio_selected.csv"
  --trials "$result_root/search.individual.portfolio_trials.csv"
  --trials "$result_root/search.donor_beam.portfolio_trials.csv"
  --trials "$result_root/search.combine.portfolio_trials.csv"
  --events "$result_root/search.individual.portfolio_vr_events.csv"
  --events "$result_root/search.donor_beam.portfolio_vr_events.csv"
  --events "$result_root/search.combine.portfolio_vr_events.csv"
)

run_builder() {
  local name="$1"
  local allow_f4cp="$2"
  local allow_f4vr="$3"
  local candidate_s1="$4"
  local directory="$result_root/plans/$name"
  mkdir -p "$directory"
  "$work_root/bin/build_selective_plan" \
    "${builder_common[@]}" \
    --allow-f4cp "$allow_f4cp" --allow-f4vr "$allow_f4vr" \
    --baseline-s1 "$(stat -c%s "$work_root/bin/cmix_baseline")" \
    --candidate-s1 "$candidate_s1" \
    --output-f4cp "$directory/winners.f4cp" \
    --output-f4vr "$directory/winners.f4vr" \
    --output-selected "$directory/selected.csv" \
    --summary "$directory/summary.txt" >"$directory/builder.log"
}

write_state plan_selection
run_builder baseline 0 0 "$(stat -c%s "$work_root/bin/cmix_baseline")"
run_builder f4cp 1 0 "$(stat -c%s "$work_root/bin/cmix_f4cp")"
run_builder f4vr 0 1 "$(stat -c%s "$work_root/bin/cmix_f4vr")"
run_builder combined 1 1 "$(stat -c%s "$work_root/bin/cmix_combined")"

best=baseline
best_net=0
for candidate in f4cp f4vr combined; do
  net="$(awk -F= '$1 == "projected_hutter_net_bytes" {print $2}' \
    "$result_root/plans/$candidate/summary.txt")"
  if [[ "$net" =~ ^-?[0-9]+$ ]] && (( net > best_net )); then
    best="$candidate"
    best_net="$net"
  fi
done

echo "$best" >"$work_root/selected_architecture"
case "$best" in
  baseline) selected_s1="$work_root/bin/cmix_baseline" ;;
  f4cp) selected_s1="$work_root/bin/cmix_f4cp" ;;
  f4vr) selected_s1="$work_root/bin/cmix_f4vr" ;;
  combined) selected_s1="$work_root/bin/cmix_combined" ;;
esac
stat -c%s "$selected_s1" >"$work_root/selected_s1_bytes"
cp "$result_root/plans/$best/winners.f4cp" "$work_root/winners.f4cp"
rm -f "$work_root/winners.f4vr"
if [[ -f "$result_root/plans/$best/winners.f4vr" ]]; then
  cp "$result_root/plans/$best/winners.f4vr" "$work_root/winners.f4vr"
fi
cp "$result_root/plans/$best/selected.csv" "$result_root/selected_winners.csv"
cp "$result_root/plans/$best/summary.txt" "$result_root/selected_summary.txt"
cp "$selected_s1" "$work_root/cmix"
chmod 0755 "$work_root/cmix"

selected_f4cp="$(awk -F= '$1 == "f4cp_required" {print $2}' \
  "$result_root/selected_summary.txt")"
selected_f4vr="$(awk -F= '$1 == "f4vr_required" {print $2}' \
  "$result_root/selected_summary.txt")"
echo "Selected release architecture: $best"
cat "$result_root/selected_summary.txt"

write_state cooldown
echo "Cooling down for $cooldown seconds before final single-core encode..."
sleep "$cooldown"
write_state compress
cd "$work_root"
rm -f archive9 cmix_payload cmix_payload.cmix.temp ppm.temp
rm -f compress.log compress.time.txt final.complete
final_environment=(
  "MALLOC_ARENA_MAX=1"
  "MALLOC_TRIM_THRESHOLD_=131072"
  "CUDA_VISIBLE_DEVICES="
  "HIP_VISIBLE_DEVICES="
  "ROCR_VISIBLE_DEVICES="
  "OMP_NUM_THREADS=1"
  "OPENBLAS_NUM_THREADS=1"
  "MKL_NUM_THREADS=1"
  "CMIX_PPM_RSS_MB=$ppm_rss_mb"
)
if [[ "$selected_f4cp" == 1 ]]; then
  final_environment+=("FX4_DONOR_PLAN=$work_root/winners.f4cp")
fi
if [[ "$selected_f4vr" == 1 ]]; then
  final_environment+=("FX4_VR_PLAN=$work_root/winners.f4vr")
fi
/usr/bin/time -v -o compress.time.txt \
  env "${final_environment[@]}" \
    nice -n "$(printenv FX4_NICE 2>/dev/null || printf 5)" \
    taskset -c "$cpu" ./cmix -e enwik9 cmix_payload \
    2>&1 | tee compress.log
test -s archive9
touch final.complete

archive_bytes="$(stat -c%s archive9)"
s1_bytes="$(stat -c%s cmix)"
hutter_total=$((archive_bytes + s1_bytes))
{
  echo "architecture=$best"
  echo "archive9_bytes=$archive_bytes"
  echo "s1_bytes=$s1_bytes"
  echo "hutter_total_bytes=$hutter_total"
  echo "archive9_sha256=$(sha256sum archive9 | cut -d' ' -f1)"
  echo "s1_sha256=$(sha256sum cmix | cut -d' ' -f1)"
} >"$result_root/final_score.txt"

if [[ "$validate_roundtrip" == 1 ]]; then
  write_state decompress
  roundtrip="$work_root/roundtrip"
  rm -rf -- "$roundtrip"
  mkdir -p "$roundtrip"
  cp archive9 "$roundtrip/archive9"
  chmod 0755 "$roundtrip/archive9"
  cd "$roundtrip"
  /usr/bin/time -v -o "$result_root/decompress.time.txt" \
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      MALLOC_ARENA_MAX=1 CMIX_PPM_RSS_MB="$ppm_rss_mb" \
      nice -n "$(printenv FX4_NICE 2>/dev/null || printf 5)" \
      taskset -c "$cpu" ./archive9 \
      >"$result_root/decompress.log" 2>&1
  cmp "$work_root/enwik9" enwik9_uncompressed
  {
    echo "input_sha256=$(sha256sum "$work_root/enwik9" | cut -d' ' -f1)"
    echo "output_sha256=$(sha256sum enwik9_uncompressed | cut -d' ' -f1)"
    echo "roundtrip=exact"
  } >"$result_root/roundtrip.txt"
fi

write_state complete
cp "$work_root/archive9" "$export_path"
cp "$work_root/cmix" "$export_path.s1"
cp "$result_root/final_score.txt" "$export_path.score.txt"
cp "$result_root/selected_summary.txt" "$export_path.selective.txt"
echo "Exported archive: $export_path"
cat "$result_root/final_score.txt"
