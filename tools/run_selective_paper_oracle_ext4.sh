#!/usr/bin/env bash
set -euo pipefail

readonly group_id="${1:-421}"
readonly cpu="${2:-7}"
readonly root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly stream=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin
readonly groups="$root/results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv"
readonly pages="$root/results/donor_multiday_seed/enwik9.page_recipients.csv"
readonly work=/root/fx4_selective_paper_oracle
readonly output="$root/results/selective_paper_oracle_20260811/group_${group_id}"
readonly experts=oracle+structural+ppmd_escape_order+sparse_virtual_ppm+word_xml_ppm+residual_lstm+rare_residual+cts_skipcts+dmc+token_match+context_mixer

[[ "$group_id" =~ ^[0-9]+$ ]]
[[ "$cpu" =~ ^[0-9]+$ ]]
test -f "$stream"
test -f "$groups"
test -f "$pages"

row="$(awk -F, -v group="$group_id" 'NR > 1 && $1 == group {print; exit}' "$groups")"
test -n "$row"
offset="$(cut -d, -f4 <<<"$row")"
length="$(cut -d, -f6 <<<"$row")"

mkdir -p "$work" "$output"
exec 9>"$work/run.lock"
flock -n 9 || { echo "another selective oracle owns $work" >&2; exit 3; }

if [[ ! -x "$work/cmix_oracle" ]]; then
  rm -rf "$work/source"
  mkdir -p "$work/source"
  cp -a "$root/src" "$root/dictionary" "$root/tools" "$work/source/"
  cp "$root/makefile" "$root/LICENSE" "$work/source/"
  make -C "$work/source" cmix -j"$(nproc)" \
    DONOR=1 POSTR1=1 OUT=cmix_oracle
  mv "$work/source/cmix_oracle" "$work/cmix_oracle"
fi

dd if="$stream" of="$work/recipient.bin" bs=4M \
  skip="$offset" count="$length" iflag=skip_bytes,count_bytes status=none

printf '%s\n' \
  'offset,length,experts,stream_class,profile,residual_gain,pack_id,status' \
  "0,$length,$experts,mixed,0,0.25,$group_id,oracle_only" \
  >"$work/experts.csv"

python3 "$work/source/tools/build_postr1_portfolio.py" \
  "$work/recipient.bin" --prefix "$work/all_experts" \
  --expert-csv "$work/experts.csv" --entropy-size "$length"

test -s "$work/all_experts.f4cp"
rm -f "$work/oracle.fx4" "$work/oracle.csv" "$work/ppm.temp"

/usr/bin/time -v -o "$work/oracle.time.txt" \
  env FX4_RAW_ENTROPY_INPUT=1 \
      FX4_DONOR_PLAN="$work/all_experts.f4cp" \
      FX4_POSTR1_ORACLE_ONLY=1 \
      FX4_POSTR1_ORACLE="$work/oracle.csv" \
  taskset -c "$cpu" "$work/cmix_oracle" -n \
    "$work/source/dictionary/english.dic" \
    "$work/recipient.bin" "$work/oracle.fx4" \
    >"$work/oracle.log" 2>&1

cp "$work/oracle.csv" "$work/oracle.time.txt" "$work/oracle.log" \
   "$work/experts.csv" "$work/all_experts.f4cp" "$work/oracle.fx4" \
   "$output/"

{
  echo "group=$group_id"
  echo "offset=$offset"
  echo "length=$length"
  echo "plan_bytes=$(stat -c%s "$work/all_experts.f4cp")"
  echo "archive_bytes=$(stat -c%s "$work/oracle.fx4")"
} >"$output/result.txt"

cat "$output/result.txt"
cat "$output/oracle.csv"
