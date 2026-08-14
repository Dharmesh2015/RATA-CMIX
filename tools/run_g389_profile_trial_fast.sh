#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly full_stream=/root/fx4crawler_98090/enwik9.post_r1.bin
readonly work_root=/root/fx4_g389_profile_fast_20260812

mkdir -p "$work_root"
exec 9>"$work_root/run.lock"
flock -n 9 || { echo "G389 trial is already running" >&2; exit 3; }
find "$work_root" -mindepth 1 -maxdepth 1 ! -name run.lock -delete

cp /root/fx4_g3_warm6_20260811_v1/run/cmix "$work_root/cmix"
cp "$source_root/dictionary/english.dic" "$work_root/english.dic"

# Continuous, page-aligned G388 -> G389 slice from the canonical post-R1 stream.
dd if="$full_stream" of="$work_root/g388_g389.postr1" \
  bs=1M skip=406790108 count=2090195 \
  iflag=skip_bytes,count_bytes status=none

printf '%s\n' \
  'profile,donor_offset,length,order' \
  '0,783396,4096,0' \
  '0,791588,2048,1' \
  '0,985636,2048,2' \
  '0,997412,4096,3' \
  '0,1003556,4096,4' \
  '0,1009700,4096,5' \
  '0,1019940,2048,6' > "$work_root/donors.csv"

printf '%s\n' \
  'offset,length,experts,stream_class,profile,residual_gain,mini_models' \
  '1048822,1041373,donor_profile,url,0,0.25,' > "$work_root/experts.csv"

python3 "$source_root/tools/build_postr1_portfolio.py" \
  "$work_root/g388_g389.postr1" \
  --prefix "$work_root/g389_profile" \
  --expert-csv "$work_root/experts.csv" \
  --donor-csv "$work_root/donors.csv" \
  --donor-profile-bank > "$work_root/plan.report"

cd "$work_root"
echo BASELINE_START
/usr/bin/time -v -o baseline.time \
  env FX4_RAW_ENTROPY_INPUT=1 FX4_FORCE_FULL_VOCAB=1 \
  taskset -c 7 ./cmix -n english.dic g388_g389.postr1 baseline.fx4
baseline=$(stat -c%s baseline.fx4)
echo "BASELINE_DONE bytes=$baseline"

rm -f ppm.temp
echo PROFILE_START
/usr/bin/time -v -o profile.time \
  env FX4_RAW_ENTROPY_INPUT=1 FX4_FORCE_FULL_VOCAB=1 \
      FX4_DONOR_PLAN="$work_root/g389_profile.f4cp" \
  taskset -c 7 ./cmix -n english.dic g388_g389.postr1 profile.fx4
candidate=$(stat -c%s profile.fx4)
saving=$((baseline - candidate))
echo "PROFILE_DONE bytes=$candidate baseline=$baseline saving=$saving"

if (( candidate < baseline )); then
  echo GAIN_STARTING_DECOMPRESSION
  rm -f ppm.temp restored.postr1
  /usr/bin/time -v -o profile.decompress.time \
    env FX4_RAW_ENTROPY_OUTPUT=1 \
    taskset -c 7 ./cmix -d english.dic profile.fx4 restored.postr1
  cmp g388_g389.postr1 restored.postr1
  echo ROUNDTRIP_EXACT
else
  echo NO_GAIN_DECOMPRESSION_SKIPPED
fi
