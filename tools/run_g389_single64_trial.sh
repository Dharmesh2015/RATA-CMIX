#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4_g389_profile_aligned_20260812

cd "$work_root"
test -s baseline.fx4
test -s cmix_gate
exec 9>single64.lock
flock -n 9 || { echo "G389 single-window trial is already running" >&2; exit 3; }

printf '%s\n' \
  'profile,donor_offset,length,order' \
  '0,983296,65536,0' > donors_single64.csv
printf '%s\n' \
  'offset,length,experts,stream_class,profile,residual_gain,mini_models' \
  '1049042,1041373,donor_profile,url,0,0.25,' > experts_single64.csv

python3 "$source_root/tools/build_postr1_portfolio.py" \
  g388_g389.postr1 \
  --prefix g389_single64 \
  --expert-csv experts_single64.csv \
  --donor-csv donors_single64.csv \
  --donor-profile-bank > plan_single64.report

rm -f ppm.temp single64.fx4 single64.restored
echo SINGLE64_START
/usr/bin/time -v -o single64.time \
  env FX4_RAW_ENTROPY_INPUT=1 FX4_FORCE_FULL_VOCAB=1 \
      FX4_DONOR_PLAN="$work_root/g389_single64.f4cp" \
  taskset -c 7 ./cmix_gate -n english.dic g388_g389.postr1 single64.fx4

baseline=$(stat -c%s baseline.fx4)
candidate=$(stat -c%s single64.fx4)
saving=$((baseline - candidate))
echo "SINGLE64_DONE bytes=$candidate baseline=$baseline saving=$saving"

if (( candidate < baseline )); then
  echo GAIN_STARTING_DECOMPRESSION
  rm -f ppm.temp
  /usr/bin/time -v -o single64.decompress.time \
    env FX4_RAW_ENTROPY_OUTPUT=1 \
    taskset -c 7 ./cmix_gate -d english.dic single64.fx4 single64.restored
  cmp g388_g389.postr1 single64.restored
  echo ROUNDTRIP_EXACT
else
  echo NO_GAIN_DECOMPRESSION_SKIPPED
fi
