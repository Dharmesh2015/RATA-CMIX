#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 GAIN (0.25, 0.5, 0.75, or 1.0)" >&2
  exit 2
fi

readonly gain="$1"
case "$gain" in
  0.25|0.5|0.75|1.0) ;;
  *) echo "unsupported gain: $gain" >&2; exit 2 ;;
esac

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4_g389_profile_aligned_20260812
readonly tag=${gain/./}

cd "$work_root"
test -s baseline.fx4
test -s g388_g389.postr1
test -s donors.csv

exec 9>gain_trial.lock
flock -n 9 || { echo "a G389 gain trial is already running" >&2; exit 3; }

printf '%s\n' \
  'offset,length,experts,stream_class,profile,residual_gain,mini_models' \
  "1049042,1041373,donor_profile,url,0,$gain," > "experts_g${tag}.csv"

python3 "$source_root/tools/build_postr1_portfolio.py" \
  g388_g389.postr1 \
  --prefix "g389_profile_g${tag}" \
  --expert-csv "experts_g${tag}.csv" \
  --donor-csv donors.csv \
  --donor-profile-bank > "plan_g${tag}.report"

rm -f ppm.temp "profile_g${tag}.fx4" "restored_g${tag}.postr1"
echo "PROFILE_GAIN_${gain}_START"
/usr/bin/time -v -o "profile_g${tag}.time" \
  env FX4_RAW_ENTROPY_INPUT=1 FX4_FORCE_FULL_VOCAB=1 \
      FX4_DONOR_PLAN="$work_root/g389_profile_g${tag}.f4cp" \
  taskset -c 7 ./cmix -n english.dic g388_g389.postr1 \
    "profile_g${tag}.fx4"

baseline=$(stat -c%s baseline.fx4)
candidate=$(stat -c%s "profile_g${tag}.fx4")
saving=$((baseline - candidate))
echo "PROFILE_GAIN_${gain}_DONE bytes=$candidate baseline=$baseline saving=$saving"

if (( candidate < baseline )); then
  echo GAIN_STARTING_DECOMPRESSION
  rm -f ppm.temp
  /usr/bin/time -v -o "profile_g${tag}.decompress.time" \
    env FX4_RAW_ENTROPY_OUTPUT=1 \
    taskset -c 7 ./cmix -d english.dic "profile_g${tag}.fx4" \
      "restored_g${tag}.postr1"
  cmp g388_g389.postr1 "restored_g${tag}.postr1"
  echo ROUNDTRIP_EXACT
else
  echo NO_GAIN_DECOMPRESSION_SKIPPED
fi
