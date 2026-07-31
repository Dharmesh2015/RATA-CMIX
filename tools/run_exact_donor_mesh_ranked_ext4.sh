#!/usr/bin/env bash
set -u

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 ENWIK9 F4CD_CANDIDATE_PLAN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly result_root="${FX4_DONOR_RESULT_ROOT:-$source_root/research/donor_exact_mesh_20260729}"
readonly ledger="$result_root/edges.csv"

set +e
bash "$source_root/tools/run_exact_donor_mesh_ext4.sh" "$1" "$2"
status=$?
set -e

if [[ -f "$ledger" ]]; then
  python3 "$source_root/tools/rank_donor_edges.py" \
    "$ledger" --output-dir "$result_root" \
    | tee "$result_root/ranking_latest.txt"
fi

exit "$status"
