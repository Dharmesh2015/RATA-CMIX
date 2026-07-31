#!/usr/bin/env bash
set -u

readonly result_root="${FX4_DONOR_RESULT_ROOT:-/mnt/d/mywork/myideas/latestcompressor/fx4-cmix/research/donor_exact_mesh_full_9819_v1}"
readonly ledger="$result_root/edges.csv"

echo "utc_time=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if [[ -f "$ledger" ]]; then
  awk -F, '
    NR > 1 {
      tested++
      if ($8 == "error") errors++
      else if ($7 > 0) wins++
      else if ($7 == 0) ties++
      else losses++
    }
    END {
      printf "tested_edges=%d\n", tested
      printf "remaining_edges=%d\n", 9819 - tested
      printf "wins=%d ties=%d losses=%d errors=%d\n",
          wins, ties, losses, errors
    }
  ' "$ledger"
else
  echo "tested_edges=0"
  echo "remaining_edges=9819"
  echo "wins=0 ties=0 losses=0 errors=0"
fi

if [[ -f "$result_root/edges.csv.complete" ]]; then
  echo "scan_state=complete"
elif [[ -f "$result_root/edges.csv.paused" ]]; then
  echo "scan_state=paused"
else
  echo "scan_state=running_or_not_started"
fi

if [[ -f "$result_root/edges.csv.status" ]]; then
  echo "--- current status ---"
  cat "$result_root/edges.csv.status"
fi

echo "--- process ---"
pgrep -af '[.]\/cmix -e enwik9 discovery_payload' || true

if [[ -f "$result_root/ranking_summary.json" ]]; then
  echo "--- last clean-batch ranking ---"
  cat "$result_root/ranking_summary.json"
fi

if [[ -f "$ledger" ]]; then
  echo "--- latest tested edges ---"
  tail -n 5 "$ledger"
fi

if [[ -f "$result_root/all_edges_ranked.csv" ]]; then
  echo "--- top ranked edges ---"
  head -n 6 "$result_root/all_edges_ranked.csv"
fi
