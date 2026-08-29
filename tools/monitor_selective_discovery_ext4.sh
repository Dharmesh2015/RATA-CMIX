#!/usr/bin/env bash
set -euo pipefail

readonly source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly result_root="${1:-$source_root/results/selective_discovery_winner_first}"
readonly interval="${FX4_MONITOR_SECONDS:-60}"

show_phase() {
  local phase="$1"
  local status_file="$result_root/search.$phase.winner.status"
  local selected_file="$result_root/search.$phase.portfolio_selected.csv"

  echo
  echo "Phase: $phase"
  if [[ -f "$status_file" ]]; then
    cat "$status_file"
  else
    echo "Not started."
  fi

  if [[ -f "$selected_file" ]]; then
    awk -F, 'NR > 1 {
      rows++
      if ($20 == "accepted_after_side") {
        winners++
        net += $17
      }
    }
    END {
      print "Completed recipients: " (rows + 0)
      print "Winners after standalone side data: " (winners + 0)
      print "Conservative selected net bytes: " (net + 0)
    }' "$selected_file"
  fi
}

while :; do
  clear 2>/dev/null || true
  echo "WSL UTC: $(date -u '+%F %T')"
  echo "India:   $(TZ=Asia/Kolkata date '+%F %T %Z')"

  pid="$(pgrep -n -f 'cmix -e enwik9 discovery_payload_(individual|donor_beam|combine)' || true)"
  if [[ -n "$pid" ]]; then
    ps -p "$pid" -o pid,psr,pcpu,pmem,etime,time,rss,cmd
    entropy_path=""
    entropy_pos=0
    entropy_size=0
    for fd in /proc/"$pid"/fd/*; do
      path="$(readlink "$fd" 2>/dev/null || true)"
      if [[ "$path" == /root/fx4_selective_discovery/discovery_payload_*.cmix.temp ]]; then
        pos="$(awk '/^pos:/{print $2}' /proc/"$pid"/fdinfo/"${fd##*/}")"
        size="$(stat -c%s "$path")"
        if (( pos >= entropy_pos )); then
          entropy_path="$path"
          entropy_pos="$pos"
          entropy_size="$size"
        fi
      fi
    done
    if [[ -n "$entropy_path" ]]; then
      awk -v f="$entropy_path" -v p="$entropy_pos" -v s="$entropy_size" 'BEGIN {
        print "Entropy input: " f " (" s " bytes)"
        printf "Entropy progress: %.2f%% (%d/%d bytes)\n", (s ? 100*p/s : 0), p, s
      }'
    else
      echo "Stage: preprocessing, R1, or discovery setup."
    fi
    if [[ -f /root/fx4_selective_discovery/ppm.temp ]]; then
      stat -c 'ppm.temp: %s bytes' /root/fx4_selective_discovery/ppm.temp
    fi
  else
    echo "Selective discovery is not running."
  fi

  show_phase individual
  show_phase donor_beam
  show_phase combine

  if [[ "${2:-}" != "--watch" || -z "$pid" ]]; then
    break
  fi
  sleep "$interval"
done
