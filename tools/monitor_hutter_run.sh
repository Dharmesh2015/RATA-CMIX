#!/usr/bin/env bash
set -u

run_dir="${1:-$PWD}"
interval="${2:-60}"
cd "$run_dir" || exit 2

while :; do
  clear
  printf 'UTC:   %s\n' "$(date -u '+%F %T')"
  printf 'India: %s\n\n' "$(TZ=Asia/Kolkata date '+%F %T %Z')"

  pid="$(pgrep -n -f '(^|/)cmix -e enwik9 archive9' 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    echo "FX4 compression is not running."
    [[ -s archive9 ]] && stat -c 'archive9: %s bytes' archive9
    break
  fi

  ps -p "$pid" -o pid,psr,pcpu,pmem,etime,time,rss,cmd
  input_path=""
  input_pos=0
  input_size=0
  for fd in /proc/"$pid"/fd/*; do
    path="$(readlink "$fd" 2>/dev/null || true)"
    case "$path" in
      "$run_dir"/*.cmix.temp)
        pos="$(awk '$1=="pos:" {print $2}' \
          /proc/"$pid"/fdinfo/"${fd##*/}" 2>/dev/null)"
        size="$(stat -c%s "$path" 2>/dev/null || echo 0)"
        if [[ -n "$pos" && "$size" -gt 0 && "$pos" -le "$size" ]]; then
          input_path="$path"
          input_pos="$pos"
          input_size="$size"
        fi
        ;;
    esac
  done

  if [[ -n "$input_path" ]]; then
    payload_size="$(stat -c%s archive9 2>/dev/null || echo 0)"
    overhead=16
    for file in .decomp_bin .dict.comp .tfweights; do
      size="$(stat -c%s "$file" 2>/dev/null || echo 0)"
      overhead=$((overhead + size))
    done
    awk -v p="$input_pos" -v s="$input_size" -v a="$payload_size" \
        -v o="$overhead" '
      BEGIN {
        pct = 100.0 * p / s
        printf "\nEntropy input: %s\n", "'"$input_path"'"
        printf "Entropy progress: %.2f%% (%d/%d bytes)\n", pct, p, s
        printf "Current coded payload on disk: %d bytes\n", a
        if (p > 1048576 && a > 37) {
          projected_payload = (a - 37) * s / p + 37
          printf "Linear archive9 projection: %.0f bytes\n",
              projected_payload + o
          printf "Projection is diagnostic only; model loss is nonstationary.\n"
        }
      }'
  else
    echo
    echo "Stage: self-extraction, preprocessing, article ordering, PHDA9 or WRT."
  fi

  stat -c 'ppm.temp: %s bytes' ppm.temp 2>/dev/null || true
  free -h | sed -n '1,3p'
  sleep "$interval"
done
