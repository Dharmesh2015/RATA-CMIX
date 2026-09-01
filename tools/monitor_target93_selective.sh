#!/usr/bin/env bash
set -u

run_root="${1:-/root/fx4_target93_selective_full}"
refresh="${2:-300}"
once="${FX4_MONITOR_ONCE:-0}"
state_file="$run_root/state.env"

state_value() {
  local key="$1"
  [[ -f "$state_file" ]] || return 0
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' \
    "$state_file"
}

fd_position() {
  local pid="$1"
  local fd="$2"
  awk '/^pos:/ {print $2; exit}' "/proc/$pid/fdinfo/$fd" 2>/dev/null ||
    printf 0
}

find_cmix_pid() {
  local pid
  local command
  local selected=""
  for process in /proc/[0-9]*; do
    pid="${process##*/}"
    [[ -r "$process/cmdline" ]] || continue
    command="$(tr '\0' ' ' <"$process/cmdline" 2>/dev/null || true)"
    case "$command" in
      *cmix*"-e enwik9"*) selected="$pid" ;;
    esac
  done
  printf '%s' "$selected"
}

show_once() {
  local phase
  local result_root
  local pid
  phase="$(state_value phase)"
  result_root="$(state_value result_root)"
  [[ -n "$phase" ]] || phase=unknown

  echo "WSL UTC: $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "India:   $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Phase:   $phase"
  architecture="$(state_value architecture)"
  [[ -n "$architecture" ]] && echo "Choice:  $architecture"
  ppm_rss_mb="$(state_value ppm_rss_mb)"
  [[ -n "$ppm_rss_mb" ]] && echo "PPM RSS purge trigger: $ppm_rss_mb MiB"

  pid="$(find_cmix_pid)"
  if [[ -z "$pid" ]]; then
    echo
    echo "No active FX4 entropy process."
    if [[ "$phase" == complete && -n "$result_root" &&
          -f "$result_root/final_score.txt" ]]; then
      cat "$result_root/final_score.txt"
    fi
    return
  fi

  echo
  ps -p "$pid" -o pid,psr,%cpu,%mem,etime,time,rss,cmd
  elapsed="$(ps -o etimes= -p "$pid" | xargs)"
  input_path=""
  input_pos=0
  input_size=0
  output_path=""
  output_pos=0

  for link in /proc/"$pid"/fd/*; do
    path="$(readlink "$link" 2>/dev/null || true)"
    path="$(printf '%s' "$path" | sed 's/ (deleted)$//')"
    fd="$(basename "$link")"
    pos="$(fd_position "$pid" "$fd")"
    size="$(stat -Lc%s "$link" 2>/dev/null || printf 0)"
    case "$path" in
      "$run_root/"*.cmix.temp)
        if (( size > input_size )); then
          input_path="$path"
          input_pos="$pos"
          input_size="$size"
        fi
        ;;
      "$run_root/cmix_payload"|"$run_root/payload_"*)
        if (( pos >= output_pos )); then
          output_path="$path"
          output_pos="$pos"
        fi
        ;;
    esac
  done

  if (( input_size > 0 )); then
    progress="$(awk -v p="$input_pos" -v s="$input_size" \
      'BEGIN {printf "%.4f", s ? 100*p/s : 0}')"
    echo
    echo "Entropy input: $input_path"
    echo "Entropy progress: $progress% ($input_pos/$input_size bytes)"
    if (( input_pos > 0 && elapsed > 0 )); then
      awk -v e="$elapsed" -v p="$input_pos" -v s="$input_size" \
        'BEGIN {
          total=e*s/p; remaining=total-e;
          printf "Projected active time: %.2f h total, %.2f h remaining\n",
              total/3600, remaining/3600
        }'
    fi
  else
    echo
    echo "Stage has not opened the post-R1 entropy stream yet."
  fi

  if [[ "$phase" == compress && "$input_pos" -gt 0 &&
        "$output_pos" -gt 0 ]]; then
    fixed=0
    for file in .decomp_bin .dict.comp .tfweights test.dat; do
      [[ -f "$run_root/$file" ]] &&
        fixed=$((fixed + $(stat -c%s "$run_root/$file")))
    done
    s1="$(state_value s1_bytes)"
    [[ "$s1" =~ ^[0-9]+$ ]] || s1="$(stat -c%s "$run_root/cmix")"
    echo
    awk -v out="$output_pos" -v pos="$input_pos" -v total="$input_size" \
        -v fixed="$fixed" -v s1="$s1" '
      BEGIN {
        bpb=8*out/pos;
        payload=out*total/pos;
        archive=payload+fixed;
        score=archive+s1;
        printf "Current entropy rate: %.6f bpb\n", bpb;
        printf "Projected cmix payload: %.0f bytes\n", payload;
        printf "Projected archive9:     %.0f bytes\n", archive;
        printf "Projected S1 + S2:      %.0f bytes\n", score;
        printf "Delta vs 93,000,000:    %+.0f bytes\n", score-93000000;
        printf "Delta vs 93,434,410:    %+.0f bytes\n", score-93434410;
        if (100*pos/total < 2)
          print "Projection warning: under 2% complete; early rate is unstable.";
      }'
    echo "Current payload FD: $output_path ($output_pos bytes written)"
    echo "Fixed archive envelope: $fixed bytes"
  fi

  if [[ "$phase" == discovery_* && -n "$result_root" ]]; then
    discovery_phase="${phase#discovery_}"
    status="$result_root/search.$discovery_phase.winner.status"
    selected="$result_root/search.$discovery_phase.portfolio_selected.csv"
    echo
    if [[ -f "$status" ]]; then
      echo "Discovery status:"
      sed 's/^/  /' "$status"
    fi
    if [[ -f "$selected" ]]; then
      completed=$(( $(wc -l <"$selected") - 1 ))
      winners="$(awk -F, 'NR > 1 && $20 == "accepted_after_side" {n++}
        END {print n+0}' "$selected")"
      echo "Completed recipients: $completed"
      echo "Standalone winners:   $winners"
    fi
  fi

  echo
  [[ -f "$run_root/ppm.temp" ]] &&
    echo "ppm.temp: $(stat -c%s "$run_root/ppm.temp") bytes"
  free -h
}

while true; do
  if [[ "$once" != 1 ]]; then clear || true; fi
  show_once
  [[ "$once" == 1 ]] && break
  [[ "$(state_value phase)" == complete ]] && break
  sleep "$refresh"
done
