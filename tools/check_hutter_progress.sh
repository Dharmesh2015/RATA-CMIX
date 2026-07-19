#!/usr/bin/env bash
set -u

RUN_ROOT="/root/fx4run"
REFRESH="${1:-60}"

while true; do
    clear

    echo "WSL UTC: $(date -u '+%Y-%m-%d %H:%M:%S')"
    echo "India:   $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S %Z')"

    pid="$(pgrep -n -x cmix || true)"

    if [[ -z "$pid" ]]; then
        echo "cmix is not running"
        exit 0
    fi

    echo
    ps -p "$pid" -o pid,psr,%cpu,%mem,etime,time,rss,cmd

    elapsed="$(ps -o etimes= -p "$pid" | xargs)"
    cpu_time="$(ps -o cputimes= -p "$pid" | xargs)"

    best_path=""
    best_pos=0
    best_size=0

    for fd in /proc/"$pid"/fd/*; do
        path="$(readlink "$fd" 2>/dev/null || true)"
        path="${path% (deleted)}"

        case "$path" in
            "$RUN_ROOT/enwik9"|\
            "$RUN_ROOT/"*.cmix.temp|\
            "$RUN_ROOT/.ready4cmix"*|\
            "$RUN_ROOT/"*.temp)
                fdnum="${fd##*/}"
                pos="$(awk '/^pos:/ {print $2}' \
                    "/proc/$pid/fdinfo/$fdnum" 2>/dev/null || echo 0)"
                size="$(stat -Lc%s "$fd" 2>/dev/null || echo 0)"

                if [[ "$size" -gt "$best_size" && "$pos" -ge 0 ]]; then
                    best_path="$path"
                    best_pos="$pos"
                    best_size="$size"
                fi
                ;;
        esac
    done

    echo

    if [[ "$best_size" -gt 0 ]]; then
        awk \
            -v path="$best_path" \
            -v pos="$best_pos" \
            -v size="$best_size" \
            -v elapsed="$elapsed" \
            -v cpu="$cpu_time" '
        BEGIN {
            pct = 100 * pos / size

            printf "Entropy input: %s\n", path
            printf "Progress: %.2f%% (%d/%d bytes)\n", pct, pos, size

            if (pos > 0) {
                total = elapsed * size / pos
                remaining = elapsed * (size-pos) / pos
                cpu_total = cpu * size / pos

                printf "Projected active time: %.1f h total\n", total/3600
                printf "Projected remaining: %.1f h\n", remaining/3600
                printf "Projected CPU time: %.1f h\n", cpu_total/3600
            }
        }'
    else
        echo "Stage: preprocessing/reordering"
        echo "Percentage becomes available when entropy coding starts."
    fi

    echo
    echo "Files:"

    for file in \
        "$RUN_ROOT"/cmix_payload* \
        "$RUN_ROOT"/ppm.temp \
        "$RUN_ROOT"/archive9
    do
        [[ -e "$file" ]] || continue
        printf "%-30s %s bytes\n" \
            "${file##*/}" "$(stat -c%s "$file")"
    done

    echo
    free -h

    sleep "$REFRESH"
done