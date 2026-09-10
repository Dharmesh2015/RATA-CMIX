#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
input="${1:-}"
run_dir="${2:-$HOME/ratarun}"
cpu="${3:-0}"

if [[ -z "$input" || ! -s "$input" ]]; then
  echo "usage: $0 /path/to/enwik9 [run-directory] [cpu]" >&2
  exit 2
fi
if [[ ! "$cpu" =~ ^[0-9]+$ ]]; then
  echo "cpu must be a nonnegative integer" >&2
  exit 2
fi
test -s "$root/cmix"
if [[ -e "$run_dir" ]]; then
  echo "refusing to reuse an existing run directory: $run_dir" >&2
  exit 2
fi

mkdir -p "$run_dir"
install -m 0555 "$root/cmix" "$run_dir/cmix"
cp --reflink=auto "$input" "$run_dir/enwik9"
cd "$run_dir"

fs_type="$(stat -f -c %T .)"
case "$fs_type" in
  ext2/ext3|ext2/ext3/ext4|xfs) ;;
  *) echo "warning: $fs_type is not the recommended local ext4/xfs filesystem" >&2 ;;
esac

available="$(df --output=avail -B1 . | tail -1 | tr -d ' ')"
if (( available < 25000000000 )); then
  echo "at least 25 GB free local disk is required" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf 'Run directory: %s\n' "$run_dir"
printf 'Filesystem:    %s\n' "$fs_type"
printf 'CPU:           %s (single core)\n' "$cpu"
printf 'Monitor:       %s/tools/monitor_hutter_run.sh %s 60\n' \
  "$root" "$run_dir"

/usr/bin/time -v taskset -c "$cpu" ./cmix -e enwik9 archive9 \
  2>&1 | tee compression.log
test -s archive9
sha256sum archive9 > archive9.sha256
