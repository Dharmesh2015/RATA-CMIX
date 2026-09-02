#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
archive="${1:-}"
output_root="${2:-$root/dist}"
entry_name="${3:-FX4}"

if [[ -z "$archive" || ! -s "$archive" ]]; then
  echo "usage: $0 /path/to/archive9 [output-root] [entry-name]" >&2
  exit 2
fi
if [[ ! "$entry_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "entry name must contain only letters, digits, dot, underscore or dash" >&2
  exit 2
fi

archive="$(realpath "$archive")"
entry_dir="$output_root/Entries/$entry_name"
if [[ -e "$entry_dir" ]]; then
  echo "refusing to replace existing entry directory: $entry_dir" >&2
  exit 2
fi

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT HUP INT TERM
source_dir="$stage/fx4-cmix-source"
mkdir -p "$source_dir/src/third_party/fx2_transformer/opt"
mkdir -p "$source_dir/src/readalike_prepr/data"
mkdir -p "$source_dir/models/transformer6m"
mkdir -p "$source_dir/docs"

for file in LICENSE README.md THIRD_PARTY_NOTICES.txt makefile \
    install.sh build.sh build_and_construct_comp.sh comp9.args; do
  cp -a "$root/$file" "$source_dir/$file"
done
cp -a "$root/docs/." "$source_dir/docs/"
cp -a "$root/dictionary" "$source_dir/dictionary"
cp -a "$root/models/transformer6m/6m-q4-fp32.tfwc2" \
  "$source_dir/models/transformer6m/"
if [[ -s "$root/pgo/default.profdata" ]]; then
  mkdir -p "$source_dir/pgo"
  cp -a "$root/pgo/default.profdata" "$source_dir/pgo/"
fi
if [[ -d "$root/prof_input" ]]; then
  cp -a "$root/prof_input" "$source_dir/prof_input"
fi

for dir in coder contexts ds mixer models preprocess states utils; do
  cp -a "$root/src/$dir" "$source_dir/src/$dir"
done
for file in context-manager.cpp context-manager.h fx4_config.h \
    predictor.cpp predictor.h r1_reorder_transform.cpp \
    r1_reorder_transform.h runner.cpp; do
  cp -a "$root/src/$file" "$source_dir/src/$file"
done
for file in article_reorder.h misc.h phda9_preprocess.h self_extract.h; do
  cp -a "$root/src/readalike_prepr/$file" \
    "$source_dir/src/readalike_prepr/$file"
done
cp -a "$root/src/readalike_prepr/data/new_article_order" \
  "$source_dir/src/readalike_prepr/data/"

tf_root="$root/src/third_party/fx2_transformer"
cp -a "$tf_root/LICENSE" "$tf_root/kernels.h" "$tf_root/weights_io.h" \
  "$tf_root/weights_io_compressed.cpp" \
  "$source_dir/src/third_party/fx2_transformer/"
find "$tf_root/opt" -maxdepth 1 -type f -name '*.h' -exec \
  cp -a {} "$source_dir/src/third_party/fx2_transformer/opt/" \;
for file in arena_build.cpp attn.cpp glue.cpp kda.cpp model_opt.cpp \
    qmat_dense.cpp qmat_sparse.cpp; do
  cp -a "$tf_root/opt/$file" \
    "$source_dir/src/third_party/fx2_transformer/opt/$file"
done

chmod 0555 "$source_dir/install.sh" "$source_dir/build.sh" \
  "$source_dir/build_and_construct_comp.sh"
if LC_ALL=C grep -R $'\r' "$source_dir/install.sh" "$source_dir/build.sh" \
    "$source_dir/build_and_construct_comp.sh" "$source_dir/comp9.args"; then
  echo "submission scripts or argument file contain CR bytes" >&2
  exit 1
fi
if [[ "$(wc -c < "$source_dir/comp9.args")" -ne 19 ]]; then
  echo "comp9.args must be exactly 19 LF-only bytes" >&2
  exit 1
fi

(
  cd "$source_dir"
  find . -type f ! -name SOURCE_MANIFEST.sha256 -print0 |
    LC_ALL=C sort -z |
    xargs -0 sha256sum > SOURCE_MANIFEST.sha256
)

mkdir -p "$entry_dir"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "$entry_dir/fx4-cmix-source.tar.gz" -C "$stage" fx4-cmix-source
install -m 0555 "$archive" "$entry_dir/archive9"
cp -a "$root/submission/entry.env" "$entry_dir/entry.env"

top_count="$(tar -tzf "$entry_dir/fx4-cmix-source.tar.gz" |
  cut -d/ -f1 | LC_ALL=C sort -u | wc -l)"
[[ "$top_count" -eq 1 ]]
tar -tzf "$entry_dir/fx4-cmix-source.tar.gz" |
  grep -qx 'fx4-cmix-source/install.sh'
tar -tzf "$entry_dir/fx4-cmix-source.tar.gz" |
  grep -qx 'fx4-cmix-source/build.sh'
tar -tzf "$entry_dir/fx4-cmix-source.tar.gz" |
  grep -qx 'fx4-cmix-source/comp9.args'

printf 'Entry directory: %s\n' "$entry_dir"
printf 'archive9:       %s bytes  %s\n' \
  "$(stat -c%s "$entry_dir/archive9")" \
  "$(sha256sum "$entry_dir/archive9" | cut -d' ' -f1)"
printf 'source package: %s bytes  %s\n' \
  "$(stat -c%s "$entry_dir/fx4-cmix-source.tar.gz")" \
  "$(sha256sum "$entry_dir/fx4-cmix-source.tar.gz" | cut -d' ' -f1)"
