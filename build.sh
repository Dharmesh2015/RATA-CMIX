#!/bin/sh
set -eu

# HutterPrizeJudgingAssistant mounts /entry read-only and starts this script
# as UID/GID 65532 in writable /work with no network.
source_root=/entry
build_root=/work/fx4-build

test -r "$source_root/makefile"
test -r "$source_root/dictionary/english.dic"
test -r "$source_root/src/readalike_prepr/data/new_article_order"
test -r "$source_root/models/transformer6m/6m-q4-fp32.tfwc2"
test ! -e "$build_root"
mkdir -p "$build_root"

cp -a "$source_root/src" "$build_root/src"
cp -a "$source_root/dictionary" "$build_root/dictionary"
cp -a "$source_root/models" "$build_root/models"
cp -a "$source_root/makefile" "$build_root/makefile"
cp -a "$source_root/build_and_construct_comp.sh" \
  "$build_root/build_and_construct_comp.sh"

cd "$build_root"
chmod 0555 build_and_construct_comp.sh
CXX=clang++ ./build_and_construct_comp.sh
test -s cmix
install -m 0555 cmix /work/cmix
