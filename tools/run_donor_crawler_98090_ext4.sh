#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 ENWIK9_POST_R1_BIN" >&2
  exit 2
fi

readonly source_root=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
readonly work_root=/root/fx4crawler_98090
readonly runtime_root="$work_root/run"
readonly result_root="${FX4_CRAWLER_RESULTS:-$source_root/research/donor_crawler_98090}"
readonly corpus_source="$(readlink -f "$1")"
readonly corpus="$work_root/enwik9.post_r1.bin"
readonly seed_csv="$source_root/research/donor_graph_post_r1_20260727/winner_421_multidonor.csv"
readonly proxy_csv="$source_root/research/donor_graph_post_r1_20260727/exact_mesh_all_candidates.csv"
readonly scr2_tool=/mnt/d/mywork/myideas/latestcompressor/special_scanner/postr1_structural_codec/postr1_structural_codec.py
readonly cpu="${FX4_CRAWLER_CPU:-7}"

test -f "$corpus_source"
test -f "$seed_csv"
test -f "$proxy_csv"
test -f "$scr2_tool"
if [[ "$(stat -c%s "$corpus_source")" -ne 587138826 ]]; then
  echo "expected the canonical 587,138,826-byte post-R1 stream" >&2
  exit 2
fi

mkdir -p "$work_root" "$runtime_root" "$result_root"
exec 9>"$work_root/crawler.lock"
if ! flock -n 9; then
  echo "another donor crawler owns $work_root" >&2
  exit 3
fi

rm -rf -- "$work_root/source.new"
mkdir -p "$work_root/source.new"
cp -a "$source_root/src" "$source_root/dictionary" "$source_root/tools" \
  "$work_root/source.new/"
cp "$source_root/makefile" "$work_root/source.new/"
find "$work_root/source.new" -type f \( -name '*.sh' -o -name 'makefile' \) \
  -exec sed -i 's/\r$//' {} +
rm -rf -- "$work_root/source"
mv "$work_root/source.new" "$work_root/source"

cd "$work_root/source"
make clean
make cmix -j"$(nproc)" OUT=cmix_crawler \
  CFLAGS_DEFINES='-DSEED=923 -DUPDATE_LIMIT=3000 -DNDEBUG -DFX4_STDERR_PROGRESS=0 -DFX4_PROGRESS_LOG=0 -DFX4_LSTM_MID_BRIDGE=2 -DFX4_RESEARCH_DONOR_BOOTSTRAP=1'
cp cmix_crawler "$work_root/cmix_crawler"
chmod 0755 "$work_root/cmix_crawler"

if [[ ! -f "$corpus" ]] || [[ "$(stat -c%s "$corpus")" -ne 587138826 ]]; then
  cp "$corpus_source" "$corpus.new"
  mv "$corpus.new" "$corpus"
fi
readonly corpus_hash="$(sha256sum "$corpus" | cut -d' ' -f1)"
if [[ "$corpus_hash" != 7826ff63dedd526c119dda08e6e044be8fa8f6e89a55f3d6b1f3447cdfc5c1ce ]]; then
  echo "canonical post-R1 SHA-256 mismatch: $corpus_hash" >&2
  exit 2
fi

readonly metadata="$result_root/run.meta"
readonly metadata_new="$result_root/run.meta.new"
{
  echo "corpus_sha256=$corpus_hash"
  echo "crawler_binary_sha256=$(sha256sum "$work_root/cmix_crawler" | cut -d' ' -f1)"
  echo "crawler_script_sha256=$(sha256sum "$work_root/source/tools/donor_scr2_crawler.py" | cut -d' ' -f1)"
  echo "scr2_tool_sha256=$(sha256sum "$scr2_tool" | cut -d' ' -f1)"
  echo "seed_sha256=$(sha256sum "$seed_csv" | cut -d' ' -f1)"
  echo "proxy_sha256=$(sha256sum "$proxy_csv" | cut -d' ' -f1)"
} >"$metadata_new"
if [[ -f "$metadata" ]] && ! cmp -s "$metadata" "$metadata_new"; then
  echo "existing crawler ledger belongs to a different binary or input" >&2
  diff -u "$metadata" "$metadata_new" >&2 || true
  rm -f "$metadata_new"
  exit 2
fi
mv -f "$metadata_new" "$metadata"

echo "Bounded donor/SCR2 crawler"
echo "  seed: seven donors / historical 98,090-byte isolated calibration"
echo "  corpus: canonical post-R1 $corpus_hash"
echo "  results: $result_root"
echo "  budget: ${FX4_CRAWLER_HOURS:-46} hours"
echo "  regions: ${FX4_CRAWLER_MAX_REGIONS:-48}"
echo "  profiles/region: ${FX4_CRAWLER_PROFILES:-4}"
echo "  profile bank: ${FX4_CRAWLER_BANK:-8} (seven-donor seed is pinned)"
echo "  donor lengths: 256B..64KiB (power-of-two proxy screen)"

cd "$runtime_root"
exec nice -n -10 python3 "$work_root/source/tools/donor_scr2_crawler.py" \
  --corpus "$corpus" \
  --cmix "$work_root/cmix_crawler" \
  --dictionary "$work_root/source/dictionary/english.dic" \
  --seed-csv "$seed_csv" \
  --proxy-csv "$proxy_csv" \
  --scr2-tool "$scr2_tool" \
  --work-dir "$runtime_root" \
  --results "$result_root" \
  --hours "${FX4_CRAWLER_HOURS:-46}" \
  --max-regions "${FX4_CRAWLER_MAX_REGIONS:-48}" \
  --max-profiles-per-region "${FX4_CRAWLER_PROFILES:-4}" \
  --profile-bank-size "${FX4_CRAWLER_BANK:-8}" \
  --proxy-per-recipient "${FX4_CRAWLER_PROXY_PER_REGION:-2}" \
  --trial-timeout "${FX4_CRAWLER_TRIAL_TIMEOUT:-1800}" \
  --cpu "$cpu"
