@echo off
setlocal
REM Encode a 1%%-scale prefix, decode it back, confirm ROUNDTRIP=IDENTICAL.
REM
REM   tools\run_1pct_roundtrip.cmd                 full 1%%, shipped config
REM   set LIMIT=65536 && tools\run_1pct_roundtrip.cmd   a quick smoke test
REM   set BENCH_LTO=0 && set BENCH_PGO=0 && set BENCH_F16=0 && ^
REM     tools\run_1pct_roundtrip.cmd                like-for-like vs v521
REM
REM S1 (the compressor binary) does not scale with input size -- it is a
REM fixed artifact from build_and_construct_comp.sh, same bytes at 1%% or
REM 100%%. This measures S2 (the archive) at 1%% scale AND verifies the
REM round trip, via the same Predictor/Encoder/Decoder the real codec uses.
REM
REM A real S1-based `-e`/self-extract round trip cannot run on a prefix --
REM confirmed directly: article reorder needs the genuine, complete enwik9
REM article set. That is a full-enwik9-only test
REM (tools\run_google_cloud_hutter.sh already does it, hours-to-days). This
REM is the fast WSL alternative for everything short of that.
REM
REM About 100 minutes for the full 1%% (encode + decode, roughly double a
REM plain run_1pct.cmd). Blocks on purpose -- keep this window open.
set REPO=/mnt/d/mywork/myideas/latestcompressor/RATA-CMIX
set STREAM=/root/fx4donor/ready.pre_r1.bin
set BRANCH=main
if "%CKPT%"=="" set CKPT=65536
if "%LIMIT%"=="" set LIMIT=5871388
if "%BENCH_LTO%"=="" set BENCH_LTO=1
if "%BENCH_PGO%"=="" set BENCH_PGO=1
if "%BENCH_F16%"=="" set BENCH_F16=1
echo === 1%% round trip: %BRANCH% ===
echo lto=%BENCH_LTO% pgo=%BENCH_PGO% f16=%BENCH_F16% limit=%LIMIT%
wsl -d Ubuntu -- bash -lc "set -e; cp %REPO%/tools/run_1pct_roundtrip.sh /tmp/gch_rt.sh; test -s /tmp/gch_rt.sh || { echo 'FATAL: run_1pct_roundtrip.sh is empty in the repo -- re-checkout the branch' >&2; exit 1; }; bash -n /tmp/gch_rt.sh; BRANCH=%BRANCH% STREAM=%STREAM% LIMIT=%LIMIT% CKPT=%CKPT% BENCH_LTO=%BENCH_LTO% BENCH_PGO=%BENCH_PGO% BENCH_F16=%BENCH_F16% bash /tmp/gch_rt.sh"
exit /b %ERRORLEVEL%
