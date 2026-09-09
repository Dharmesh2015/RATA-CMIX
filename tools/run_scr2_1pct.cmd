@echo off
setlocal
REM Does SCR2 pay on this stream once the CODER is asked, not the byte count?
REM
REM   tools\run_scr2_1pct.cmd                 code the transformed 1%% prefix
REM   set CKPT=8192 && tools\run_scr2_1pct.cmd    finer checkpoints
REM   set BENCH_F16=0 && tools\run_scr2_1pct.cmd  match the 673,813 build exactly
REM
REM The transform's own accounting says it wins big: 5,871,388 bytes of our 1%%
REM prefix become 5,117,435 plus 1,232 of metadata, a 12.82%% raw saving. That
REM counts bytes REMOVED. It does not count what the coder was already paying
REM for them -- for a pattern that appears 3,835 times, a fraction of a bit
REM rather than eight bytes.
REM
REM So this codes the transformed stream with the same binary, weights and
REM flags as the baseline and compares TOTAL cost to reconstruct the same
REM 5,871,388 original bytes:
REM
REM   baseline   archive(5,871,388 original)                = 673,813
REM   scr2       archive(5,117,435 transformed) + metadata
REM
REM Roughly 55 minutes -- the stream is 13%% shorter than the baseline's.
REM
REM Requires the transform outputs at /root/scr2_ours/ours1pct.{bin,meta},
REM already produced. To regenerate them:
REM   python3 special_scanner/postr1_structural_codec/postr1_structural_codec.py ^
REM     run prefix1pct.bin --prefix ours1pct
set REPO=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
set BRANCH=exp/gch-optimization
if "%CKPT%"=="" set CKPT=65536
if "%BENCH_LTO%"=="" set BENCH_LTO=1
if "%BENCH_PGO%"=="" set BENCH_PGO=1
if "%BENCH_F16%"=="" set BENCH_F16=1
if "%SCR2_DIR%"=="" set SCR2_DIR=/root/scr2_ours
echo === SCR2 1%% test: %BRANCH% ===
REM Staged onto WSL's filesystem first: DrvFs can hand bash a short read of a
REM script Windows has just rewritten.
wsl -d Ubuntu -- bash -lc "set -e; cp %REPO%/tools/run_scr2_1pct.sh /tmp/scr2_run.sh; test -s /tmp/scr2_run.sh; bash -n /tmp/scr2_run.sh; BRANCH=%BRANCH% SCR2_DIR=%SCR2_DIR% CKPT=%CKPT% BENCH_LTO=%BENCH_LTO% BENCH_PGO=%BENCH_PGO% BENCH_F16=%BENCH_F16% bash /tmp/scr2_run.sh"
exit /b %ERRORLEVEL%
