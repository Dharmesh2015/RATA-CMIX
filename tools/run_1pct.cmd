@echo off
setlocal
REM One complete 1%% run of this branch on WSL.
REM
REM   tools\run_1pct.cmd                    the shipped configuration
REM   set CKPT=8192 && tools\run_1pct.cmd   finer checkpoints
REM   set BENCH_LTO=0 && tools\run_1pct.cmd build WITHOUT -flto=thin
REM   set BENCH_PGO=0 && tools\run_1pct.cmd build WITHOUT the PGO profile
REM   set BENCH_F16=0 && tools\run_1pct.cmd build WITHOUT the fp16 LSTM path
REM   set LIMIT=262144 && tools\run_1pct.cmd  a short screen, about 4 minutes
REM
REM Always builds from `git archive`: an uncommitted edit is reported and
REM refused rather than silently measured. Commit first, then run this --
REM there is no mode that tests uncommitted work.
REM
REM There are no arms: this branch compiles one codec. The default build is
REM the SHIPPED one -- LTO, the committed PGO profile, and the fp16 LSTM
REM path all on -- because measuring a build nobody ships measures the
REM wrong thing.
REM
REM Set all three to 0 for a like-for-like comparison against
REM fx2-cmix-transformer-v521's completed 673,799. That configuration is
REM measured at 673,813, so this branch's optimization tier costs 14 bytes
REM and returns 32.5%% of the entropy time.
REM
REM Prints a checkpoint every 65536 bytes:
REM   progress_bytes=65536 payload=5813 seconds=82.262
REM and ends with FINAL plus the delta against both completed references.
REM
REM About 100 minutes for the full 1%%. Blocks on purpose -- WSL2 tears its VM
REM down once no client is attached, so keep this window open.
set REPO=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix
set STREAM=/root/fx4donor/ready.pre_r1.bin
set BRANCH=release/google-cloud-hutter
if "%CKPT%"=="" set CKPT=65536
if "%LIMIT%"=="" set LIMIT=5871388
if "%BENCH_LTO%"=="" set BENCH_LTO=1
if "%BENCH_PGO%"=="" set BENCH_PGO=1
if "%BENCH_F16%"=="" set BENCH_F16=1
echo === 1%% run: %BRANCH% ===
echo lto=%BENCH_LTO% pgo=%BENCH_PGO% f16=%BENCH_F16% limit=%LIMIT%
REM cmd.exe environment variables persist for the rest of an interactive
REM session, so a `set FOO=1` typed for an earlier run is still there for
REM this one unless every variable is set again explicitly. The line above
REM exists so a stale value like that is caught here, not 100 minutes into
REM a run that silently built the wrong configuration. If anything above
REM does not match what this command line set, open a fresh cmd window or
REM set every variable explicitly rather than relying on defaults.
REM Staged onto WSL's filesystem first: DrvFs can hand bash a short read of a
REM script Windows has just rewritten, which surfaces as
REM   error reading input file: No data available
wsl -d Ubuntu -- bash -lc "set -e; cp %REPO%/tools/run_1pct.sh /tmp/gch_run.sh; test -s /tmp/gch_run.sh; bash -n /tmp/gch_run.sh; BRANCH=%BRANCH% STREAM=%STREAM% LIMIT=%LIMIT% CKPT=%CKPT% BENCH_LTO=%BENCH_LTO% BENCH_PGO=%BENCH_PGO% BENCH_F16=%BENCH_F16% bash /tmp/gch_run.sh"
exit /b %ERRORLEVEL%
