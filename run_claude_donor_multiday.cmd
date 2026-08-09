@echo off
setlocal EnableExtensions

set "TRIALS=%~1"
if not defined TRIALS set "TRIALS=25"

set "CPU=%~2"
if not defined CPU set "CPU=7"

set "ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "STREAM=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin"
set "SEED=%ROOT%/results/donor_multiday_seed"
set "RESULT=%ROOT%/results/claude_donor_multiday"

echo FX4 selective donor search
echo   New exact trials this batch: %TRIALS%
echo   WSL CPU:                    %CPU%
echo   Durable result directory:  %RESULT%
echo.

wsl -d Ubuntu -- bash -lc "cd '%ROOT%' && FX4_DONOR_RESULT_ROOT='%RESULT%' FX4_WINNER_PAGE_CANDIDATES_CSV='%SEED%/page_candidates.csv' FX4_WINNER_USE_RANKED=1 FX4_WINNER_RERANK=0 FX4_WINNER_PLANNED_ONLY=0 FX4_WINNER_PLANNED_DONORS=7 FX4_WINNER_MAX_DEPTH=8 FX4_WINNER_BEAM_WIDTH=4 FX4_WINNER_COMBO_CANDIDATES=8 FX4_WINNER_NEAR_BYTES=64 FX4_WINNER_LEAVE_ONE_OUT=1 FX4_DONOR_MAX_NEW_TRIALS=%TRIALS% FX4_DONOR_CPU=%CPU% FX4_DONOR_TRIAL_CPU=%CPU% bash tools/run_page_donor_specialist_ext4.sh '%STREAM%' '%SEED%/page_candidates.csv.recipients.csv' '%SEED%/page_7donor_full.f4cd' 559"

set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="0" (
  echo.
  echo Batch completed. Run this same command again to continue.
) else (
  echo.
  echo Batch stopped with exit code %STATUS%. Completed trial rows remain durable.
  echo Rerun unchanged to resume after resolving any reported error.
)
exit /b %STATUS%
