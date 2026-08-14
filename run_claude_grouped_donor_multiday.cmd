@echo off
setlocal EnableExtensions

set "TRIALS=%~1"
if not defined TRIALS set "TRIALS=25"

set "CPU=%~2"
if not defined CPU set "CPU=7"

set "ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "STREAM=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin"
set "SEED=%ROOT%/results/donor_multiday_seed"
set "PACKS=%SEED%/enwik9.grouped_1mib_recipients.csv"
set "CANDIDATES=%ROOT%/results/selective_group_campaign/enwik9.grouped_selective_candidates.csv"
set "RECIPIENTS=%CANDIDATES%.recipients.csv"
set "RESULT=%ROOT%/results/claude_donor_grouped_selective"

echo FX4 page-aligned 1 MiB donor search
echo   Recipient groups:             531
echo   New exact trials this batch: %TRIALS%
echo   WSL CPU:                     %CPU%
echo   Durable result directory:   %RESULT%
echo.

wsl -d Ubuntu -- bash -lc "cd '%ROOT%' && FX4_DONOR_RESULT_ROOT='%RESULT%' FX4_WINNER_PAGE_CANDIDATES_CSV='%CANDIDATES%' FX4_WINNER_RECIPIENTS_CSV='%RECIPIENTS%' FX4_WINNER_USE_RANKED=1 FX4_WINNER_RERANK=0 FX4_WINNER_PLANNED_ONLY=0 FX4_WINNER_PLANNED_DONORS=7 FX4_WINNER_MAX_DEPTH=8 FX4_WINNER_BEAM_WIDTH=4 FX4_WINNER_COMBO_CANDIDATES=12 FX4_WINNER_CANDIDATES=19 FX4_WINNER_NEAR_BYTES=64 FX4_WINNER_LEAVE_ONE_OUT=1 FX4_DONOR_MAX_NEW_TRIALS=%TRIALS% FX4_DONOR_CPU=%CPU% FX4_DONOR_TRIAL_CPU=%CPU% bash tools/run_page_donor_specialist_ext4.sh '%STREAM%' '%PACKS%' '%SEED%/page_7donor_full.f4cd' 560"

set "STATUS=%ERRORLEVEL%"
if "%STATUS%"=="0" (
  echo.
  echo Batch completed. Run this same command again to continue.
) else (
  echo.
  echo Batch stopped with exit code %STATUS%. Completed rows remain durable.
  echo The current physical-page sweep must be stopped before this runner can own the WSL work lock.
)
exit /b %STATUS%
