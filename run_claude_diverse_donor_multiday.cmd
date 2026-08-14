@echo off
setlocal EnableExtensions

set "TRIALS=%~1"
if not defined TRIALS set "TRIALS=100"

set "CPU=%~2"
if not defined CPU set "CPU=7"

set "RECIPIENTS_TO_SCAN=%~3"
if not defined RECIPIENTS_TO_SCAN set "RECIPIENTS_TO_SCAN=64"

set "ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "STREAM=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin"
set "SEED=%ROOT%/results/donor_multiday_seed"
set "PACKS=%SEED%/enwik9.grouped_1mib_recipients.csv"
set "CAMPAIGN=%ROOT%/results/diverse_group_campaign"
set "CANDIDATES=%CAMPAIGN%/enwik9.grouped_diverse_candidates.csv"
set "RECIPIENTS=%CANDIDATES%.recipients.csv"
set "RESULT=%ROOT%/results/claude_donor_grouped_diverse"

if not exist "%~dp0results\diverse_group_campaign\campaign.json" (
  echo Missing diverse campaign. Run prepare_diverse_donor_campaign.cmd first.
  exit /b 2
)

echo FX4 source-diverse post-R1 donor breadth sweep
echo   Accepted G388 source:        excluded from this discovery pass
echo   Recipients:                  all eligible page-aligned groups
echo   Exact trials this batch:    %TRIALS%
echo   Priority recipients active: %RECIPIENTS_TO_SCAN%
echo   WSL CPU:                    %CPU%
echo   Durable result directory:  %RESULT%
echo.

wsl -d Ubuntu -- bash -lc "cd '%ROOT%' && FX4_DONOR_RESULT_ROOT='%RESULT%' FX4_WINNER_PAGE_CANDIDATES_CSV='%CANDIDATES%' FX4_WINNER_RECIPIENTS_CSV='%RECIPIENTS%' FX4_WINNER_RECIPIENT_LIMIT=%RECIPIENTS_TO_SCAN% FX4_WINNER_USE_RANKED=1 FX4_WINNER_RERANK=0 FX4_WINNER_PLANNED_ONLY=1 FX4_WINNER_PLANNED_PREFIXES=1 FX4_WINNER_PLANNED_DONORS=8 FX4_WINNER_LEAVE_ONE_OUT=1 FX4_WINNER_CANDIDATES=1 FX4_WINNER_MAX_REGIONS=0 FX4_DONOR_MAX_NEW_TRIALS=%TRIALS% FX4_DONOR_CPU=%CPU% FX4_DONOR_TRIAL_CPU=%CPU% bash tools/run_page_donor_specialist_ext4.sh '%STREAM%' '%PACKS%' '%SEED%/page_7donor_full.f4cd' 560"
set "STATUS=%ERRORLEVEL%"

if exist "%~dp0results\claude_donor_grouped_diverse\page.winner_trials.csv" (
  python "%~dp0tools\summarize_diverse_donor_results.py" ^
    "%~dp0results\donor_multiday_seed\enwik9.grouped_1mib_recipients.csv" ^
    "%~dp0results\claude_donor_grouped_diverse\page" ^
    "%~dp0results\claude_donor_grouped_diverse"
)

if "%STATUS%"=="0" (
  echo Diverse donor breadth sweep completed.
) else (
  echo Sweep paused or stopped with exit code %STATUS%.
  echo Exact rows already written under results\claude_donor_grouped_diverse are durable.
  echo Rerun this command to resume.
)
exit /b %STATUS%
