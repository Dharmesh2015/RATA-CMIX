@echo off
setlocal EnableExtensions

set "DISTRO=Ubuntu"
set "WSL_ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "WSL_ENWIK9=/mnt/d/mywork/myideas/latestcompressor/enwik9"
set "WSL_PLAN=%WSL_ROOT%/research/donor_graph_post_r1_20260727/exact_mesh_all_candidates.f4cd"
set "WSL_RESULT=%WSL_ROOT%/research/donor_exact_mesh_full_9819_v1"
set "WIN_ENWIK9=D:\mywork\myideas\latestcompressor\enwik9"
set "WIN_PLAN=D:\mywork\myideas\latestcompressor\fx4-cmix\research\donor_graph_post_r1_20260727\exact_mesh_all_candidates.f4cd"
set "WIN_RESULT=D:\mywork\myideas\latestcompressor\fx4-cmix\research\donor_exact_mesh_full_9819_v1"

if not defined FX4_BATCH_TRIALS set "FX4_BATCH_TRIALS=250"
if not defined FX4_DONOR_CPU set "FX4_DONOR_CPU=7"

where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo WSL is not installed or is not on PATH.
  exit /b 2
)
if not exist "%WIN_ENWIK9%" (
  echo Missing exact enwik9: %WIN_ENWIK9%
  exit /b 2
)
if not exist "%WIN_PLAN%" (
  echo Missing candidate plan: %WIN_PLAN%
  exit /b 2
)

echo FX4 exhaustive post-R1 donor/recipient scan
echo Candidate edges: 9,819
echo Batch size:      %FX4_BATCH_TRIALS% new exact compression trials
echo CPU:             %FX4_DONOR_CPU%
echo Results:         %WIN_RESULT%
echo.
echo Every tested WIN, TIE, LOSS, and ERROR remains in edges.csv.
echo Ranked reports are rebuilt after every clean batch.
echo No candidate decompression is performed.
echo.
echo Ctrl+C may stop the current batch. Rerun this file to resume.
echo Do not change the source, enwik9, candidate plan, or result directory.
echo Do not run two donor scans concurrently.
echo.

:next_batch
if exist "%WIN_RESULT%\edges.csv.complete" goto complete

wsl -d %DISTRO% -- bash -lc "if pgrep -f '[.]\/cmix -e enwik9 discovery_payload' >/dev/null; then echo 'A donor scan is already running in WSL.' >&2; exit 73; fi"
if errorlevel 1 exit /b %ERRORLEVEL%

wsl -d %DISTRO% -- bash -lc "cd '%WSL_ROOT%' && FX4_DONOR_MAX_NEW_TRIALS=%FX4_BATCH_TRIALS% FX4_DONOR_CPU=%FX4_DONOR_CPU% FX4_DONOR_TRIAL_CPU=%FX4_DONOR_CPU% FX4_DONOR_RESULT_ROOT='%WSL_RESULT%' bash tools/run_exact_donor_mesh_ranked_ext4.sh '%WSL_ENWIK9%' '%WSL_PLAN%'"
set "RC=%ERRORLEVEL%"

if exist "%WIN_RESULT%\edges.csv.complete" goto complete
if not "%RC%"=="0" goto failed

echo.
echo Batch committed. All tested edges have been ranked.
echo The next batch starts in 60 seconds; press Ctrl+C for a clean pause.
timeout /t 60 /nobreak >nul
goto next_batch

:complete
echo.
echo All 9,819 candidate edges have been tested and ranked.
call "%~dp0rank_donor_9819_all_edges.cmd"
exit /b 0

:failed
echo.
echo Donor scan stopped with exit code %RC%.
echo Completed edge rows are durable. Rerun this file to resume.
exit /b %RC%
