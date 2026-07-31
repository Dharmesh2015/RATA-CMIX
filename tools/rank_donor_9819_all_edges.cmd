@echo off
setlocal EnableExtensions

set "DISTRO=Ubuntu"
set "WSL_ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "WSL_RESULT=%WSL_ROOT%/research/donor_exact_mesh_full_9819_v1"
set "WIN_RESULT=D:\mywork\myideas\latestcompressor\fx4-cmix\research\donor_exact_mesh_full_9819_v1"

if not exist "%WIN_RESULT%\edges.csv" (
  echo No donor ledger exists yet: %WIN_RESULT%\edges.csv
  exit /b 2
)

wsl -d %DISTRO% -- bash -lc "cd '%WSL_ROOT%' && python3 tools/rank_donor_edges.py '%WSL_RESULT%/edges.csv' --output-dir '%WSL_RESULT%'"
if errorlevel 1 exit /b %ERRORLEVEL%

echo.
echo Complete all-edge reports:
echo   %WIN_RESULT%\all_edges_ranked.csv
echo   %WIN_RESULT%\all_edges_by_recipient.csv
echo   %WIN_RESULT%\donor_rankings.csv
echo   %WIN_RESULT%\recipient_rankings.csv
echo   %WIN_RESULT%\ranking_summary.json

exit /b 0
