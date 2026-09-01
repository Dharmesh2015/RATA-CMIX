@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "FX4_REPO=%%~fI"
for /f "delims=" %%I in ('wsl -d Ubuntu -- wslpath -a "%FX4_REPO%\tools\monitor_target93_selective.sh"') do set "WSL_MONITOR=%%I"

set "REFRESH=%~1"
if "%REFRESH%"=="" set "REFRESH=300"
if not defined FX4_WORK_ROOT set "FX4_WORK_ROOT=/root/fx4_target93_selective_full"

wsl -d Ubuntu -- bash -lc "exec bash '%WSL_MONITOR%' '%FX4_WORK_ROOT%' '%REFRESH%'"
exit /b %errorlevel%
