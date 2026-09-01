@echo off
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "FX4_REPO=%%~fI"
if "%~1"=="" (
  set "FX4_INPUT=%FX4_REPO%\..\enwik9"
) else (
  set "FX4_INPUT=%~f1"
)
if "%~2"=="" (
  set "FX4_OUTPUT=%FX4_REPO%\..\archive9_target93_selective"
) else (
  set "FX4_OUTPUT=%~f2"
)

if not exist "%FX4_INPUT%" (
  echo Input not found: %FX4_INPUT%
  exit /b 2
)

for /f "delims=" %%I in ('wsl -d Ubuntu -- wslpath -a "%FX4_INPUT%"') do set "WSL_INPUT=%%I"
for /f "delims=" %%I in ('wsl -d Ubuntu -- wslpath -a "%FX4_OUTPUT%"') do set "WSL_OUTPUT=%%I"
for /f "delims=" %%I in ('wsl -d Ubuntu -- wslpath -a "%FX4_REPO%\tools\run_target93_selective_full_ext4.sh"') do set "WSL_SCRIPT=%%I"

if not defined FX4_TEST_CPU set "FX4_TEST_CPU=7"
if not defined FX4_BUILD_JOBS set "FX4_BUILD_JOBS=3"
if not defined FX4_PPM_RSS_MB set "FX4_PPM_RSS_MB=8704"
if not defined FX4_VALIDATE_ROUNDTRIP set "FX4_VALIDATE_ROUNDTRIP=1"
if not defined FX4_COOLDOWN_SECONDS set "FX4_COOLDOWN_SECONDS=120"

echo Full run input:   %FX4_INPUT%
echo Export archive:  %FX4_OUTPUT%
echo CPU core:        %FX4_TEST_CPU%
echo PPM RSS trigger: %FX4_PPM_RSS_MB% MiB
echo.

wsl -d Ubuntu -- bash -lc "export FX4_TEST_CPU='%FX4_TEST_CPU%' FX4_BUILD_JOBS='%FX4_BUILD_JOBS%' FX4_PPM_RSS_MB='%FX4_PPM_RSS_MB%' FX4_VALIDATE_ROUNDTRIP='%FX4_VALIDATE_ROUNDTRIP%' FX4_COOLDOWN_SECONDS='%FX4_COOLDOWN_SECONDS%'; exec bash '%WSL_SCRIPT%' '%WSL_INPUT%' '%WSL_OUTPUT%'"
exit /b %errorlevel%
