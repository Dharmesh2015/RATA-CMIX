@echo off
setlocal EnableExtensions
if "%~1"=="" (
  set "INPUT=D:/mywork/myideas/latestcompressor/enwik9"
) else (
  set "INPUT=%~f1"
)
if not exist "%INPUT%" (
  echo Input not found: %INPUT%
  exit /b 2
)
for /f "usebackq delims=" %%I in (`wsl -d Ubuntu -- wslpath -a "%INPUT%"`) do set "WSL_INPUT=%%I"
if "%~2"=="" (
  wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix-discovery/tools/run_selective_discovery_ext4.sh "%WSL_INPUT%"
) else (
  for /f "usebackq delims=" %%I in (`wsl -d Ubuntu -- wslpath -a "%~f2"`) do set "WSL_RESULT=%%I"
  wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix-discovery/tools/run_selective_discovery_ext4.sh "%WSL_INPUT%" "%WSL_RESULT%"
)
exit /b %ERRORLEVEL%