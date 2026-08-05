@echo off
setlocal
set "FX4_WINNER_START_REGION=421"
set "FX4_WINNER_MAX_REGIONS=1"
call "%~dp0run_postr1_warm_full.cmd" %*
exit /b %ERRORLEVEL%
