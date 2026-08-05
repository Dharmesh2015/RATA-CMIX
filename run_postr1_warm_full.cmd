@echo off
setlocal
set "POSTR1=%~1"
if "%POSTR1%"=="" set "POSTR1=D:\mywork\myideas\latestcompressor\enwik9.post_r1.bin"
if "%FX4_WINNER_MAX_REGIONS%"=="" set "FX4_WINNER_MAX_REGIONS=10"
if "%FX4_WINNER_START_REGION%"=="" set "FX4_WINNER_START_REGION=1"
wsl -d Ubuntu -- bash -lc "p=\$(wslpath -u '%POSTR1%'); cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && FX4_WARM_RESULT_ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix/research/postr1_warm_full_v1 FX4_WINNER_START_REGION=%FX4_WINNER_START_REGION% FX4_WINNER_MAX_REGIONS=%FX4_WINNER_MAX_REGIONS% bash tools/run_postr1_warm_portfolio_ext4.sh \"\$p\" 559"
exit /b %ERRORLEVEL%
