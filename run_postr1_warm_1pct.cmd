@echo off
setlocal
set "POSTR1=%~1"
if "%POSTR1%"=="" set "POSTR1=D:\mywork\myideas\latestcompressor\enwik9.post_r1.bin"
wsl -d Ubuntu -- bash -lc "p=\$(wslpath -u '%POSTR1%'); cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && bash tools/run_postr1_warm_portfolio_ext4.sh \"\$p\" 6"
exit /b %ERRORLEVEL%
