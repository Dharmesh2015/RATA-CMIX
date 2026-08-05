@echo off
setlocal
if not defined FX4_CRAWLER_HOURS set "FX4_CRAWLER_HOURS=46"
if not defined FX4_CRAWLER_MAX_REGIONS set "FX4_CRAWLER_MAX_REGIONS=48"
if not defined FX4_CRAWLER_PROFILES set "FX4_CRAWLER_PROFILES=4"
if not defined FX4_CRAWLER_BANK set "FX4_CRAWLER_BANK=8"
if not defined FX4_CRAWLER_PROXY_PER_REGION set "FX4_CRAWLER_PROXY_PER_REGION=2"
if not defined FX4_CRAWLER_CPU set "FX4_CRAWLER_CPU=7"

echo FX4 donor/SCR2 crawler
echo   Time budget: %FX4_CRAWLER_HOURS% hours
echo   Region cap:  %FX4_CRAWLER_MAX_REGIONS%
echo   Profile bank: %FX4_CRAWLER_BANK%
echo   CPU:          %FX4_CRAWLER_CPU%
echo   Resume:      automatic from research\donor_crawler_98090\trials.csv
echo.

wsl -d Ubuntu -- bash -lc "cd /mnt/d/mywork/myideas/latestcompressor/fx4-cmix && FX4_CRAWLER_HOURS='%FX4_CRAWLER_HOURS%' FX4_CRAWLER_MAX_REGIONS='%FX4_CRAWLER_MAX_REGIONS%' FX4_CRAWLER_PROFILES='%FX4_CRAWLER_PROFILES%' FX4_CRAWLER_BANK='%FX4_CRAWLER_BANK%' FX4_CRAWLER_PROXY_PER_REGION='%FX4_CRAWLER_PROXY_PER_REGION%' FX4_CRAWLER_CPU='%FX4_CRAWLER_CPU%' bash tools/run_donor_crawler_98090_ext4.sh /mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo Crawler stopped with exit code %RC%. Completed rows are durable.
exit /b %RC%
