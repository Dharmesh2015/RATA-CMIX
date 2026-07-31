@echo off
setlocal EnableExtensions

wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/tools/status_donor_9819_all_edges.sh
exit /b %ERRORLEVEL%
