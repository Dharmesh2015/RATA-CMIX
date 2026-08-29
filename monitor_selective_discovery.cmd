@echo off
wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix-discovery/tools/monitor_selective_discovery_ext4.sh /mnt/d/mywork/myideas/latestcompressor/fx4-cmix-discovery/results/selective_discovery_winner_first --watch
exit /b %ERRORLEVEL%
