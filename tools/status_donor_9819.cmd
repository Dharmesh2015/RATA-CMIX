@echo off
setlocal EnableExtensions

set "DISTRO=Ubuntu"
set "WSL_RESULT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix/research/donor_exact_mesh_full_9819_v1"

wsl -d %DISTRO% -- bash -lc "echo utc_time=$(date -u '+%%Y-%%m-%%dT%%H:%%M:%%SZ'); if test -f '%WSL_RESULT%/edges.csv'; then echo -n tested_edges=; awk 'END{print NR-1}' '%WSL_RESULT%/edges.csv'; else echo tested_edges=0; fi; if test -f '%WSL_RESULT%/edges.csv.winners.csv'; then echo -n positive_edges=; awk 'END{print NR-1}' '%WSL_RESULT%/edges.csv.winners.csv'; else echo positive_edges=0; fi; if test -f '%WSL_RESULT%/edges.csv.selected.csv'; then echo -n completed_recipients=; awk 'END{print NR-1}' '%WSL_RESULT%/edges.csv.selected.csv'; else echo completed_recipients=0; fi; if test -f '%WSL_RESULT%/edges.csv.status'; then echo ---status---; cat '%WSL_RESULT%/edges.csv.status'; fi; if test -f '%WSL_RESULT%/edges.csv.complete'; then echo scan_state=complete; elif test -f '%WSL_RESULT%/edges.csv.paused'; then echo scan_state=paused; else echo scan_state=running_or_not_started; fi; echo ---process---; pgrep -af '[.]\/cmix -e enwik9 discovery_payload' || true; echo ---latest-winners---; if test -f '%WSL_RESULT%/edges.csv.winners.csv'; then tail -n 5 '%WSL_RESULT%/edges.csv.winners.csv'; fi"

exit /b %ERRORLEVEL%
