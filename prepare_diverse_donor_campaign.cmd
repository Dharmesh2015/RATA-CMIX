@echo off
setlocal EnableExtensions

set "ROOT=/mnt/d/mywork/myideas/latestcompressor/fx4-cmix"
set "STREAM=/mnt/d/mywork/myideas/latestcompressor/enwik9.post_r1.bin"
set "GROUPS=%ROOT%/results/donor_multiday_seed/enwik9.grouped_1mib_recipients.csv"
set "EDGES=%ROOT%/results/claude_donor_grouped_full/all_group_edges.csv"
set "OUT=%ROOT%/results/diverse_group_campaign"
set "RANKED=%OUT%/ranked_source_diverse.csv"

echo Preparing source-diverse donor candidates over the complete post-R1 stream...
wsl -d Ubuntu -- bash -lc "set -e; cd '%ROOT%'; mkdir -p '%OUT%'; make page_donor_ranker -j$(nproc); ./postr1_page_donor_ranker '%STREAM%' '%GROUPS%' '%RANKED%' 32 587138826; python3 tools/build_diverse_group_campaign.py '%GROUPS%' '%EDGES%' '%RANKED%' '%OUT%' --exclude-source 388 --defer-recipient 421 --defer-recipient 422 --max-sources 12 --per-source 2 --max-candidates 24"
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" exit /b %STATUS%

echo.
echo Campaign ready:
echo   results\diverse_group_campaign\donor_source_rankings.csv
echo   results\diverse_group_campaign\candidate_source_map.csv
echo.
echo Start or resume exact breadth testing with:
echo   run_claude_diverse_donor_multiday.cmd 100 7
exit /b 0
