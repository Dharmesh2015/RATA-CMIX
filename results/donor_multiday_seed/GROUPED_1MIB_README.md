# Page-aligned 1 MiB donor recipients

The canonical post-R1 page map contains 243,425 physical pages (243,426 CSV
lines including its header). It has been deterministically grouped into 531
contiguous donor-discovery recipients.

| Property | Value |
|---|---:|
| Post-R1 stream bytes | 587,138,826 |
| Target group bytes | 1,048,576 |
| Group count | 531 |
| Minimum group | 383,400 bytes |
| Median group | 1,048,439 bytes |
| Mean group | 1,105,722.84 bytes |
| 95th percentile | 1,054,756 bytes |
| Maximum group | 32,271,513 bytes |
| Compact boundary data | **1,609 bytes** |

Every ordinary boundary is the start of a physical post-R1 page. The groups
tile `[0, 587138826)` without overlap or gaps. A final 32,271,513-byte physical
page cannot be reduced to approximately 1 MiB without splitting that page, so
it remains one oversized recipient.

Files:

```text
enwik9.grouped_1mib_recipients.csv
enwik9.grouped_1mib_boundaries.f4gb
```

The CSV contains discovery attributes, dominant stream class and aggregated
feature masks. It is not production side data. `F4GB` is the compact boundary
representation:

```text
magic "F4GB"
version 1
varint target_bytes
varint stream_bytes
varint group_count
varint group_length[group_count]
```

Starts are reconstructed by prefix-summing group lengths. The generator reads
the file back and verifies all 531 boundaries and final stream length.

Regenerate:

```bat
python tools\group_postr1_pages.py ^
  results\donor_multiday_seed\enwik9.page_recipients.csv ^
  results\donor_multiday_seed\enwik9.grouped_1mib_recipients.csv ^
  results\donor_multiday_seed\enwik9.grouped_1mib_boundaries.f4gb ^
  --stream-size 587138826 --target-bytes 1048576
```

Run resumable grouped discovery from Windows CMD after the existing physical-
page donor sweep has stopped:

```bat
run_claude_grouped_donor_multiday.cmd 25 7
```

The first invocation ranks donor candidates for all 531 groups. It uses a
560 MiB stop ceiling so the final 587,138,826-byte stream position and the
oversized final group are included. Later invocations reuse those rankings and
durable exact-trial ledgers.

For a final archive, prefer storing only selected winning span boundaries in
F4CP v7. The complete 1,609-byte F4GB map is an upper bound and need not be
included when only a small number of groups wins.
