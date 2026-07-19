# fx4-cmix

FX4 is a cmix-lex/fxcm-v26 Hutter compressor for enwik9.

## Pipeline

~~~text
embedded English dictionary + article order
-> split intro/main/coda
-> article reorder
-> PHDA9/WIT transform
-> WRT dictionary preprocessing
-> payload_lex / R1 tail reorder
-> PPMD order 25 (14,000 MB file-backed arena)
-> online LSTM-170 + FXCM-v26 + full-strength internal LSTM bridge
-> match/context models
-> arithmetic-coded payload
-> self-extracting archive9
~~~

The PPM arena uses a stable file-backed mapping with MADV_RANDOM,
MADV_DONTNEED, O_NOATIME, and ftruncate. Compression is single-core.

## Accepted 1 MiB Gate

Input is enwik9 offset 441,450,496, length 1,048,576.

~~~text
archive:             99,345 bytes
bits/byte:           0.757942200
compression:         821.91 s internal, 13:46.12 wall (cold/page-fault-heavy run)
decompression:       442.70 s internal, 7:24.06 wall
peak RSS:            about 8.30 GB
roundtrip:           exact SHA-256 match
~~~

The bridge adds no serialized state, mode bit, predictor table, or replay.
Compact bootstrap tables were rejected because measured calibration savings
did not cover even the smallest proposed 16 KiB state.


## Final Artifacts

~~~text
cmix                 Hutter S1 with embedded dictionary/order
run/cmix_orig        packed accepted core
run/cmix             same Hutter S1
~~~

The accepted package is 440,172 bytes (139,524-byte packed core). Current
sizes are recorded in tools/last_package.txt. Embedded
dictionary and article-order streams are always decoded and byte-compared
before S1 is accepted.

## Validate 1 MiB

From Windows PowerShell or CMD:

~~~bat
wsl --shutdown
wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/tools/validate_1mb_ext4.sh /mnt/d/mywork/myideas/latestcompressor/fx4-cmix /mnt/d/mywork/myideas/latestcompressor/enwik9
~~~

This creates a fresh /root/fx4run, builds on native WSL ext4, packs with
UPX 5.1.1, checks the 99,345-byte gate, and verifies SHA-256 roundtrip.

## Full enwik9 Run

Use the already verified S1:

~~~bat
powercfg /setactive SCHEME_MIN
wsl --shutdown
wsl -d Ubuntu -- bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/tools/run_hutter_ext4.sh /mnt/d/mywork/myideas/latestcompressor/enwik9 /mnt/d/mywork/myideas/latestcompressor/archive9
~~~

To rebuild S1 from source first:

~~~bat
wsl -d Ubuntu -- env REBUILD_S1=1 bash /mnt/d/mywork/myideas/latestcompressor/fx4-cmix/tools/run_hutter_ext4.sh /mnt/d/mywork/myideas/latestcompressor/enwik9 /mnt/d/mywork/myideas/latestcompressor/archive9
~~~

The -e path is used internally. The full 1 GB result still must be measured;
a strong 1 MiB sample does not prove the final archive9 size.