# Google Cloud And Alpha Judging

## VM And Disk

Use:

- Ubuntu 20.04 LTS (focal), x86-64 -- the actual judging image
  (`ubuntu-2004-focal-v20240731`, `ubuntu-os-cloud`).
- No GPU.
- At least 16 GiB RAM. More host RAM is fine for development, but the judged
  process must remain below 10 GiB RSS.
- At least 100 GB local disk for the VM. Put the working directory on ext4 or
  XFS, not a network filesystem, Cloud Storage FUSE or an SMB mount.
- No swap when reproducing the judging limit.

The codec is single-threaded during compression and decompression. A larger
CPU count only speeds compilation.

## Install And Build

    cd /path/to/fx4-cmix
    sudo ./install.sh
    ./build_and_construct_comp.sh

The installer is suitable for plain Ubuntu and for the judging assistant's
dependency-image phase. It fetches clang++-17 from the official
`apt.llvm.org` repository (focal's own archive only carries clang-10) and
verifies the same UPX 5.1.1 archive checksum used by the judging repository.

The build creates cmix, which is S1. Record its size and hash:

    stat -c '%n %s bytes' cmix
    sha256sum cmix

## Full Compression

Start the run from a persistent SSH multiplexer or service:

    ./tools/run_google_cloud_hutter.sh /data/enwik9 /data/ratarun 0

The script copies both S1 and enwik9 into a fresh local run directory. It pins
execution to one core and writes /data/ratarun/compression.log.

Do not set old FX4 discovery, model-selection or transformer environment
variables. The release configuration is compiled in.

## Monitoring

    ./tools/monitor_hutter_run.sh /data/ratarun 60

The monitor reads the entropy input file descriptor and reports:

- process CPU, elapsed time and RSS
- post-R1 entropy position and percentage
- current streamed arithmetic payload size
- a linear archive9 size projection including known overlay assets
- ppm.temp size and host memory

The size projection is not a score prediction. Compression loss varies across
the transformed stream, especially near structural boundaries.

## Clean Decompression Test

    mkdir /data/ratadecode
    cp /data/ratarun/archive9 /data/ratadecode/
    cd /data/ratadecode
    env -i PATH=/usr/bin:/bin HOME="$HOME" \
      /usr/bin/time -v taskset -c 0 ./archive9
    cmp /data/enwik9 enwik9_uncompressed
    sha256sum /data/enwik9 enwik9_uncompressed

The decode directory must not contain the source enwik9, dictionary,
article-order file or transformer model. This verifies that S2 is genuinely
self-contained.

## Create The Entrant Directory

    cd /path/to/fx4-cmix
    ./tools/create_judging_entry.sh \
      /data/ratarun/archive9 /data/rata-entry RATA

Inspect it:

    find /data/rata-entry/Entries/RATA -maxdepth 1 -type f -printf '%f %s\n'
    tar -tzf /data/rata-entry/Entries/RATA/rata-cmix-source.tar.gz | head

The manifest declares:

    ENTRY_FORMAT=self-extracting
    EXECUTION_PLATFORM=linux-x86_64
    COMPRESSOR=cmix
    COMPRESSOR_FORMAT=upx-overlay
    ARCHIVE=archive9
    ARCHIVE_FORMAT=upx-overlay
    DECOMPRESSED_OUTPUT=enwik9_uncompressed

comp9.args is exactly:

    -e
    enwik9
    archive9

## Run HutterPrizeJudgingAssistant

Prepare an ext4/XFS work volume with at least the configured 100 GB allowance,
then:

    git clone https://github.com/jabowery/HutterPrizeJudgingAssistant.git
    cd HutterPrizeJudgingAssistant
    cp -a /data/rata-entry/Entries/RATA Entries/
    cp /data/enwik9 ./enwik9
    ./judging_assistance.sh \
      --serial \
      --runtime-exec-policy process-tree \
      --work-root /mnt/large-disk/HutterPrizeJudging \
      Entries/RATA ./enwik9

Use --serial for the final timing run. The default can overlap archive
qualification and rebuilt compression.

The process-tree relaxation is necessary because the counted S1/S2 images
extract and execute their own packed core for helper-stream decoding. No
independently supplied executable or model is used at runtime.

The assistant enforces one-core execution, 16 GiB container memory with no
swap, at most 10 GiB measured RSS, at most 100 GB temporary disk, offline
runtime, exact output verification and calibrated timing. Human review remains
authoritative.
