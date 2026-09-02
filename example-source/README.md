# Hutter Prize judging-system example codec

This is a deliberately simple procedural fixture. It is not derived from a
Hutter Prize submission and is not intended to be competitive.

`comp9` is one statically linked, baseline x86-64 executable. In compression
mode it copies itself, appends a single-threaded Zstandard level-1 frame, and
adds a fixed trailer. Running the resulting executable without arguments
decompresses that frame to `data9`. It launches no helper executable.

The generic build deliberately uses `-march=x86-64 -mtune=generic`, avoiding
CPU-specific code generation so that the same artifact runs on both Intel and
AMD x86-64 machines supported by the Linux worker.

The source is provided only to exercise every ordinary stage of the judging
system quickly and reproducibly.
