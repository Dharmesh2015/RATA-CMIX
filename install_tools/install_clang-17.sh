#!/usr/bin/env bash
set -euo pipefail

REQUIRED_TOOLS=(
    clang-17
    clang++-17
    llvm-profdata-17
    llvm-strip-17
    ld.lld-17
    make
    objcopy
)

missing=0

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing: $tool"
        missing=1
    fi
done

if [[ "$missing" -eq 1 ]]; then
    sudo apt-get update

    sudo apt-get install -y \
        build-essential \
        binutils \
        clang-17 \
        llvm-17 \
        llvm-17-tools \
        lld-17 \
        make \
        file \
        time
fi

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Installation failed: $tool is unavailable." >&2
        exit 1
    fi
done

echo "Installed build tools:"
clang++-17 --version | head -n1
llvm-profdata-17 --version | head -n2
llvm-strip-17 --version | head -n1
ld.lld-17 --version | head -n1
make --version | head -n1