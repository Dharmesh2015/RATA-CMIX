#!/bin/sh
set -eu

# This is the only root/network phase allowed by HutterPrizeJudgingAssistant.
export DEBIAN_FRONTEND=noninteractive
apt-get update -o Acquire::Retries=3
apt-get install --yes --no-install-recommends \
  -o Acquire::Retries=3 \
  binutils coreutils curl gnupg libc6-dev make \
  time util-linux xz-utils

# The judging base image is a plain Ubuntu 20.04 (focal) Google Cloud host --
# the image James Bowery's own alpha-testing instructions specify
# (ubuntu-2004-focal-v20240731, ubuntu-os-cloud). focal's stock archive only
# carries clang-10, so the toolchain this project actually builds and tests
# with (clang-17, matching cmix-lex's own submission) is fetched from LLVM's
# official apt.llvm.org installer -- the same repeatable, noninteractive
# mechanism this script already uses for UPX, just via LLVM's own script
# instead of a raw curl+checksum.
if ! command -v clang++-17 >/dev/null 2>&1; then
  llvm_sh="$(mktemp)"
  trap 'rm -f "$llvm_sh"' EXIT HUP INT TERM
  curl --fail --location --retry 3 --output "$llvm_sh" https://apt.llvm.org/llvm.sh
  chmod +x "$llvm_sh"
  "$llvm_sh" 17
  apt-get install --yes --no-install-recommends -o Acquire::Retries=3 \
    lld-17
fi
rm -rf /var/lib/apt/lists/*

upx=/opt/upx/upx-5.1.1-amd64_linux/upx
if [ ! -x "$upx" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT HUP INT TERM
  archive="$tmp/UPX-5.1.1-amd64_linux.tar.xz"
  curl --fail --location --retry 3 --output "$archive" \
    https://github.com/upx/upx/releases/download/v5.1.1/upx-5.1.1-amd64_linux.tar.xz
  echo "1ff660454227861e00772f743f66b900072116b9dc24f6ee28b97cce88a7828a  $archive" \
    | sha256sum --check -
  mkdir -p /opt/upx
  tar -xJf "$archive" -C /opt/upx
  test -x "$upx"
fi
