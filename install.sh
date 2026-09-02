#!/bin/sh
set -eu

# This is the only root/network phase allowed by HutterPrizeJudgingAssistant.
export DEBIAN_FRONTEND=noninteractive
apt-get update -o Acquire::Retries=3
apt-get install --yes --no-install-recommends \
  -o Acquire::Retries=3 \
  binutils clang coreutils curl libc6-dev lld make \
  time util-linux xz-utils
rm -rf /var/lib/apt/lists/*

# The judging base image already provides this exact pinned binary. Download
# it only for a plain Ubuntu 20.04 (focal) Google Cloud host -- the image
# James Bowery's own alpha-testing instructions specify
# (ubuntu-2004-focal-v20240731, ubuntu-os-cloud). libstdc++-12-dev does not
# exist in focal's archive (GCC 12 postdates the release); this target
# builds -std=c++17 only, which focal's default libc6-dev/libstdc++ already
# supports, so no newer libstdc++ package is needed.
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
