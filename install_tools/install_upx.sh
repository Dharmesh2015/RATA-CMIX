#!/usr/bin/env bash
set -euo pipefail

UPX_VERSION="${UPX_VERSION:-5.1.1}"
PACKAGE="upx-${UPX_VERSION}-amd64_linux"
ARCHIVE="${PACKAGE}.tar.xz"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
UPX_DIR="${TOOLS_DIR}/${PACKAGE}"
UPX_LINK="${TOOLS_DIR}/upx"
UPX_UCL_LINK="${TOOLS_DIR}/upx-ucl"

URL="https://github.com/upx/upx/releases/download/v${UPX_VERSION}/${ARCHIVE}"

TMP_DIR="$(mktemp -d)"
TMP_ARCHIVE="${TMP_DIR}/${ARCHIVE}"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

sudo apt-get update
sudo apt-get install -y \
    curl \
    ca-certificates \
    tar \
    xz-utils

mkdir -p "$TOOLS_DIR"

echo "Downloading UPX ${UPX_VERSION}..."
curl \
    --fail \
    --location \
    --retry 3 \
    --output "$TMP_ARCHIVE" \
    "$URL"

tar -xJf "$TMP_ARCHIVE" -C "$TMP_DIR"

if [[ ! -x "${TMP_DIR}/${PACKAGE}/upx" ]]; then
    echo "Downloaded UPX executable is missing." >&2
    exit 1
fi

rm -rf "$UPX_DIR"
mv "${TMP_DIR}/${PACKAGE}" "$UPX_DIR"

# Repair the earlier incorrect tools/upx directory.
if [[ -d "$UPX_LINK" && ! -L "$UPX_LINK" ]]; then
    BACKUP="${UPX_LINK}.bad.$(date +%Y%m%d%H%M%S)"
    echo "Moving incorrect tools/upx directory to: $BACKUP"
    mv "$UPX_LINK" "$BACKUP"
else
    rm -f "$UPX_LINK"
fi

rm -f "$UPX_UCL_LINK"

ln -s "${UPX_DIR}/upx" "$UPX_LINK"
ln -s "${UPX_DIR}/upx" "$UPX_UCL_LINK"

VERSION_TEXT="$("$UPX_LINK" --version | head -n1)"

if [[ "$VERSION_TEXT" != "upx ${UPX_VERSION}" ]]; then
    echo "Unexpected UPX version: ${VERSION_TEXT}" >&2
    exit 1
fi

echo "Installed: $UPX_LINK"
echo "Resolved:  $(readlink -f "$UPX_LINK")"

"$UPX_LINK" --version