#!/bin/sh
set -eu

export DEBIAN_FRONTEND=noninteractive
apt-get update -o Acquire::Retries=3
apt-get install --yes --no-install-recommends \
  -o Acquire::Retries=3 \
  gcc libc6-dev libzstd-dev
