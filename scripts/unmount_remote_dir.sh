#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <local-mountpoint>" >&2
  exit 1
fi

LOCAL_MOUNTPOINT="$1"

if command -v fusermount3 >/dev/null 2>&1; then
  fusermount3 -u "${LOCAL_MOUNTPOINT}"
elif command -v fusermount >/dev/null 2>&1; then
  fusermount -u "${LOCAL_MOUNTPOINT}"
else
  umount "${LOCAL_MOUNTPOINT}"
fi
