#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <remote-host> <remote-dir> <local-mountpoint>" >&2
  exit 1
fi

if ! command -v sshfs >/dev/null 2>&1; then
  echo "sshfs is required but not installed." >&2
  exit 1
fi

REMOTE_HOST="$1"
REMOTE_DIR="$2"
LOCAL_MOUNTPOINT="$3"

MOUNT_MODE="${SSHFS_MODE:-ro}"
BASE_OPTIONS="${SSHFS_OPTIONS:-reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks,default_permissions}"

mkdir -p "${LOCAL_MOUNTPOINT}"

if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "${LOCAL_MOUNTPOINT}"; then
  echo "Mountpoint already active: ${LOCAL_MOUNTPOINT}"
  exit 0
fi

OPTIONS="${BASE_OPTIONS}"
if [[ "${MOUNT_MODE}" == "ro" ]]; then
  OPTIONS="${OPTIONS},ro"
fi

echo "Mounting ${REMOTE_HOST}:${REMOTE_DIR} -> ${LOCAL_MOUNTPOINT}"
sshfs -o "${OPTIONS}" "${REMOTE_HOST}:${REMOTE_DIR}" "${LOCAL_MOUNTPOINT}"
