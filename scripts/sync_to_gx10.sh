#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${GX10_HOST:-gx10}"
REMOTE_REPO_DIR="${GX10_REPO_DIR:-/home/jimmy/quantum_computing}"
SYNC_EXCLUDE_FILE="${GX10_SYNC_EXCLUDE_FILE:-${REPO_ROOT}/.gx10-sync-excludes}"

rsync -av --delete \
  --exclude-from "${REPO_ROOT}/.gitignore" \
  --exclude-from "${SYNC_EXCLUDE_FILE}" \
  "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_REPO_DIR}/"
