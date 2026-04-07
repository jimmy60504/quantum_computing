#!/usr/bin/env bash
# Start the Problem 2 viewer on gx10 and open an SSH tunnel to it.
# Usage: ./viewer.sh [stop]

set -euo pipefail

REMOTE="gx10"
SESSION="prob2_viewer"
REMOTE_DIR="/home/jimmy/quantum_computing/HW1/problem2/hf_space"
PORT=8782
URL="http://localhost:${PORT}"

stop() {
  echo "Closing tunnel on port ${PORT}..."
  pkill -f "ssh -f -N.*${PORT}:localhost:${PORT}" 2>/dev/null || true
  echo "Stopping tmux session ${SESSION} on ${REMOTE}..."
  ssh "${REMOTE}" "tmux kill-session -t ${SESSION} 2>/dev/null || true"
  echo "Done."
}

if [[ "${1:-}" == "stop" ]]; then
  stop
  exit 0
fi

# 1. Start HTTP server on gx10 if not already running
if ssh "${REMOTE}" "tmux has-session -t ${SESSION} 2>/dev/null"; then
  echo "Server already running (tmux: ${SESSION})"
else
  echo "Starting HTTP server on ${REMOTE}:${PORT}..."
  ssh "${REMOTE}" "tmux new-session -d -s ${SESSION} \
    'cd ${REMOTE_DIR} && python3 -m http.server ${PORT}'"
fi

# 2. Open SSH tunnel if not already open
if lsof -i "TCP:${PORT}" -sTCP:LISTEN &>/dev/null; then
  echo "Tunnel already open on localhost:${PORT}"
else
  echo "Opening SSH tunnel localhost:${PORT} -> ${REMOTE}:${PORT}..."
  ssh -f -N -L "${PORT}:localhost:${PORT}" "${REMOTE}"
fi

# 3. Open in browser
echo "Opening ${URL}"
open "${URL}" 2>/dev/null || xdg-open "${URL}" 2>/dev/null || echo "Browse to ${URL}"
