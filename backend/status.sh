#!/usr/bin/env bash
# Reports whether the server started by run.sh is running.
set -uo pipefail

cd "$(dirname "$0")/.."
PID_FILE="backend/server.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  PID="$(cat "$PID_FILE")"
  echo "Running (PID $PID)"
  ps -p "$PID" -o pid,etime,cmd --no-headers 2>/dev/null
  echo
  echo "--- health check ---"
  curl -s "http://127.0.0.1:${PORT:-8000}/api/status" || echo "(no response on port ${PORT:-8000})"
  echo
else
  echo "Not running (no active PID file at $PID_FILE)"
  exit 1
fi
