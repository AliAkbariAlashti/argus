#!/usr/bin/env bash
# Stops the server started by run.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
PID_FILE="backend/server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found — nothing to stop."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Sent stop signal to PID $PID."
else
  echo "PID $PID is not running (stale PID file)."
fi

rm -f "$PID_FILE"
