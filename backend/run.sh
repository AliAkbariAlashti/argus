#!/usr/bin/env bash
# Starts the server in the background and returns immediately.
#
#   cd ~/dok/qwenvl
#   source ~/dok/venv/bin/activate
#   bash backend/run.sh
#
# Check it's up:   bash backend/status.sh
# Stop it:         bash backend/stop.sh
# Logs:            tail -f backend/server.log
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="backend/server.pid"
LOG_FILE="backend/server.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running (PID $(cat "$PID_FILE")). Run backend/stop.sh first if you want to restart."
  exit 1
fi

nohup uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started (PID $!). Logs: $LOG_FILE"
