#!/usr/bin/env bash
# Run from the repo root on the VM, with the venv already activated:
#   cd ~/dok/qwenvl
#   source ~/dok/venv/bin/activate
#   bash backend/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/backend:${PYTHONPATH:-}"

uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"
