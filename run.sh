#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${JEV_PYTHON:-python3}"
HOST="${JEV_HOST:-127.0.0.1}"
PORT="${JEV_PORT:-8000}"
export JEV_MODEL_PATH="${JEV_MODEL_PATH:-Qwen/Qwen3.5-2B}"

exec "$PYTHON_BIN" -m uvicorn app:app \
  --app-dir "$APP_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
