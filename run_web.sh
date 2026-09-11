#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.pythonlibs/bin/python3.11" ]]; then
    PYTHON_BIN="$ROOT_DIR/.pythonlibs/bin/python3.11"
fi
if [[ -z "$PYTHON_BIN" ]] && command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.11 tidak ditemukan. Jalankan bash install.sh terlebih dahulu." >&2
    exit 1
fi

PORT="${PORT:-5000}"
if [[ -z "${DZECK_MODEL_PATH:-}" && -f "$ROOT_DIR/dzeck-model/config.json" ]]; then
    export DZECK_MODEL_PATH="$ROOT_DIR/dzeck-model"
elif [[ -z "${DZECK_MODEL_PATH:-}" && -f "$ROOT_DIR/qwen2.5-0.5b-instruct/config.json" ]]; then
    export DZECK_MODEL_PATH="$ROOT_DIR/qwen2.5-0.5b-instruct"
fi
export DZECK_DEVICE="${DZECK_DEVICE:-cpu}"
exec "$PYTHON_BIN" -m streamlit run scripts/web_demo.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false