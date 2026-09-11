#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/dzeck-large-id}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.pythonlibs/bin/python3.11" ]]; then
    PYTHON_BIN="$ROOT_DIR/.pythonlibs/bin/python3.11"
fi

if [[ -z "$PYTHON_BIN" ]] && command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
fi

if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Python 3.11 tidak ditemukan."
    echo "Pasang Python 3.11 lalu jalankan ulang: bash install.sh"
    exit 1
fi

echo "==> Menggunakan Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "==> Memasang PyTorch CPU..."
"$PYTHON_BIN" -m pip install \
    --break-system-packages \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0"

echo "==> Memasang dependensi proyek..."
"$PYTHON_BIN" -m pip install --break-system-packages -r requirements.txt

if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "==> Mengunduh fondasi model besar $MODEL_NAME..."
    "$PYTHON_BIN" -m modelscope.cli.cli download \
        --model "$MODEL_NAME" \
        --local_dir "$MODEL_DIR"
else
    echo "==> Checkpoint model sudah ada, lewati unduhan."
fi

echo "==> Memvalidasi instalasi..."
"$PYTHON_BIN" - <<'PY'
import torch
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA tersedia: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
PY

"$PYTHON_BIN" -m compileall -q \
    dataset model scripts trainer eval_llm.py

echo
echo "Instalasi selesai."
echo
echo "Identitas model:"
echo "  Nama aplikasi/API: Dzeck Large ID"
echo "  Fondasi default:  $MODEL_NAME"
echo "  Catatan: folder ini menjadi model Dzeck milik Anda setelah hasil training diekspor."
echo
echo "Untuk menjalankan AI secara interaktif:"
echo "  $PYTHON_BIN eval_llm.py --device cpu"
echo
echo "Untuk menjalankan tes otomatis:"
echo "  printf '0\\n' | $PYTHON_BIN eval_llm.py --device cpu --max_new_tokens 64"

if [[ "${1:-}" == "--test" ]]; then
    echo
    echo "==> Membuka AI dalam mode interaktif..."
    exec "$PYTHON_BIN" eval_llm.py --device cpu
fi