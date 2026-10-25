#!/bin/bash
# One-time setup on macOS (Apple Silicon). Creates .venv next to this file.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
echo "using $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Ready. Try:  source .venv/bin/activate && python live2d.py your_picture.png"
