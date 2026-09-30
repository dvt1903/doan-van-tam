#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prepare.py
(cd web && npm install && npm run build)
exec python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
