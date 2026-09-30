#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "[INFO] Created .env from .env.example"
fi

python -m uvicorn src.interfaces.api:app --reload --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT
sleep 2
python -m streamlit run src.interfaces.ui
