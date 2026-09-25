#!/bin/bash
# Roda API local (precisa fastapi+uvicorn). Uso: ./scripts/run_local.sh
set -e
cd "$(dirname "$0")/.."
PY=python3
UV_TARGET=--system
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
  UV_TARGET="--python .venv/bin/python"
fi
if ! $PY -c "import fastapi" 2>/dev/null; then
  echo "[run] instalando fastapi+uvicorn via uv..."
  ~/.local/bin/uv pip install $UV_TARGET 'fastapi' 'uvicorn[standard]' websockets pyyaml 2>&1 | tail -3
fi
export ORCHESTRATOR_MODE=${ORCHESTRATOR_MODE:-auto}
echo "[run] API em http://localhost:8000 | UI em http://localhost:8000/ui | modo=$ORCHESTRATOR_MODE"
$PY -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir apps/api
