#!/usr/bin/env bash
# Reproduce el entorno en una sesión nueva: venv (Python 3.12) + dependencias + modelos HF.
# Uso: bash scripts/setup.sh [--skip-models]
set -euo pipefail
cd "$(dirname "$0")/.."

# Python 3.12: requirements.txt fija pandas==2.2.2, que no tiene wheels para 3.13.
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || { echo "Necesito $PY (o define PYTHON=/ruta/a/python3.10-3.12)"; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt

if [ "${1:-}" != "--skip-models" ]; then
  python scripts/download_models.py   # requiere acceso a huggingface.co
fi
echo "Listo. Activa el entorno con: source .venv/bin/activate"
