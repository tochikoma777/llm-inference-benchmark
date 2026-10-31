#!/usr/bin/env bash
# List stored runs, newest first.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
fi

exec "$PY" -m llmbench runs "$@"
