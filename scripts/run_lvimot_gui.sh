#!/usr/bin/env sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_CMD="${LVIMOT_PYTHON:-python3}"
if [ -x .venv_live38/bin/python ]; then PYTHON_CMD=.venv_live38/bin/python; fi
exec "$PYTHON_CMD" -m lvimot.output.live_gui "$@"
