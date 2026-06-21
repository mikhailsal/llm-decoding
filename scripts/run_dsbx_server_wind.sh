#!/usr/bin/env bash
# Launch the dsbx HTTP server on `dsbx-host`, hosting one heavy in-process backend.
#
# Usage (on dsbx-host):
#   bash scripts/run_dsbx_server_wind.sh [BACKEND] [PORT] [HOST]
#
# Defaults: BACKEND=llamacpp-py PORT=8000 HOST=0.0.0.0
#
# The server holds the loaded model in memory for its lifetime; the
# client's CLI connects via a [remote.NAME] entry in config.toml whose
# base_url points here (e.g. http://192.0.2.42:8000).
#
# Run one process per backend if you want both ``hf`` and ``llamacpp-py``
# available simultaneously -- pick different ports:
#   bash scripts/run_dsbx_server_wind.sh llamacpp-py 8000
#   bash scripts/run_dsbx_server_wind.sh hf          8001
#
# Caveat: the server has no auth. HOST=0.0.0.0 makes it reachable from
# anywhere on the LAN; keep this box behind your local network only.
set -euo pipefail

BACKEND="${1:-${BACKEND:-llamacpp-py}}"
PORT="${2:-${PORT:-8000}}"
HOST="${3:-${HOST:-0.0.0.0}}"

# Pick up the ~/.cache/dsbx caches the rest of the project relies on.
if [ -f scripts/env_wind.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_wind.sh
fi

if [ ! -d .venv ]; then
  echo "ERROR: .venv not found. Run scripts/setup_wind.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "==> installing [server] extra (fastapi + uvicorn)"
  pip install -e ".[server]"
fi

echo "==> dsbx serve --backend ${BACKEND} --host ${HOST} --port ${PORT}"
exec dsbx serve --backend "${BACKEND}" --host "${HOST}" --port "${PORT}"
