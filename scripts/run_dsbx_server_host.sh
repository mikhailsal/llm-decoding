#!/usr/bin/env bash
# Launch the dsbx HTTP server on `dsbx-host`, hosting one heavy in-process backend.
#
# Usage (on dsbx-host):
#   bash scripts/run_dsbx_server_host.sh [BACKEND] [PORT] [HOST]
#
# Defaults: BACKEND=llamacpp-py PORT=8000 HOST=0.0.0.0
#
# The server holds the loaded model in memory for its lifetime; the
# client's CLI connects via a [remote.NAME] entry in config.toml whose
# base_url points here (e.g. http://192.0.2.42:8000).
#
# Run one process per backend if you want both ``hf`` and ``llamacpp-py``
# available simultaneously -- pick different ports:
#   bash scripts/run_dsbx_server_host.sh llamacpp-py 8000
#   bash scripts/run_dsbx_server_host.sh hf          8001
#
# Caveat: the server has no auth. HOST=0.0.0.0 makes it reachable from
# anywhere on the LAN; keep this box behind your local network only.
set -euo pipefail

BACKEND="${1:-${BACKEND:-llamacpp-py}}"
PORT="${2:-${PORT:-8000}}"
HOST="${3:-${HOST:-0.0.0.0}}"

# Pick up the bulk caches the rest of the project relies on.
if [ -f scripts/env_host.sh ]; then
  # shellcheck disable=SC1091
  source scripts/env_host.sh
fi

if [ ! -d .venv ]; then
  echo "ERROR: .venv not found. Run scripts/setup_host.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Heal both (a) first-run on a venv that hasn't seen the [server] extra and
# (b) a venv that was set up before the `decoding_sandbox` -> `dsbx` rename
# and whose entry-point script still imports the old module. We check the
# installed binary directly (not `import dsbx`), because the synced source
# directory is on sys.path via cwd and would mask a stale install.
if ! dsbx --version >/dev/null 2>&1 || ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "==> (re)installing dsbx[server]"
  pip install -e ".[server]"
fi

echo "==> dsbx serve --backend ${BACKEND} --host ${HOST} --port ${PORT}"
exec dsbx serve --backend "${BACKEND}" --host "${HOST}" --port "${PORT}"
