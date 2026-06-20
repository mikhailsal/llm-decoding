#!/usr/bin/env bash
# Sync the repo from the client to `dsbx-host` (run-on-dsbx-host workflow).
#
# Code is edited here and executed on dsbx-host (the box with the P40 GPU). This
# rsync pushes source only -- never the venv, caches, models, or secrets.
#
# Usage: scripts/sync_to_wind.sh [DSBX_DEST]
set -euo pipefail

DSBX_HOST="${DSBX_HOST:-dsbx-host}"
DEST="${1:-${DSBX_DEST:-llm-decoding}}"   # relative to dsbx-host's home (ext4)
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

echo "Syncing ${SRC} -> ${DSBX_HOST}:${DEST}"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'models/' \
  --exclude 'cache/' \
  --exclude 'sessions/' \
  --exclude 'out/' \
  --exclude 'config.toml' \
  --exclude '.env' \
  --exclude 'scripts/env_wind.sh' \
  "${SRC}" "${DSBX_HOST}:${DEST}/"

echo "Done. On dsbx-host: cd ${DEST} && source .venv/bin/activate && dsbx doctor"
