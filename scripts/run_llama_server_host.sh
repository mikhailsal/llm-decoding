#!/usr/bin/env bash
# Launch llama-server on `dsbx-host` with the Qwen3.5-9B-Base GGUF.
#
# The GPU has only 6 GB VRAM and the Q4_K_M model is ~5.3 GB, so we offload a
# partial number of layers (-ngl) and keep the context modest. Tune NGL/CTX to
# fit; check `nvidia-smi` while it runs.
#
# Usage on dsbx-host:  bash scripts/run_llama_server_host.sh [NGL] [CTX]
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
NGL="${1:-${NGL:-20}}"
CTX="${2:-${CTX:-4096}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# Resolve the GGUF from the HF cache (downloaded by the setup step).
MODEL="${MODEL:-$(find ~/.cache/dsbx/huggingface -name 'Qwen3.5-9B-Base-Q4_K_M.gguf' 2>/dev/null | head -1)}"
if [ -z "${MODEL}" ] || [ ! -f "${MODEL}" ]; then
  echo "ERROR: GGUF not found. Download it first (see setup)." >&2
  exit 1
fi

echo "==> model : ${MODEL}"
echo "==> ngl=${NGL} ctx=${CTX} -> http://${HOST}:${PORT}"
exec "${LLAMA_DIR}/build/bin/llama-server" \
  -m "${MODEL}" \
  -ngl "${NGL}" \
  -c "${CTX}" \
  --host "${HOST}" \
  --port "${PORT}"
