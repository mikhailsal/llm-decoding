#!/usr/bin/env bash
# One-time environment setup on `dsbx-host` (the P40 GPU box).
#
# - Puts bulk caches on ~/.cache/dsbx (large local SSD), NOT on the tight C:/ext4 disk.
# - Creates a venv on ext4 (fast) at ./.venv.
# - Installs core deps, then torch (CUDA 12.4) and the local-model extra.
#
# Safe to re-run. Run from the synced repo root on dsbx-host:
#   bash scripts/setup_wind.sh
set -euo pipefail

CACHE_ROOT="${CACHE_ROOT:-~/.cache/dsbx}"
HF_DIR="${CACHE_ROOT}/huggingface"
PIP_DIR="${CACHE_ROOT}/pip"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
GGUF_REPO="${GGUF_REPO:-Jessylg27/Qwen3.5-9B-Base-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3.5-9B-Base-Q4_K_M.gguf}"
DOWNLOAD_GGUF="${DOWNLOAD_GGUF:-1}"

echo "==> Cache root: ${CACHE_ROOT}"
if [ ! -d "$(dirname "${CACHE_ROOT}")" ]; then
  echo "ERROR: $(dirname "${CACHE_ROOT}") not mounted. Is ~/.cache/dsbx available?" >&2
  exit 1
fi
mkdir -p "${HF_DIR}" "${PIP_DIR}"

export HF_HOME="${HF_DIR}"
export HF_HUB_CACHE="${HF_DIR}/hub"
export PIP_CACHE_DIR="${PIP_DIR}"

# Persist these for interactive shells / future runs.
cat > scripts/env_wind.sh <<EOF
# Source me on dsbx-host to use the ~/.cache/dsbx caches: source scripts/env_wind.sh
export HF_HOME="${HF_DIR}"
export HF_HUB_CACHE="${HF_DIR}/hub"
export PIP_CACHE_DIR="${PIP_DIR}"
EOF
echo "==> Wrote scripts/env_wind.sh (HF_HOME=${HF_DIR})"

echo "==> Creating venv on ext4 (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> Installing core package"
pip install -e .

echo "==> Installing torch (CUDA 12.4) from ${TORCH_INDEX}"
pip install torch --index-url "${TORCH_INDEX}"

echo "==> Installing local-model extra (transformers/accelerate/bitsandbytes)"
pip install -e ".[local]"

if [ "${DOWNLOAD_GGUF}" = "1" ]; then
  echo "==> Downloading ${GGUF_REPO}/${GGUF_FILE} into HF cache"
  python - <<PY
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="${GGUF_REPO}",
    filename="${GGUF_FILE}",
    cache_dir="${HF_DIR}",
)
print(path)
PY
else
  echo "==> Skipping GGUF download (DOWNLOAD_GGUF=${DOWNLOAD_GGUF})"
fi

echo
echo "==> Verifying CUDA on the P40"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))
PY

echo
echo "Done. Next:  source .venv/bin/activate && source scripts/env_wind.sh && dsbx doctor"
