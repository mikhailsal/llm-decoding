#!/usr/bin/env bash
# Build llama.cpp with CUDA on `dsbx-host` for the Pascal-class GPU (compute capability 6.1).
#
# Notes:
# - System nvcc is CUDA 12.0; gcc-13 is the default but CUDA 12.0 only supports
#   up to gcc-12, so we force the CUDA host compiler to g++-12.
# - We target only one CUDA architecture (sm_61 here) to keep the build small
#   and fast; adjust -DCMAKE_CUDA_ARCHITECTURES below for your GPU.
# - LLAMA_CURL is off: we always pass a local GGUF path, no remote fetch needed.
#
# Usage on dsbx-host:  bash scripts/build_llamacpp_host.sh
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "${LLAMA_DIR}/.git" ]; then
  echo "==> Cloning llama.cpp into ${LLAMA_DIR}"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
else
  echo "==> Updating existing ${LLAMA_DIR}"
  git -C "${LLAMA_DIR}" pull --ff-only || true
fi

cd "${LLAMA_DIR}"
echo "==> llama.cpp at commit $(git rev-parse --short HEAD)"

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  `# ^ adjust for your GPU (e.g. 61 = Pascal, 75 = Turing, 86 = Ampere)` \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DLLAMA_CURL=OFF

echo "==> Building (server + cli) with ${JOBS} jobs"
cmake --build build --config Release -j "${JOBS}" --target llama-server llama-cli

echo
echo "==> Build artifacts:"
ls -lh build/bin/llama-server build/bin/llama-cli
echo "Done."
