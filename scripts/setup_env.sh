#!/usr/bin/env bash
# One environment for the generator (FragFM) and the oracle (ms-pred).
#
# Rationale: FragFM pins numpy==1.26 / rdkit==2023.9 / torch==2.1, ms-pred wants
# numpy>=2 / rdkit==2025.3 / torch==2.6 + dgl.  The pins are incompatible on
# paper, but FragFM's source uses no numpy-1-only or rdkit-2023-only API, so we
# resolve on ms-pred's (newer) stack and install FragFM with --no-deps.  This
# keeps the GRPO reward in-process; a two-env split would force IPC on ~160k
# oracle calls per epoch.  scripts/preflight.py verifies the gamble.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${THIRD_PARTY:-$(dirname "$REPO_ROOT")}"
CUDA_TAG="${CUDA_TAG:-cu124}"
TORCH_VER=2.6.0

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

clone_if_missing() {  # url dir
  [ -d "$2" ] || git clone --depth 1 "$1" "$2"
}
clone_if_missing https://github.com/lee-jwon/FragFM.git      "$THIRD_PARTY/FragFM"
clone_if_missing https://github.com/coleygroup/ms-pred.git   "$THIRD_PARTY/ms-pred"

cd "$REPO_ROOT"
uv venv -p 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install "torch==${TORCH_VER}" torchvision==0.21.0 \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install dgl \
  -f "https://data.dgl.ai/wheels/torch-2.6/${CUDA_TAG}/repo.html"
uv pip install torch-scatter torch-sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
uv pip install torch-geometric

# ms-pred: base deps only (its cpu/cu* extras would re-pin torch).
uv pip install -e "$THIRD_PARTY/ms-pred"

# FragFM: source only, plus the deps of its that ms-pred does not already cover.
uv pip install -e "$THIRD_PARTY/FragFM" --no-deps
uv pip install lmdb easydict parmap wandb tqdm huggingface_hub
uv pip install --no-deps "descriptastorus @ git+https://github.com/bp-kelley/descriptastorus.git"
uv pip install pandas-flavor  # descriptastorus runtime dep, installed above with --no-deps

uv pip install -e . --no-deps

echo
echo "Env ready: $REPO_ROOT/.venv"
echo "Next: python scripts/preflight.py"
