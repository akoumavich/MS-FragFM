#!/usr/bin/env bash
# runai submit template for MS-FragFM batch jobs.
#
#   NAME=msfragfm-e0 GPUS=1 SCRIPT=scripts/run_e0.sh bash cluster/submit.sh
#
# The interactive session (1 GPU, sleep infinity) is launched with
# cluster/submit_interactive.sh.
set -euo pipefail

NAME="${NAME:?set NAME}"
SCRIPT="${SCRIPT:?set SCRIPT (path relative to the repo root)}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-64}"
MEM="${MEM:-250G}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
NODE_POOL="${NODE_POOL:-default}"

MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' runai submit "$NAME" \
  -i "$IMAGE" \
  --gpu "$GPUS" --cpu "$CPUS" --cpu-limit 124 --memory "$MEM" --memory-limit 500G --large-shm \
  --pvc home:/home/akoum --pvc nlp-scratch:/scratch \
  --run-as-uid 307371 --run-as-gid 30204 --supplemental-groups 11131,76208,77864 \
  -e HOME=/scratch/home/akoum -e USER=akoum -e LOGNAME=akoum \
  -e TORCHINDUCTOR_CACHE_DIR=/scratch/home/akoum/.cache/torchinductor \
  -e COND="${COND:-}" -e EPOCHS="${EPOCHS:-}" -e BS="${BS:-}" -e LR="${LR:-}" \
  -e RESUME="${RESUME:-auto}" \
  -e N_SPECTRA="${N_SPECTRA:-}" -e GROUP="${GROUP:-}" -e STEPS="${STEPS:-}" \
  -e SPECTRA_PER_BATCH="${SPECTRA_PER_BATCH:-}" -e TAG="${TAG:-}" \
  -e FORMULA_MASK="${FORMULA_MASK:-}" -e TREE_DECODE="${TREE_DECODE:-}" \
  -e VALENCY="${VALENCY:-}" -e TREE_ASSEMBLY="${TREE_ASSEMBLY:-}" \
  -e COMPOSITION="${COMPOSITION:-}" -e RANK="${RANK:-}" \
  -e COND_MODE="${COND_MODE:-}" -e CKPT="${CKPT:-}" \
  -e NODE_NOISE="${NODE_NOISE:-}" -e EDGE_NOISE="${EDGE_NOISE:-}" \
  -e FRAG_TEMP="${FRAG_TEMP:-}" -e N_FRAG="${N_FRAG:-}" \
  -e ARM="${ARM:-}" -e SPECTRA_PER_STEP="${SPECTRA_PER_STEP:-}" \
  -e EULER_STEPS="${EULER_STEPS:-}" \
  -e EXP_NAME="${EXP_NAME:-$NAME}" -e WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}" \
  -e WANDB_DIR=/scratch/home/akoum/.cache/wandb \
  -e MSFRAGFM_DATA=/scratch/home/akoum/data/msfragfm \
  --node-pools "$NODE_POOL" \
  -- bash -lc "cd /scratch/home/akoum/repos/MS-FragFM && source .venv/bin/activate && bash $SCRIPT"
