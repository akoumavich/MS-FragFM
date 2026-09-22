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
  -e EXP_NAME="${EXP_NAME:-$NAME}" -e WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}" \
  -e WANDB_DIR=/scratch/home/akoum/.cache/wandb \
  -e MSFRAGFM_DATA=/scratch/home/akoum/data/msfragfm \
  --node-pools "$NODE_POOL" \
  -- bash -lc "cd /scratch/home/akoum/repos/MS-FragFM && source .venv/bin/activate && bash $SCRIPT"
