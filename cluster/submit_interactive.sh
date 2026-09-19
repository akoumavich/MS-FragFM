#!/usr/bin/env bash
# 1-GPU interactive session; exec into it with:
#   runai exec -it msfragfm-dev -- bash
set -euo pipefail

MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' runai submit "${NAME:-msfragfm-dev}" \
  -i "${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}" \
  --interactive \
  --gpu 1 --cpu 64 --cpu-limit 124 --memory 128G --memory-limit 256G --large-shm \
  --pvc home:/home/akoum --pvc nlp-scratch:/scratch \
  --run-as-uid 307371 --run-as-gid 30204 --supplemental-groups 11131,76208,77864 \
  -e HOME=/scratch/home/akoum -e USER=akoum -e LOGNAME=akoum \
  -e TORCHINDUCTOR_CACHE_DIR=/scratch/home/akoum/.cache/torchinductor \
  -e MSFRAGFM_DATA=/scratch/home/akoum/data/msfragfm \
  --node-pools default \
  --command sleep --args infinity
