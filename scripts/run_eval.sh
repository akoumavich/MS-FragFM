#!/usr/bin/env bash
# Evaluation arm, for cluster/submit.sh.
#
#   NAME=eval-2k N_SPECTRA=2000 SCRIPT=scripts/run_eval.sh bash cluster/submit.sh
#
# Constraint flags default to everything on, which is the R22 configuration.
# Override any of them to run an ablation as its own job:
#
#   NAME=eval-2k-noval VALENCY=off N_SPECTRA=2000 SCRIPT=scripts/run_eval.sh \
#     bash cluster/submit.sh
set -euo pipefail

export EXP_NAME="${EXP_NAME:-eval}"
export WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}"

python scripts/eval_denovo.py \
  --ckpt "${CKPT:-results/flow_spectrum.pt}" \
  --n-spectra "${N_SPECTRA:-2000}" \
  --group "${GROUP:-16}" \
  --steps "${STEPS:-100}" \
  --spectra-per-batch "${SPECTRA_PER_BATCH:-8}" \
  --formula-mask "${FORMULA_MASK:-on}" \
  --tree-decode "${TREE_DECODE:-on}" \
  --valency "${VALENCY:-on}" \
  --tree-assembly "${TREE_ASSEMBLY:-on}" \
  --composition "${COMPOSITION:-on}" \
  --rank "${RANK:-frequency}" \
  --cond-mode "${COND_MODE:-real}" \
  --frag-temp "${FRAG_TEMP:-1.0}" \
  ${NODE_NOISE:+--node-noise "$NODE_NOISE"} \
  ${EDGE_NOISE:+--edge-noise "$EDGE_NOISE"} \
  --tag "${TAG:-${EXP_NAME}}"
