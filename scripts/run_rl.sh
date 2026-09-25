#!/usr/bin/env bash
# RL / self-conditioning arm, for cluster/submit.sh.
#
#   NAME=rl-selfcond ARM=selfcond SCRIPT=scripts/run_rl.sh bash cluster/submit.sh
#
# selfcond is the diagnostic and needs no reward; grpo and dmpo reweight the same
# rollouts by it, so all three are paired at equal rollout budget.
set -euo pipefail

export EXP_NAME="${EXP_NAME:-rl_${ARM:-selfcond}}"
export WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}"

python scripts/train_rl.py \
  --arm "${ARM:-selfcond}" \
  --ckpt "${CKPT:-results/e3b-xattn.pt}" \
  --steps "${STEPS:-300}" \
  --spectra-per-step "${SPECTRA_PER_STEP:-8}" \
  --group "${GROUP:-8}" \
  --euler-steps "${EULER_STEPS:-100}" \
  --lr "${LR:-1e-5}" \
  --tag "${TAG:-${EXP_NAME}}"
