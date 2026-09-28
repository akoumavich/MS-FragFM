#!/usr/bin/env bash
# Self-conditioning arm, for cluster/submit.sh.
#
#   NAME=sc-selfcond ARM=selfcond SCRIPT=scripts/run_rl.sh bash cluster/submit.sh
#   NAME=sc-teacher  ARM=teacher  SCRIPT=scripts/run_rl.sh bash cluster/submit.sh
#
# The two arms share every line of code and differ only in what sits in the
# revealed slots -- the model's own sampled fragments, or the true ones -- so any
# difference between them is the state distribution and nothing else.
set -euo pipefail

export EXP_NAME="${EXP_NAME:-sc_${ARM:-selfcond}}"
export WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}"

python scripts/train_rl.py \
  --arm "${ARM:-selfcond}" \
  --ckpt "${CKPT:-results/e3b-xattn.pt}" \
  --steps "${STEPS:-2000}" \
  --bs "${BS:-64}" \
  --lr "${LR:-1e-5}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --time-power "${TIME_POWER:-1.0}" \
  --resume "${RESUME:-auto}" \
  --tag "${TAG:-${EXP_NAME}}"
