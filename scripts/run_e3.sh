#!/usr/bin/env bash
# E3 arm, for cluster/submit.sh.  COND is "spectrum" or "none".
#
#   NAME=e3-spectrum COND=spectrum SCRIPT=scripts/run_e3.sh bash cluster/submit.sh
#   NAME=e3-blind    COND=none     SCRIPT=scripts/run_e3.sh bash cluster/submit.sh
#
# The two arms are separate jobs so they run on separate GPUs at the same time:
# the spectrum-blind control is only useful if it lands with the real one.
set -euo pipefail

# EXP_NAME names the wandb run; submit.sh defaults it to the runai job name so
# a run can be traced back to its job and forward from it.
export EXP_NAME="${EXP_NAME:-flow_${COND:-spectrum}}"
export WANDB_PROJECT="${WANDB_PROJECT:-MS-FragFM}"

python scripts/train_flow_cond.py \
  --cond "${COND:-spectrum}" \
  --epochs "${EPOCHS:-30}" \
  --bs "${BS:-256}" \
  --lr "${LR:-2e-4}" \
  --tag "flow_${COND:-spectrum}"
