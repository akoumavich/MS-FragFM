"""Weights & Biases logging.

Run name comes from `EXP_NAME` so it matches the runai job name, which is what
makes a wandb run findable from `runai list` and back again.  Project defaults to
MS-FragFM, overridable with `WANDB_PROJECT`.

Every call degrades to a no-op if wandb is missing, disabled or unreachable.
A training run that has survived nine hours should not die at the logging call.
"""

import hashlib
import os


def init(default_name, config=None, project="MS-FragFM"):
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        name = os.environ.get("EXP_NAME") or default_name
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", project),
            name=name,
            # A preempted job restarts into the same wandb run rather than
            # littering the project with fragments of one experiment.  The id is
            # derived from the name, so no state has to survive the preemption.
            id=hashlib.md5(name.encode()).hexdigest()[:16],
            resume="allow",
            config=config,
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] disabled: {type(exc).__name__}: {exc}")
        return None


def log(run, metrics, step=None):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed: {type(exc).__name__}: {exc}")


def finish(run):
    if run is not None:
        try:
            run.finish()
        except Exception:  # noqa: BLE001
            pass
