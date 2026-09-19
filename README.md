# MS-FragFM

Spectrum-conditioned fragment-level graph flow matching for MS/MS structure elucidation.

Plan: `Fragment-Level Graph Flow Matching for MS MS Structure Elucidation — Proposal v2.md`
(kept outside this repo, in the parent directory).
Progress log: [PROGRESS.md](PROGRESS.md) · Measurements: [RESULTS.md](RESULTS.md)

## Layout

```
msfragfm/        library code
scripts/         runnable entry points
cluster/         runai submit templates
configs/         experiment configs
results/         JSON measurement dumps (committed; large artifacts are not)
```

Reference checkouts (FragFM, ms-pred, FRIGID, Graph-GRPO, remdm, ml-fs-dfm) live as
siblings of this repo and are never modified in place.

## Setup (cluster)

```bash
cd ~/repos/MS-FragFM
bash scripts/setup_env.sh          # clones FragFM + ms-pred if missing, builds .venv
source .venv/bin/activate
python scripts/preflight.py        # must pass before anything else
```

Data root defaults to `$HOME/data/msfragfm`; override with `MSFRAGFM_DATA`.
