# PROGRESS

Append-only log. Newest entry at the top.

---

## 2026-09-19 — Repo bootstrap, environment strategy, week-1 scaffolding

**Done**

- Repo skeleton: `msfragfm/` package, `scripts/`, `cluster/`, `configs/`, `results/`.
- `scripts/setup_env.sh` — single environment for generator + oracle.
- `scripts/preflight.py` — eight checks gating everything downstream.
- `scripts/download_massspecgym.py` — targeted HF file fetch + split report.
- `scripts/fragment_coverage.py` — experiment **E0-a** (see below).

**Decision: one environment, not two.**
FragFM pins `numpy==1.26 / rdkit==2023.9 / torch==2.1`; ms-pred wants
`numpy>=2 / rdkit==2025.3 / torch==2.6 + dgl`. On paper these are irreconcilable.
Grepping FragFM's source finds no numpy-1-only aliases and no rdkit-2023-only
APIs, so the pins look conservative rather than load-bearing. We therefore
resolve on ms-pred's newer stack and install FragFM with `--no-deps`.
Why it matters: GRPO makes ~160k oracle calls per epoch, and a two-environment
split would force all of them through IPC. `preflight.py` is the empirical test
of this gamble; the fallback is a two-env split with the oracle behind a socket.

**New experiment E0-a, pulled forward from week 3.**
The proposal schedules "validate fragment vocabulary coverage" in week 3, but it
is a *base-model selection* question, not a tuning question: if FragFM's BRICS
decomposition fails on a large share of MassSpecGym, FragFM cannot be the base
and we fall back to FlowMS/DiffMS. It costs minutes on 64 cores, so it runs in
week 1, before any environment work is sunk into the wrong base.
`scripts/fragment_coverage.py` measures, over `brics`/`rbrics` x strict/relaxed
canonicalisation: decomposition success rate, fragment-count distribution,
**measured edge-slot reduction** (the proposal's 3-6x lever), fragment
vocabulary size, and preprocessing throughput.

**Findings while reading the reference code**

- MassSpecGym v1.5 is a single file, `data/MassSpecGym1.5.tsv`, inside a 52.8 GB
  HF repo — fetch per-file, never snapshot.
- The HF repo already ships
  `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv` and
  `MassSpecGym1.5_pretraining_molecules_{2.5M,50M}_tani070.txt`. The proposal
  budgets work for MCES-filtering a pretraining corpus against the test fold;
  the benchmark authors have already done it. Proposal updated.
- GLACIER's `predict_smis_joint.py` documents that **featurization, not the GPU
  forward, is the inference bottleneck** — it already runs a CPU worker pool with
  a per-worker LRU cache and size-sorted batching. This changes the oracle
  throughput plan: the RL reward loop is CPU-bound, so it wants many CPU cores
  next to the GPU, and SMILES-level caching is worth more than a bigger batch.
  Proposal updated.
- `ms_pred/glacier/` and `ms_pred/iceberg/` ship without `__init__.py`; import
  their modules directly (`from ms_pred.glacier import joint_model`).

**Next (blocking on the interactive A100 session)**

1. `bash scripts/setup_env.sh && python scripts/preflight.py`
2. `python scripts/download_massspecgym.py --which main`
3. `python scripts/fragment_coverage.py --n 5000`

**Housekeeping**

- The `origin` remote URL embeds a GitHub PAT in plaintext. It is readable by
  anything that can run `git remote -v`. Rotate it and move it to a credential
  helper or `~/.git-credentials`.
