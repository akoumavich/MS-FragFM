# PROGRESS

Append-only log. Newest entry at the top.

---

## 2026-09-19 — Preflight round 1: 4/8. Four fixes, one of them an upstream bug

Environment built on the A100 without incident (torch 2.6.0+cu124, numpy 2.4.6,
rdkit 2025.03.6, pyg 2.8.0, dgl 2.5.0+cu124, python 3.11.16).

**The single-environment gamble paid off.** `fragfm.process` and
`fragfm.model.flow` import clean under numpy 2 / rdkit 2025.03 despite FragFM
pinning numpy 1.26 / rdkit 2023.9. The GRPO reward can stay in-process.

**FragFM's released preprocessing is broken at HEAD.**
`fragfm/process.py` calls `reconstruct_to_rdmol(..., remove_h=True/False)` in
four places, but `fragfm/utils/mol_ops.py:13` defines no such parameter — it
unconditionally calls `Chem.RemoveHs`. Every call to `process_sample` raises
`TypeError`, so the published data-processing path cannot run as released.
Upstream commit `6b7e648` ("fix: make mol reconstruction work for custom
dataset") changed `mol_generator.py` only and left the callers in `process.py`
stranded. This is not a version-skew artifact; it is broken for everyone.

The repair matters for correctness, not just for getting past the exception.
`remove_h=False` callers feed the result to `remolstar.GetSubstructMatch(canmolh)`
and index into the match to build the fragment reordering map, so that path must
skip *both* `Chem.RemoveHs` *and* the SMILES round-trip inside
`valid_mol_can_with_seg` — the round-trip reorders atoms and would silently
produce a wrong permutation rather than an error. Patched accordingly in
`scripts/apply_patches.py`, and `preflight.py` now verifies the fix by
round-tripping: rebuild the molecule from the permuted `h`/`e_index`/`e` and
require canonical-SMILES equality with the input. Absence of an exception is not
evidence the permutation is right.

Worth reporting to the FragFM authors — it doubles as the contact the proposal
already schedules for week 1.

**Three environment fixes**

- *triton missing → `torch.compile` fails.* `--index-url .../whl/cu124` suppresses
  PyPI, and the cu124 index carries no triton, so torch's `triton==3.2.0`
  dependency was silently dropped. Installed explicitly.
- *`libXrender.so.1` missing → all of `ms_pred` unimportable.*
  `ms_pred/common/__init__.py` eagerly imports `plot_utils` → `rdkit.Chem.Draw`
  → `libXrender`. The cluster image lacks it and we have no root. Patched to a
  PEP-562 module `__getattr__` so the plotting helpers import on first use.
  Nothing on the training or RL path draws molecules. `cairosvg` was already
  lazy, so this is the only blocker of its kind in `common`.
- *bf16 benchmark read 9 TFLOP/s on an A100* — my measurement bug, not the
  hardware: the first matmul pays cuBLAS handle initialisation and kernel
  autotune, which dominated a 20-iteration timing. Added warmup.

**Decision: patch the reference checkouts, do not fork them.**
`scripts/apply_patches.py` applies exact-anchor string replacements, is
idempotent, and fails loudly if an anchor disappears. Each patch carries the
reason inline. This keeps the delta we depend on visible and lets it drop out
when upstream fixes things — a fork would bury both.

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
