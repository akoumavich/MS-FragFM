# PROGRESS

Append-only log. Newest entry at the top.

---

## 2026-09-19 — rBRICS reverses the E0-a decision; the attachment latent is global

**Decomposition: rBRICS, not BRICS.** E0-a chose BRICS on edge-slot compression
(14.4x vs 9.0x) before coverage existed as a number. With the rBRICS arm in,
rBRICS reaches **higher coverage with a 35% smaller pool at every scale**: 82.3%
vs 72.5% of test structures at a 200k-molecule pool, from 53,220 fragments
against 81,739. Finer fragments recur across molecules more, so each one is worth
more. Full table in RESULTS.md R5.

Scoring all three axes: compression is *not* binding (9.0x still triples the
proposal budget of 3-6x), coverage is (it caps exact top-1), and pool fraction
per bag draw is (384/53,220 = 0.72% against 0.47%). rBRICS wins both binding axes
and loses the one with slack.

The pool-fraction column is what decided it, and it only exists because of
yesterday's finding that the bag is redrawn every Euler step: a smaller pool is
explored more thoroughly per draw, and that advantage compounds exactly when step
count is cut for speed. Compression could not have told us this. Cost: 8.6
fragments per molecule instead of 7.1, ~20% slower to decompose.

**The attachment-point latent is global — verified, and it dents C2.** Flagged by
a parallel review and confirmed in the code: `gen_z = torch.randn(bs,
latent_z_dim)` is one latent *per molecule*, and `AE.decode` turns it into a
single graph-level embedding (`ae.py:205`) from which inter-fragment bonds are
predicted. So the generative variables are coarse nodes, coarse edges, and one
global z. Fragment identity and fragment connectivity take per-variable
advantages cleanly; **attachment-point choice has only z**, so for that failure
mode Method D's structured advantage degrades to a scalar — the same aggregation
loss the proposal criticises FRIGID for, arriving through a different door. It
bites because attachment errors are the isomer confusions MS/MS resolves worst:
right fragments, wrong joins.

Not yet quantified, and it is quantifiable. The decoder is deterministic given
(coarse graph, z), so if coarse-to-fine reconstruction is near-lossless on
MassSpecGym chemistry then attachment is nearly determined by the coarse graph
and z carries little of the decision. Week-3 architecture decision either way,
per RESULTS.md R6.

**A quantity that gates everything, and is cheap.** The ceiling on exact top-1
factorises as P(all fragments of the target are in the pool) x P(coarse-to-fine
reconstruction correct) = 0.823 x (unmeasured) at a 200k rBRICS pool. That
product bounds every accuracy number the project can report, independent of
generator, oracle and RL. Promoted in the proposal to a week-1 item rather than
an ablation.

**Built:** `scripts/build_msg_lmdb.py` — decomposes MassSpecGym into FragFM's
LMDB format with train_/valid_/test_ key prefixes taken from the benchmark's own
folds, so FragFM's `process_fragment_from_lmdb.py` runs unchanged afterwards.
Prerequisite for measuring the second ceiling factor and for any training.

**Converging independently:** a parallel reading of FragFM reached the same
conclusion about spectrum-conditioning the bag. Their reweighting hook already
exists in code — `random_select_frags_by_occurance` multiplies the occurrence
prior by `exp(-bag_guide_strength * property_mse)` when a discriminator is loaded
— so the mechanism is a drop-in. Our version is stronger than a learned property
score: subformula-of-precursor is an exact constraint, not a prediction.

**Next:** build the MassSpecGym LMDB, then coarse-to-fine reconstruction accuracy
(the second ceiling factor). E0-b throughput still pending.

---

## 2026-09-19 — R3 retracted. The bag is swappable; the cost is coupled to step count

**Correction, prompted by Mohsen.** R3 called the unseen-fragment rate a
"representational ceiling", which implies a limit set by the training data. That
is wrong. FragFM has no fragment-identity embedding table: `FragToVect`
(`fragfm/model/flow.py:11`) encodes each candidate from its own graph — atomic
numbers, junction counts, bond types, message passing — so a fragment never seen
in training is embedded and scored like any other. The candidate pool is a data
artifact loaded at inference and swappable. This is precisely the fixed-
vocabulary trap FragFM was built to escape, and I wrote it up as though the trap
were still there. The R3 numbers are correct; the conclusion was not. Retracted
in RESULTS.md R4.

**Read correctly, E0-c is a design curve and it is encouraging.** Coverage of the
test fold: 44.5% from the full train fold (9,296 fragments), 72.5% from 200k
corpus molecules (81,739 fragments), still climbing steeply with 4M available.
R3's alarming 0.286 was an artifact of harvesting from only 5,000 molecules. The
pool is a knob, and a cheap CPU-only one.

**What reading the mechanism did surface — and this one is real.** The bag is
redrawn at **every Euler step** (`mol_generator.py:398-412`): 384 fragments by
occurrence weighting, unioned with whatever is already placed. A 500-step
generation is 500 independent draws, i.e. a stochastic search over the pool.

So **cutting 500 steps to 20 also cuts bag resampling 25-fold.** The proposal's
budget table treats step reduction and fragment-level representation as
independent multiplicative levers; they are not. Pool scale and step count pull
against each other — a larger pool raises the ceiling but needs more draws to
exploit, and cutting steps shrinks exploration exactly when the pool is largest.
FragFM never cuts steps, so its paper never had to confront this. This would have
surfaced in week 6 as an unexplained quality collapse under step reduction.

**The fix is on-thesis, which is why I think it is right rather than convenient.**
Stop drawing the bag by occurrence and select it from the spectrum. MS/MS peaks
*are* fragment masses: admissible fragments have formulas that are subformulas of
the precursor, and informative ones match observed peaks within tolerance. Both
constraints are exact, free to evaluate, and instance-specific — which is what
the conditional setting needs and what occurrence weighting cannot provide. A
spectrum-filtered bag raises coverage and removes the step-count dependence
together, because few draws suffice once the candidates are right.

That is the proposal's thesis applied one level lower: generate in the units the
evidence is expressed in, and *select* in them too. Added to Method A as a fourth
conditioning signal, flagged as conditioning the support rather than the network.

**Next:** E0-d — how far do formula and peak filtering prune an 81,739-fragment
pool, and what coverage survives. Then E0-b throughput, still pending.

---

## 2026-09-19 — Held-out vocabulary: a representational ceiling at 28.6%, and a decision reopened

Preflight 8/8 (`setuptools` fixed `torch.compile`).

**The finding: 71.4% of MassSpecGym test molecules contain at least one BRICS
fragment that never appears in the train fold.** The complete train-fold
vocabulary therefore caps exact top-1 at **28.6%**, and the top-V curve is flat
from V=2,000 — the missing fragments are not rare training fragments, they are
absent from the training molecules entirely. Details in RESULTS.md R3.

This matters because the field sits at ~18% top-1. Ten points of headroom on a
ceiling imposed by our own representation is not comfortable, and it is a
ceiling of the same kind as C3's verifier ceiling, sitting on the representation
instead of the oracle. The two stack. It caps *exact* top-1 only — MCES and
Tanimoto degrade gracefully — and MassSpecGym's MCES>=10 split is designed to
make held-out structures distant, so a large unseen-fragment share is the split
working as intended rather than a defect. It is still the binding number.

**I was wrong to close the BRICS-vs-rBRICS question yesterday.** I dropped
rBRICS on compression alone (9.0x vs 14.4x) before measuring held-out coverage.
Finer fragments recur more, so rBRICS should have a *higher* ceiling, and it
already showed better in-sample top-V coverage (0.138 vs 0.099 at V=100). That
is a trade between compression and ceiling, not a straight loss. Reopened; E0-c
measures both arms.

**New experiment E0-c (`scripts/vocab_ceiling.py`).** Does the ceiling lift with
vocabulary scale, and how fast? Harvest fragments from the benchmark's
4M-molecule MCES-2-disjoint corpus at 5k -> 1M molecules and re-measure test
coverage, for both decompositions. Fragments recur across molecules far more
than molecules recur, so this should lift substantially; if it does, the fix is
CPU-only and cheap, and it becomes a concrete argument for the pretraining half
of the plan rather than a scaling claim taken on faith. The script decomposes
the largest slice once and reads nested prefixes off it, so the whole curve
costs a single pass.

If the ceiling stays low even at 1M molecules, the representation needs an
atom-level escape hatch for unseen fragments, and that is a week-3 design change
rather than a week-9 surprise.

**Fixed:** `gdown` 5.x dropped `--id`; the bare file id still works.

**Next:** E0-c, then E0-b throughput once the FragFM checkpoints land.

---

## 2026-09-19 — E0-a done. Base decomposition settled; compression is 2-4x better than budgeted

Preflight is green apart from `torch.compile`, and that had nothing to do with
triton. **triton was installed the whole time; it imports `setuptools` at
runtime, and a uv venv ships without setuptools.** torch's `has_triton()`
swallows the `ImportError` and reports triton as "not installed or too old",
which sends you auditing the wrong package. Installing `setuptools` is the fix.
Noted because the error message is actively misleading.

**E0-a result (see RESULTS.md R2): BRICS with relaxed canonicalisation.**
99.8% of MassSpecGym decomposes, 7.1 fragments per molecule, 14.4x edge-slot
reduction, 3,485 distinct fragments over 5,000 structures, 2.1 mol/s/core.

Three things follow.

*FragFM stays the base.* The question E0-a existed to answer was whether
FragFM's decomposition survives MassSpecGym chemistry. At 99.8% it does, so the
FlowMS/DiffMS fallback is off the table and week 3-4 can commit to conditioning
FragFM rather than hedging.

*rBRICS is a straight loss here, which was not obvious.* It cuts more finely --
8.6 fragments instead of 7.1 -- and that makes compression *worse*, 9.0x against
14.4x, because more fragments means more fragment-level edge slots. It is also
slower to compute. Finer decomposition is not a free upgrade; this project wants
fewer, larger generative units, and that is also what keeps the GRPO horizon
short. Dropping rBRICS from consideration.

*Relaxed canonicalisation is the chemically correct setting, not a leniency
knob.* It differs from strict only in allowing a formal charge on N/O/S carrying
excess valence, and MassSpecGym contains genuinely charged species. It recovers
2.6% of structures that strict silently drops.

**The compression lever measures 14.4x against 3-6x budgeted.** Proposal updated,
but conservatively: the edge-slot ratio is a structural bound, not a speedup --
the coarse-to-fine autoencoder and everything linear in node count do not shrink
with it. 3-6x stays the planning figure until E0-b measures wall-clock. What the
number does say is that the 100x target has more headroom than the budget table
assumed.

**Adduct is a conditioning input.** v1.5 carries `[M+H]+` (195,237) and
`[M+Na]+` (35,867). Sodiated species fragment differently from protonated ones
and they are 15.5% of the data, so the model should be told which it is looking
at rather than inferring the charge carrier from the peaks. Added to Method A's
conditioning list; it is nearly free.

**Preprocessing is cheap enough not to plan around.** 134 mol/s on 64 cores puts
the 31,602 MassSpecGym structures at ~4 minutes and the benchmark's 4M-molecule
MCES-2-disjoint pretraining corpus at ~8.3 hours on one node.

**Still open:** the fragment vocabulary top-V coverage curve — 3,485 distinct
fragments over 5,000 molecules implies a long tail, and how long decides how
hard FragFM's stochastic fragment bag has to work. Measured, not yet read back.

**Next:** E0-b, wall-clock sampling throughput of unconditioned FragFM from the
released checkpoint. That is the number the proposal calls blocking, and it
sizes group size, training-set size and the whole RL schedule.

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
