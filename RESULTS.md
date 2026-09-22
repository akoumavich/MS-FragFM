# RESULTS

Measurements only. Every number states the hardware, the code revision, and the
data it was taken on. Narrative and decisions live in [PROGRESS.md](PROGRESS.md).

---

## R1 — Environment preflight

`scripts/preflight.py` @ `e217d92` + patches · 1x A100-SXM4-80GB (runai, `nvcr.io/nvidia/pytorch:24.10-py3`)

| Check | Result |
| --- | --- |
| Stack | py 3.11.16 · torch 2.6.0+cu124 · numpy 2.4.6 · rdkit 2025.03.6 · pyg 2.8.0.post1 · dgl 2.5.0+cu124 |
| bf16 dense matmul (4096^3, warmed) | **219 TFLOP/s** |
| `torch.compile` (inductor) | pass (after installing `setuptools`) |
| FragFM imports under numpy 2 / rdkit 2025.03 | pass |
| FragFM BRICS decomposition + round-trip | pass, exact on 6/6 |
| ms-pred GLACIER + ICEBERG import | pass |

219 TFLOP/s against A100 bf16 peak of 312 is 70% of peak on a plain
`torch.matmul`, which is the expected ceiling for a non-fused dense GEMM. The
GPU is healthy; nothing here is the bottleneck.

### R1.1 — Fragment compression, preliminary

From the six preflight molecules. **Not a distribution** — six hand-picked
structures, reported only because the numbers are far from the proposal's
estimate and E0-a is worth running before anything is built on that estimate.

| Molecule | Heavy atoms | Fragments | Atom edge slots | Frag edge slots | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| quercetin | 22 | 7 | 231 | 21 | 11.0x |
| caffeine | 14 | 4 | 91 | 6 | 15.2x |
| aspirin | 13 | 5 | 78 | 10 | 7.8x |
| sultam | 23 | 3 | 253 | 3 | 84.3x |
| dipeptide | 21 | 8 | 210 | 28 | 7.5x |
| posaconazole-like | 45 | 10 | 990 | 45 | 22.0x |

Edge-slot ratio is a **structural upper bound on** the speedup, not the speedup:
it ignores the coarse-to-fine autoencoder, everything linear in node count, and
per-step fixed costs. The proposal budgets 3-6x for this lever, which stays the
planning number until E0-a and E0-b measure it properly.

---

## R2 — E0-a: fragment decomposition of MassSpecGym v1.5

`scripts/fragment_coverage.py` @ `faff7d4` · 5,000 unique structures sampled from
MassSpecGym v1.5 (seed 0) · 64 CPU cores

### Dataset, as loaded

231,104 spectra · 31,602 unique SMILES · 28,929 unique InChIKey14.

| fold | spectra | | adduct | spectra | | instrument | spectra |
| --- | ---: | --- | --- | ---: | --- | --- | ---: |
| train | 194,119 | | `[M+H]+` | 195,237 | | Orbitrap | 172,058 |
| val | 19,429 | | `[M+Na]+` | 35,867 | | QTOF | 53,823 |
| test | 17,556 | | | | | | |

Columns: `identifier, mzs, intensities, smiles, inchikey, formula,
precursor_formula, parent_mass, precursor_mz, adduct, instrument_type,
collision_energy, fold, simulation_challenge`.

Two adducts only, and `[M+Na]+` is 15.5% of the data — sodiated species
fragment differently from protonated ones, so adduct is a conditioning input,
not a nuisance variable.

### The decomposition grid

| decomposition | canonicalisation | success | mean n_frag | edge-slot reduction | vocab | throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BRICS | relaxed | **99.8%** | 7.1 | **14.4x** | 3,485 | 2.1 mol/s/core |
| BRICS | strict | 97.2% | 7.1 | 14.1x | 3,398 | 2.1 mol/s/core |
| rBRICS | relaxed | 99.8% | 8.6 | 9.0x | 3,169 | 1.7 mol/s/core |
| rBRICS | strict | 97.2% | 8.6 | 9.3x | 3,110 | 1.9 mol/s/core |

Edge-slot reduction is aggregate `sum[n(n-1)/2] / sum[k(k-1)/2]` over heavy atoms
n and fragments k.

**Decision: BRICS with relaxed canonicalisation.** Highest coverage and the best
compression of the four. Relaxed canonicalisation differs from strict only in
allowing a formal charge on N/O/S carrying excess valence, and it recovers 2.6%
of structures that strict drops — MassSpecGym contains genuinely charged species,
so this is the chemically correct setting rather than a leniency knob.

**rBRICS cuts more finely and compresses worse.** More fragments means more
fragment-level edge slots (8.6 vs 7.1 fragments costs 9.0x instead of 14.4x),
and it is also slower to compute. Finer decomposition is not free, and for this
project it is a straight loss: we want *fewer, larger* generative units.

### Consequences

**Coverage is not a risk.** 99.8% of MassSpecGym decomposes. The proposal's risk
row on fragment vocabulary missing natural-product chemistry does not fire for
coverage. Whether the *vocabulary* is tractable is a separate question, answered
by the top-V curve (pending).

**Preprocessing is cheap.** 2.1 mol/s/core x 64 cores = 134 mol/s: the 31,602
unique MassSpecGym structures take ~4 minutes, and the benchmark's 4M-molecule
MCES-2-disjoint pretraining corpus takes ~8.3 hours on one 64-core node. Neither
needs a scheduling decision.

**The compression lever measures 14.4x, against 3-6x budgeted.** This is a
structural bound on the speedup, not the speedup: it ignores the coarse-to-fine
autoencoder, everything linear in node count, and per-step fixed costs. It does
say the fragment-level lever is stronger than the proposal assumed, and that the
100x target has more headroom than the budget table suggests. E0-b (wall-clock
sampling) converts it into a real number; until then 3-6x stays the planning
figure.

### R2.1 — distribution detail (BRICS, relaxed)

| quantity | value |
| --- | ---: |
| mean heavy atoms | 27.9 |
| mean fragments | 7.06 |
| fragments, p95 | 15 |
| fragments, max | 31 |
| molecules that do not decompose (1 fragment) | 1.9% |
| edge-slot reduction, aggregate | 14.4x |
| edge-slot reduction, mean of per-molecule ratios | 33.7x |

Two notes. Mean heavy-atom count is **27.9**, not the 23 the proposal's budget
arithmetic assumes, so the atom-level baseline is worse than assumed and the
reduction correspondingly larger. And the aggregate ratio (14.4x) is the honest
one: the mean of per-molecule ratios reads 33.7x because it is dominated by
small molecules, where a 3-fragment decomposition of a 23-atom molecule gives
253/3. Aggregate is reported everywhere else in this file.

At p95 = 15 fragments the fragment graph has 105 edge slots, so the dense
fragment-level edge tensor stays small enough that sparsity engineering is a
second-order concern at this level — unlike at atom level.

**Vocabulary coverage below is in-sample and therefore not yet meaningful** —
the vocabulary was fitted on the same 5,000 molecules it was scored against, so
top-V reaches 1.0 the moment V exceeds the 3,485-entry vocabulary. Recorded only
to be superseded by the held-out run.

| top-V | in-sample coverage |
| ---: | ---: |
| 100 | 0.099 |
| 500 | 0.383 |
| 1000 | 0.547 |
| 5000 | 1.000 (trivial) |

Even in-sample, a 1,000-fragment fixed vocabulary fully covers only 55% of
molecules. That is already an argument that a fixed vocabulary is the wrong
design and FragFM's stochastic fragment bag is load-bearing rather than a
refinement. The held-out number will be worse.

---

## R3 — E0-a held-out: the representational ceiling

`scripts/fragment_coverage.py --mode holdout --n 5000` @ `b5f4af6` · BRICS,
relaxed · vocabulary fitted on the MassSpecGym train fold, scored on the test fold

| | |
| --- | ---: |
| train-fold vocabulary | 3,117 fragments |
| test-fold vocabulary | 2,663 fragments |
| **test molecules containing >=1 fragment absent from train** | **71.4%** |
| mean share of a test molecule's fragments unseen in train | 16.7% |

| top-V | in-sample | held-out |
| ---: | ---: | ---: |
| 100 | 0.144 | 0.027 |
| 250 | 0.304 | 0.071 |
| 500 | 0.445 | 0.131 |
| 1,000 | 0.607 | 0.202 |
| 2,000 | 0.795 | 0.246 |
| 3,117 (full) | 1.000 | **0.286** |

**The curve is flat by V=2,000 at 0.286.** Enlarging a vocabulary drawn from the
same 5,000 training molecules does nothing after that point — the missing
fragments are not rare training fragments, they are fragments that are simply
absent from the training molecules.

### What this means

A generator that assembles molecules from a fragment vocabulary cannot emit a
molecule containing a fragment outside that vocabulary, whatever it learns. With
the complete train-fold vocabulary, **exact top-1 is capped at 28.6%**.

Against a field at ~18% top-1 this is not fatal, but ~10 points of headroom is
not a comfortable margin, and it shrinks as the field improves. It is a ceiling
of the same kind as the proposal's verifier ceiling (C3), sitting on the
representation rather than the oracle, and the two stack.

Three qualifications. It caps **exact** top-1 only; MCES distance and Tanimoto
degrade gracefully. It is measured on 5,000 train molecules, so it is a lower
bound on what the full 194k-spectrum train fold supports. And MassSpecGym's
MCES>=10 split is *designed* to make held-out structures distant, so a large
unseen-fragment share is the split working as intended, not a defect.

The escape is vocabulary scale, and it should be cheap: fragments recur across
molecules far more than molecules recur. E0-c measures it.

### This reopens the BRICS-vs-rBRICS decision

E0-a chose BRICS on compression (14.4x vs 9.0x) before this ceiling was known.
Finer fragments recur more often, so rBRICS should have a **higher** ceiling —
and rBRICS already showed better in-sample top-V coverage (0.138 vs 0.099 at
V=100). The decision is now a trade between compression and ceiling rather than
a clear win, and E0-c measures both arms before it stands.

---

## R4 — E0-c: the ceiling is a property of the inference-time bag, not of training

`scripts/vocab_ceiling.py --corpus-n 200000` @ `b5f4af6` · BRICS, relaxed ·
test = 3,160 MassSpecGym test-fold structures

**R3's framing was wrong and is retracted.** R3 called the unseen-fragment rate a
"representational ceiling", which implies a limit set by the training data.
FragFM has no fragment-identity embedding table: `FragToVect` encodes a fragment
from its own graph — atomic numbers, junction counts, bonds, message passing —
so a fragment never seen in training can still be embedded and scored. The
candidate pool is a data artifact, loaded at inference and swappable. R3's
numbers are correct; the conclusion drawn from them was not.

| fragment pool | molecules | vocabulary | coverage of test |
| --- | ---: | ---: | ---: |
| MSG train fold (5k sample, R3) | 5,000 | 3,117 | 0.286 |
| MSG train fold (full) | 25,023 | 9,296 | 0.445 |
| MCES-2-disjoint corpus | 5,000 | 4,476 | 0.345 |
| corpus + train | 5,000 | 12,260 | 0.496 |
| corpus | 20,000 | 13,071 | 0.487 |
| corpus + train | 20,000 | 19,711 | 0.562 |
| corpus | 50,000 | 26,780 | 0.579 |
| corpus + train | 50,000 | 32,480 | 0.624 |
| corpus | 100,000 | 45,754 | 0.644 |
| corpus + train | 100,000 | 50,782 | 0.669 |
| corpus | 200,000 | 77,361 | 0.707 |
| **corpus + train** | **200,000** | **81,739** | **0.725** |

Read as a design curve, not a limit. Coverage is still climbing steeply at 200k
molecules and the corpus holds 4M, so the pool is a knob and it is a cheap one —
CPU-only, ~8 hours for the full 4M on one node. R3's 0.286 was an artifact of
harvesting from 5,000 molecules; the full train fold alone gives 0.445.

### What the bag mechanism actually does, and the cost it hides

`mol_generator.py:398-412`: at **every Euler step** the sampler draws a fresh
`n_base_frag` fragments (384 in the released config) by occurrence-weighted
sampling without replacement, unions them with the fragments already placed in
the graph, and predicts over that union. Fragments already placed persist. So a
500-step generation makes 500 independent draws of 384 fragments from the pool,
with the current state carried along — a stochastic search over the pool, not a
single fixed bag.

**This couples the two largest levers in Method B, which the budget treats as
independent.** Cutting 500 Euler steps to 20 does not only incur the
factorization error of section 7; it also cuts bag resampling from 500 draws to
20, shrinking the explored pool by the same 25x. The larger the pool, the worse
this bites: at 200k molecules the pool is 81,739 fragments, and 20 draws of 384
occurrence-weighted samples touch a vanishing share of it.

So pool scale and step count pull against each other. Growing the pool raises the
coverage ceiling but needs more draws to exploit; cutting steps for speed shrinks
exploration exactly when the pool is largest. Neither the proposal nor FragFM's
paper addresses this, because FragFM never cuts steps.

### The escape, and it is on-thesis

Stop sampling the bag by occurrence and select it from the spectrum. MS/MS peaks
*are* fragment masses: every fragment worth considering has a formula that is a
subformula of the precursor, and the informative ones have masses matching
observed peaks within tolerance. A spectrum-filtered pool is small, targeted and
free to compute, which raises coverage and removes the dependence on step count
at the same time — few draws suffice when the candidates are already the right
ones.

This is the same argument the proposal's thesis already makes, applied one level
lower: generate in the units the evidence is expressed in, and *select* in them
too. Unlike occurrence-weighted sampling it is instance-specific, which is what
the conditional setting calls for. Measuring how far formula and peak filtering
prune an 81,739-fragment pool, and what coverage survives, is E0-d.

---

## R5 — E0-c, rBRICS arm: the E0-a decomposition decision is reversed

Same run, rBRICS arm. Coverage of the same 3,160 test structures.

| fragment pool | BRICS vocab | BRICS cov. | rBRICS vocab | rBRICS cov. |
| --- | ---: | ---: | ---: | ---: |
| MSG train fold (full) | 9,296 | 0.445 | 7,326 | **0.544** |
| corpus 20k | 13,071 | 0.487 | 10,680 | **0.607** |
| corpus 50k | 26,780 | 0.579 | 20,054 | **0.712** |
| corpus 100k | 45,754 | 0.644 | 32,135 | **0.768** |
| corpus 200k | 77,361 | 0.707 | 50,873 | **0.812** |
| corpus + train 200k | 81,739 | 0.725 | 53,220 | **0.823** |

**rBRICS reaches higher coverage with a 35% smaller pool, at every scale.** Finer
fragments recur across molecules more often, so each one is worth more: rBRICS
needs 53,220 fragments to cover 82.3% of test structures where BRICS needs
81,739 to cover 72.5%.

### Decision: rBRICS, reversing E0-a

E0-a chose BRICS on edge-slot compression (14.4x vs 9.0x) before coverage was
measured. Scoring the two on all three axes that matter:

| axis | BRICS | rBRICS | binding? |
| --- | ---: | ---: | --- |
| edge-slot reduction | **14.4x** | 9.0x | no — both far above the 3-6x budgeted |
| test coverage @200k | 0.725 | **0.823** | yes — caps exact top-1 |
| pool fraction per bag draw (384/pool) | 0.47% | **0.72%** | yes — see R4 step coupling |

rBRICS wins the two axes that bind and loses the one with slack. The compression
lever has room to give: 9.0x still triples the proposal's budgeted 3-6x, and
compression trades against a ceiling that nothing else can buy back.

The pool-fraction column is the one that changed my mind. R4 found that the bag
is redrawn every Euler step, so a smaller pool is explored more thoroughly per
draw — and that advantage compounds exactly when step count is cut for speed.
rBRICS is 1.5x better there *and* covers more. The two effects point the same
way, which is not something the compression number could have told us.

Cost: 8.6 fragments per molecule instead of 7.1, and ~20% slower to decompose.

---

## R6 — The attachment-point variable is global (flagged from a parallel review)

Verified in code. `FragFMGenerator.sample_molecule_graph_dynamic` draws
`gen_z = torch.randn(bs, latent_z_dim)` — **one latent per molecule**, not per
fragment or per edge. `AE.decode` turns it into a single graph-level embedding
(`ae.py:205`) and predicts the inter-fragment bonds from it plus per-atom
features.

So the flow model's generative variables are: coarse fragment-graph nodes, coarse
fragment-graph edges, and one global `z`. Fragment identity and fragment-level
connectivity are per-node and per-edge, and fragment-level advantage shaping
lands on them cleanly. **Attachment-point choice has only the global `z` to land
on**, so for that failure mode the structured advantage of Method D degrades to a
scalar — which is exactly the aggregation loss the proposal criticises FRIGID for.

This matters because attachment-point errors are the isomer confusions MS/MS is
worst at: the right fragments joined the wrong way.

How much it matters is measurable and not yet measured. The decoder is
deterministic given (coarse graph, z), so if FragFM's coarse-to-fine
reconstruction is near-lossless on MassSpecGym chemistry then attachment is
nearly determined by the coarse graph and `z` carries little of the decision. The
experiment is FragFM's own `exe/eval_ae.py` run on a MassSpecGym LMDB, which also
yields a number worth having on its own:

> **ceiling on exact top-1 = P(all fragments in pool) x P(coarse-to-fine
> reconstruction correct)**

0.823 for the first factor at a 200k pool. The second is unmeasured, and it needs
the MassSpecGym LMDB (`scripts/build_msg_lmdb.py`).

If reconstruction is lossy, the options are a per-edge latent instead of a
per-molecule one, or reporting it as a limitation. That is a week-3 architecture
decision, not a week-12 discovery.

---

## R7 — E0-e: attachment is carried almost entirely by z

`scripts/e0e_attachment.py` @ `119b67d` · FragFM's released NPGen autoencoder ·
coarse graph held at ground truth, only z varied

| z source | NPGen edge acc | NPGen graph acc | MSG edge acc | MSG graph acc |
| --- | ---: | ---: | ---: | ---: |
| encoded | 0.9986 | **0.9872** | 0.9321 | **0.6462** |
| noised s=0.1 | 0.9985 | 0.9856 | 0.9314 | 0.6449 |
| noised s=0.25 | 0.9980 | 0.9788 | 0.9285 | 0.6310 |
| noised s=0.5 | 0.9911 | 0.9100 | 0.9164 | 0.5896 |
| noised s=1.0 | 0.9466 | 0.6561 | 0.8765 | 0.4456 |
| prior | 0.6750 | **0.0940** | 0.6495 | **0.0320** |

NPGen = in-distribution for this checkpoint (COCONUT natural products, BRICS).
MSG = MassSpecGym test fold, rBRICS, zero-shot.

### z is not redundant, and R6's concern is confirmed at the severe end

With the coarse graph fixed at ground truth and z drawn from the prior, graph
accuracy is **9.4%** in-distribution against 98.7% with the encoded z. So the
coarse fragment graph does **not** determine attachment: z carries essentially
the whole decision. Method D's per-fragment advantages have nothing to land on
for attachment errors, exactly as R6 feared. This is the "pile B is large" case,
not the "limitation paragraph" case.

The noised arms bound what the flow must achieve: accuracy is intact to s=0.25
(97.9%), costs 8 points at s=0.5, and 33 points at s=1.0. So z must be predicted
to well under one standard deviation in AE latent space.

**Do not read 9.4% as a forecast of generator accuracy.** The flow evolves z
during sampling — `_calc_euler_step` integrates `z_rate = (pred_z - gen_z)/(1-t)`
— so the z reaching the decoder is a learned function of the coarse-graph
trajectory, not the prior draw it started from. The prior arm measures whether z
is redundant (it is not); the noised arms measure how hard the flow's job is.
The script's earlier summary line overstated this and has been corrected.

### The MassSpecGym number is confounded, and the confound matters

Encoded-z graph accuracy is 64.6% on MassSpecGym against 98.7% on NPGen. Two
causes are entangled:

1. **Decomposition mismatch** — the checkpoint was trained on BRICS fragments and
   is being fed rBRICS ones.
2. **Chemistry mismatch** — trained on COCONUT natural products, applied to
   MassSpecGym's broader chemistry.

Running MassSpecGym through BRICS separates them, and it is cheap. Until then
64.6% cannot be quoted as anything.

It matters because this is the second factor of the ceiling:

> ceiling on exact top-1 = P(fragments in pool) x P(reconstruction correct)

= 0.823 x 0.646 = **0.53** if the zero-shot number held, which it should not:
the autoencoder will be retrained on MassSpecGym regardless, and 98.7%
in-distribution is what a retrained one should approach.

### The question that decides whether this needs an architecture change

If attachment is carried by z, is the true molecule recoverable by *searching*
over z? `--multiplicity K` decodes each ground-truth coarse graph under K
prior draws and reports how many distinct molecules come out and whether the true
one is among them.

- **High recall at modest K** — attachment is a search problem. The oracle ranks
  z-samples at test time, no architecture change, and the RL credit issue is
  confined to how efficiently the policy proposes z.
- **Low recall** — z is a bottleneck and needs discrete attachment sites (make
  "which site of fragment A bonds to fragment B" categorical, so it flows through
  the same discrete machinery and takes credit like every other variable) or a
  per-bond latent.

**Keep this separate from C3.** Some isomer pairs are indistinguishable because
their predicted spectra genuinely are near-identical — that is the verifier
ceiling, not a credit-assignment failure, and conflating them makes an
architecture fix look like papering over a fundamental limit. E0-e is clean by
construction: the oracle is never in the loop.

---

## R8 — E0-b: the speed problem is already solved, and the step lever is capped

`scripts/e0b_throughput.py` @ `6a72b32` · FragFM NPGen checkpoint, unconditional,
batch 256, 1x A100-80GB · 2 timed repeats after a warmup

| steps | flow ms/sample | decode ms/sample | total s/sample | decode share |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 278.81 | 58.83 | **0.338** | 0.17 |
| 200 | 112.14 | 55.88 | 0.168 | 0.33 |
| 100 | 56.33 | 57.16 | 0.113 | 0.50 |
| 50 | 28.57 | 56.85 | 0.085 | 0.67 |
| 20 | 11.47 | 56.58 | 0.068 | 0.83 |
| 10 | 5.61 | 52.14 | **0.058** | 0.90 |

Flow time is exactly linear in step count: 0.558 ms/step at 500, 0.561 at 10.

### The headline: C1's premise no longer holds

The proposal's scalability argument is built on DiffMS at 131.2 s/spectrum and a
100x speedup being the enabling condition for everything else. **Unmodified
FragFM at its default 500 steps already runs at 0.338 s/sample** — 388x faster
than DiffMS and 19x faster than FRIGID's 6.58 s/spectrum, before any of the
planned speedup work.

Rerunning the GRPO arithmetic that motivated the whole budget: G=16 over 10,000
spectra is 160,000 generations. At 0.338 s that is 15 GPU-hours per epoch; at 20
steps, 3.0 GPU-hours, or **23 minutes on 8 GPUs**. The proposal budgeted 44
GPU-hours post-speedup and called the pre-speedup version impossible at 30 days
per epoch. We are 15x better than its optimistic case, with the stock model.

The week-6 gate (<2 s/sample) is already met with 6x margin.

Caveat: this is unconditional generation on NPGen, not spectrum-conditioned
elucidation. Conditioning adds an encoder and cross-attention, but that cost is
per sample and small next to 500 flow steps. The order of magnitude stands.

### The step lever is capped at ~5.8x, not 25x

Decode — the coarse-to-fine autoencoder plus Blossom assembly — is a **hard floor
at 52-59 ms/sample that step count does not move**. So cutting 500 steps to 20
cuts flow time 24x but total time only **5.0x**; going to 10 steps buys 5.8x
total and nothing further is available from this lever.

This is exactly what the phase split was built to catch, and it would have been
invisible in an end-to-end number. The proposal's budget table lists ~25x for
step reduction; the true end-to-end ceiling is 5.8x, and past 50 steps the
returns are already mostly gone.

The practical consequences are the opposite of alarming:

- **Don't do the distillation work.** FS-DFM, T3D and Duo all attack flow-step
  count, which is now worth at most 5.8x and realistically ~2x from the operating
  point we would choose. ReMDM and corrector steps stay useful for *quality* at
  low step counts, not for speed.
- **If more speed is ever wanted, optimise decode**, which is 83% of the cost at
  20 steps and largely CPU (Blossom assembly runs unbatched, one molecule at a
  time, at 100-280 molecules/s).
- **Validity degrades gracefully**: 98.2% at 500 steps, 94.9% at 20, 91.6% at 10,
  so the low-step regime is usable.

### A cost the proposal does not account for

Instantiating the generator embeds the entire fragment pool up front: **4m35s for
NPGen's 133,823 fragments**. Fine once for inference. Under GRPO the fragment
embedder is part of the policy, so any update to it invalidates every embedding —
4.6 minutes per policy step is fatal. Either freeze the fragment embedder during
RL and train only the coarse GNN, or embed the drawn bag on demand instead of
precomputing the pool. Freezing is the simpler answer and is defensible: the
embedder is a structural encoder, and the policy has the coarse GNN to adapt
with. Decide before the RL loop is written.

(Note also that NPGen's own released pool is 133,823 fragments, 2.5x the 53,220
we built from 200k corpus molecules. Pool size is affordable.)

---

## R9 — The 64.6% was decomposition mismatch; and the rBRICS decision now has a number to beat

Same autoencoder (FragFM's released NPGen checkpoint), MassSpecGym test fold,
BRICS instead of rBRICS.

| z source | NPGen (BRICS) | MSG (BRICS) | MSG (rBRICS) |
| --- | ---: | ---: | ---: |
| encoded | 0.9872 | **0.9491** | 0.6462 |
| noised s=0.25 | 0.9788 | 0.9373 | 0.6310 |
| noised s=0.5 | 0.9101 | 0.8608 | 0.5896 |
| noised s=1.0 | 0.6561 | 0.6028 | 0.4456 |
| prior | 0.0940 | 0.0573 | 0.0320 |

**The confound resolves almost entirely to decomposition, not chemistry.** The
NPGen-trained autoencoder reconstructs **94.9%** of MassSpecGym molecules exactly,
zero-shot, when fed BRICS fragments — against 98.7% in its own distribution. A
3.8-point drop across COCONUT natural products to MassSpecGym's broader chemistry
is excellent transfer. The 64.6% on rBRICS was the autoencoder being fed a
decomposition it was never trained on.

### The rBRICS decision now has a decidable criterion

Both ceiling factors multiply, and they pull in opposite directions:

| decomposition | pool coverage | reconstruction | ceiling |
| --- | ---: | ---: | ---: |
| BRICS | 0.725 | 0.949 (measured, zero-shot) | **0.688** |
| rBRICS | 0.823 | unknown, needs a retrained AE | 0.823 x p |

**rBRICS wins iff a retrained autoencoder reaches p > 0.836 on rBRICS
fragments.** That is not a foregone conclusion: rBRICS cuts more finely, so
molecules carry more fragments (8.6 vs 7.1) and more junctions, and attachment
prediction is the harder half of the job. Retraining the autoencoder on each
decomposition settles it, and the autoencoder gets retrained regardless.

Both numbers are also floors — 0.725 and 0.823 are pool coverage at 200k corpus
molecules, and the corpus holds 4M.

### My prior arm is confounded, and `shuffled` is the control that fixes it

The z-search run is the reason to distrust it. Over K=32 prior draws per coarse
graph: **3.0 distinct molecules on average** (max 9), and the true molecule
recovered **13.3%** of the time — barely above the 9.4% a single draw achieves.
Thirty-two draws behaving like one means the draws are not exploring; the decoder
is collapsing most of the prior mass onto a handful of outputs.

That is the signature of an out-of-distribution input, not of an informative
latent. And there is a mechanism: `store_smis_from_coarse_graph` maps z by
`(z+1) * 0.5 * (max-min) + min`, which treats a standard normal as if it were
uniform on [-1,1]. About 32% of a normal's mass lies outside [-1,1], so prior
draws land beyond the encoded range, where the decoder saturates.

So the prior arm may be measuring decoder saturation rather than the information
z carries, and the 9.4% cannot be read as "the coarse graph determines almost
nothing" until that is excluded.

**The control is `shuffled`: give the decoder another molecule's *encoded* z.**
In-distribution by construction, and molecule-specific information is destroyed.

- shuffled ~ encoded -> z carries little molecule-specific attachment
  information, the prior result was an artifact, and R7's alarm is withdrawn.
- shuffled ~ prior -> z genuinely carries the attachment decision, R7 stands, and
  the discrete-attachment redesign is on.

Added, along with a print of the encoded vs prior latent ranges so the
saturation hypothesis is visible rather than inferred. **R7's conclusion is
provisional until this runs.**

---

## R10 — The control run: R7 stands, magnitude corrected

| z source | NPGen (BRICS) | MSG (BRICS) |
| --- | ---: | ---: |
| encoded | 0.9872 | 0.9491 |
| noised s=0.5 | 0.9102 | 0.8608 |
| noised s=1.0 | 0.6560 | 0.6028 |
| **shuffled** (another molecule's encoded z) | **0.1945** | **0.1519** |
| prior | 0.0940 | 0.0573 |

Latent scale, NPGen: encoded std 0.869, range [-3.72, 4.01]; prior draws std
3.800, range [-12.18, 13.60].

**The out-of-distribution hypothesis was right, and it does not change the
conclusion.** Prior draws really are ~4.4x too wide, so the 9.4% figure was
partly decoder saturation. The correct in-distribution control gives **19.5%**.
Handing the decoder a perfectly valid latent belonging to a different molecule
still collapses exact reconstruction from 98.7% to 19.5%.

So z is molecule-specific and carries the attachment decision. **R7 stands; only
its magnitude was wrong** (19.5%, not 9.4%).

A cleaner way to read 19.5%: for about one molecule in five, attachment is
determined by the coarse fragment graph alone and any latent will do. For the
other four in five, z carries real information — and z is a single global
continuous vector with no per-fragment structure for an advantage to land on.

That is the architecture case, not the limitation-paragraph case, and it arrived
in week 1.

### Two corrections this forces on earlier results

**The z-search number in R9 is also confounded** and should not be quoted. It
sampled the same too-wide prior, so its 3.0 distinct molecules and 13.3% recall
measure saturation, not the reachable attachment set. A meaningful z-search
perturbs around an in-distribution latent.

**`draw_z` models the wrong thing.** The min-max inverse is applied in
`store_smis_from_coarse_graph` to the *flow's output* at t=1, which the flow
learns to keep within [-1,1] because that is what the transform normalised
training latents to. Applying the same inverse to a standard normal models the
t=0 initialisation, not what the decoder is ever handed. The prior arm is
retained as a diagnostic and is no longer the basis of any claim.

---

## R11 — Blossom decode, and the granularity attachment is categorical at

### The decode rule was wrong in R7 and R10

FragFM's generation path does not threshold attachment scores. It expands each
junction atom into one slot per open valence, runs Blossom max-weight matching
over the slots, and contracts back
(`genererate_utils.realize_single_fine_graph_dict`). E0-e used `pred > 0.5`, so
R7 and R10 describe the scoring head rather than the model that ships. Matching
depends only on the *ordering* of scores, a threshold on their absolute values,
so the two can diverge sharply.

Paired comparison, same 1,000 MassSpecGym test molecules, BRICS, same checkpoint:

| z source | threshold | **Blossom** | gain |
| --- | ---: | ---: | ---: |
| encoded | 0.9500 | **0.9840** | +3.4 |
| noised s=0.5 | 0.9080 | 0.9630 | +5.5 |
| noised s=1.0 | 0.7310 | 0.8390 | +10.8 |
| **shuffled** | 0.2040 | **0.2570** | +5.3 |
| prior | 0.0660 | 0.2380 | +17.2 |

Blossom helps everywhere and helps most where the scores are worst, exactly as
the mechanism predicts — a global matching is robust to score shifts that destroy
a fixed threshold.

**It does not help enough.** Through the real decode, a wrong-but-valid latent
still costs 73 points of exact reconstruction, 0.984 to 0.257. **R7 and R10's
conclusion survives; their numbers do not.** The correct figure is 25.7%, not
19.5%, and the correct ceiling factor for BRICS is **0.984**, not 0.949 — so the
BRICS ceiling is 0.725 x 0.984 = **0.713**.

### Attachment is categorical per atom, not per coarse edge

| granularity | valid? | why |
| --- | --- | --- |
| per coarse edge | **no** | rBRICS cuts rings and a cut ring joins its fragments twice: 7.67% of rBRICS coarse edges carry two bonds, against 0.00% of BRICS ones |
| per slot | yes, but degenerate | slots on the same atom are interchangeable, so the categorical is unidentifiable under relabelling |
| **per atom, choose junction_count** | **yes** | by construction, in any decomposition, with no degeneracy |

Junction counts (MassSpecGym, BRICS): 82.3% of junction atoms have count 1,
17.2% count 2, 0.45% count 3 or more. So choose-one covers most atoms but not
enough of them to treat choose-k as a special case.

Candidate-set sizes per coarse edge: 21.2% have a single candidate and are free;
mean 2.80. A uniform guess scores 0.50 per edge, which is the floor any learned
head has to beat.

### Design

Per-atom softmax over its candidate partners, cross-entropy against its true
ones, Blossom unchanged at decode. Local training, global decoding.

The softmax with k targets is deliberate rather than a compromise: its optimum
puts 1/k on each true partner, ranking all k above every distractor, which is
precisely what max-weight matching consumes. BCE cannot express that, because it
scores each candidate without reference to its competitors.

No architecture change — the network already emits one logit per candidate.
