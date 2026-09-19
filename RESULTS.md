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
| `torch.compile` (inductor) | fixed — `triton` imports `setuptools`, absent from a uv venv |
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
