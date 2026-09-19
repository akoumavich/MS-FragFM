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
| `torch.compile` (inductor) | **FAIL** — `has_triton()` false; under diagnosis |
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
