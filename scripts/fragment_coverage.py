"""E0-a: does FragFM's fragment decomposition actually fit MassSpecGym chemistry?

Three questions, one pass:

1. **Coverage** -- what fraction of MassSpecGym structures survive
   ``fragfm.process.process_sample``?  A low number invalidates FragFM as the base.
2. **The compression lever** -- the proposal budgets 3-6x from "23 atoms -> ~6
   fragments, 253 edge slots -> ~15".  This measures it on the real distribution.
3. **Vocabulary** -- how many distinct fragments, and what share of molecules is
   covered by the top-V.  Risk row "fragment vocabulary misses natural-product
   chemistry" in the proposal.

Writes results/fragment_coverage_<tag>.json.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from msfragfm.paths import MSG_TSV, RESULTS

RDLogger.DisableLog("rdApp.*")

_CFG = {}


def _init(decomp, canon):
    _CFG["decomp"] = decomp
    _CFG["canon"] = canon
    RDLogger.DisableLog("rdApp.*")


def _one(smi):
    from fragfm.process import process_sample

    try:
        out = process_sample(
            {"smi": smi, "data_type": _CFG["canon"], "decomp_method": _CFG["decomp"]}
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": type(exc).__name__, "smi": smi}
    n_heavy = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
    return {
        "ok": True,
        "n_heavy": n_heavy,
        "n_frag": int(out["n_frag"]),
        "frags": out["frag_smi_list"],
    }


def run(smis, decomp, canon, procs):
    t0 = time.perf_counter()
    with mp.Pool(procs, initializer=_init, initargs=(decomp, canon)) as pool:
        recs = pool.map(_one, smis, chunksize=16)
    wall = time.perf_counter() - t0

    ok = [r for r in recs if r["ok"]]
    errs = Counter(r["err"] for r in recs if not r["ok"])
    if not ok:
        return {"decomp": decomp, "canon": canon, "success_rate": 0.0, "errors": dict(errs)}

    n_heavy = np.array([r["n_heavy"] for r in ok])
    n_frag = np.array([r["n_frag"] for r in ok])
    # Edge slots are what the generator actually pays for.
    atom_slots = n_heavy * (n_heavy - 1) / 2
    frag_slots = np.maximum(n_frag * (n_frag - 1) / 2, 1)

    vocab = Counter(f for r in ok for f in r["frags"])
    ranked = [f for f, _ in vocab.most_common()]
    frag_sets = [set(r["frags"]) for r in ok]
    cov = {}
    for v in (100, 500, 1000, 5000, 10000):
        top = set(ranked[:v])
        cov[v] = float(np.mean([s <= top for s in frag_sets]))

    return {
        "decomp": decomp,
        "canon": canon,
        "n_input": len(smis),
        "success_rate": len(ok) / len(recs),
        "errors": dict(errs.most_common(5)),
        "mols_per_sec_per_core": len(smis) / wall / procs,
        "wall_s": wall,
        "n_heavy_mean": float(n_heavy.mean()),
        "n_frag_mean": float(n_frag.mean()),
        "n_frag_p95": float(np.percentile(n_frag, 95)),
        "n_frag_max": int(n_frag.max()),
        "single_fragment_frac": float((n_frag == 1).mean()),
        "edge_slot_reduction_mean": float((atom_slots / frag_slots).mean()),
        "edge_slot_reduction_agg": float(atom_slots.sum() / frag_slots.sum()),
        "vocab_size": len(vocab),
        "frac_mols_fully_in_top_v": cov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="unique structures to sample")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="msg1.5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(MSG_TSV, sep="\t", usecols=["smiles", "fold"])
    smis = df.drop_duplicates("smiles")["smiles"]
    print(f"{len(df):,} spectra -> {len(smis):,} unique structures")
    if args.n and args.n < len(smis):
        smis = smis.sample(args.n, random_state=args.seed)
    smis = smis.tolist()

    out = []
    for decomp in ("brics", "rbrics"):
        for canon in ("npgen", "guacamol"):  # strict vs relaxed canonicalisation
            r = run(smis, decomp, canon, args.procs)
            out.append(r)
            print(
                f"{decomp:7s} {canon:9s} ok={r['success_rate']:.3f} "
                f"n_frag={r.get('n_frag_mean', 0):.1f} "
                f"slot_red={r.get('edge_slot_reduction_agg', 0):.1f}x "
                f"vocab={r.get('vocab_size', 0)} "
                f"{r.get('mols_per_sec_per_core', 0):.1f} mol/s/core"
            )

    path = RESULTS / f"fragment_coverage_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
