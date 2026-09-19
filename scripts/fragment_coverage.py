"""E0-a: does FragFM's fragment decomposition fit MassSpecGym chemistry?

Two modes.

``--mode grid`` compares brics/rbrics x strict/relaxed canonicalisation on
decomposition success rate, fragment count, edge-slot reduction (the proposal's
3-6x lever) and preprocessing throughput.

``--mode holdout`` answers the vocabulary question properly: fit the fragment
vocabulary on the train fold, measure coverage on the test fold.  Fitting and
evaluating on the same molecules makes top-V trivially reach 1.0 once V passes
the vocabulary size, which measures nothing.  The folds are MCES>=10 disjoint,
so held-out here means genuinely distant chemistry.

Writes results/fragment_coverage_<mode>_<tag>.json.
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
        return {"ok": False, "err": type(exc).__name__}
    return {
        "ok": True,
        "n_heavy": Chem.MolFromSmiles(smi).GetNumHeavyAtoms(),
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
    # Edge slots are what the generator actually pays for.  Aggregate, not mean
    # of ratios: the latter is dominated by small molecules and reads ~2x high.
    atom_slots = n_heavy * (n_heavy - 1) / 2
    frag_slots = np.maximum(n_frag * (n_frag - 1) / 2, 1)

    return {
        "decomp": decomp,
        "canon": canon,
        "n_input": len(smis),
        "success_rate": len(ok) / len(recs),
        "errors": dict(errs.most_common(5)),
        "mols_per_sec_per_core": len(smis) / wall / procs,
        "n_heavy_mean": float(n_heavy.mean()),
        "n_frag_mean": float(n_frag.mean()),
        "n_frag_p95": float(np.percentile(n_frag, 95)),
        "n_frag_max": int(n_frag.max()),
        "single_fragment_frac": float((n_frag == 1).mean()),
        "edge_slot_reduction_agg": float(atom_slots.sum() / frag_slots.sum()),
        "vocab_size": len(set(f for r in ok for f in r["frags"])),
        "_vocab": Counter(f for r in ok for f in r["frags"]),
        "_frag_sets": [set(r["frags"]) for r in ok],
    }


def coverage_curve(fit_vocab, eval_frag_sets, grid):
    ranked = [f for f, _ in fit_vocab.most_common()]
    return {
        v: float(np.mean([s <= set(ranked[:v]) for s in eval_frag_sets])) for v in grid
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grid", choices=["grid", "holdout"])
    ap.add_argument("--n", type=int, default=5000, help="structures per split")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="msg1.5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(MSG_TSV, sep="\t", usecols=["smiles", "fold"]).drop_duplicates("smiles")
    print(f"{len(df):,} unique structures")

    def pick(frame, n):
        f = frame.sample(n, random_state=args.seed) if n and n < len(frame) else frame
        return f["smiles"].tolist()

    if args.mode == "grid":
        smis = pick(df, args.n)
        out = []
        for decomp in ("brics", "rbrics"):
            for canon in ("npgen", "guacamol"):  # strict vs relaxed
                r = run(smis, decomp, canon, args.procs)
                r.pop("_vocab", None)
                r.pop("_frag_sets", None)
                out.append(r)
                print(
                    f"{decomp:7s} {canon:9s} ok={r['success_rate']:.3f} "
                    f"n_frag={r.get('n_frag_mean', 0):.1f} "
                    f"slot_red={r.get('edge_slot_reduction_agg', 0):.1f}x "
                    f"vocab={r.get('vocab_size', 0)} "
                    f"{r.get('mols_per_sec_per_core', 0):.1f} mol/s/core"
                )
    else:
        fit = run(pick(df[df.fold == "train"], args.n), "brics", "guacamol", args.procs)
        ev = run(pick(df[df.fold == "test"], args.n), "brics", "guacamol", args.procs)
        grid = (100, 250, 500, 1000, 2000, 5000, fit["vocab_size"])
        in_sample = coverage_curve(fit["_vocab"], fit["_frag_sets"], grid)
        held_out = coverage_curve(fit["_vocab"], ev["_frag_sets"], grid)
        known = set(fit["_vocab"])
        unseen = float(np.mean([len(s - known) / len(s) for s in ev["_frag_sets"]]))
        never = float(np.mean([bool(s - known) for s in ev["_frag_sets"]]))

        print(f"\ntrain vocab={fit['vocab_size']}  test vocab={ev['vocab_size']}")
        print(f"test molecules containing >=1 fragment absent from train: {never:.3f}")
        print(f"mean share of a test molecule's fragments unseen in train: {unseen:.3f}\n")
        print(f"{'top-V':>8} {'in-sample':>10} {'held-out':>10}")
        for v in grid:
            print(f"{v:>8} {in_sample[v]:>10.3f} {held_out[v]:>10.3f}")

        for r in (fit, ev):
            r.pop("_vocab")
            r.pop("_frag_sets")
        out = [
            {"split": "train", **fit},
            {"split": "test", **ev},
            {
                "frac_mols_fully_covered_in_sample": in_sample,
                "frac_mols_fully_covered_held_out": held_out,
                "frac_test_mols_with_unseen_fragment": never,
                "mean_frac_unseen_fragments_per_test_mol": unseen,
            },
        ]

    path = RESULTS / f"fragment_coverage_{args.mode}_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
