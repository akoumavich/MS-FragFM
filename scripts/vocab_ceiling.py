"""E0-c: the representational ceiling of a fragment-level generator.

E0-a's held-out run found that 71.4% of MassSpecGym test molecules contain at
least one BRICS fragment that never appears in the train fold.  A generator that
assembles molecules out of a fragment vocabulary cannot emit those structures
exactly, whatever it learns -- so the complete train-fold vocabulary caps exact
top-1 at 28.6%.  Against a field at ~18%, that is not a comfortable margin.

The question this answers: does the ceiling lift with vocabulary scale, and how
fast?  Fragments are shared far more widely than molecules are, so a vocabulary
harvested from a large structure corpus should cover test chemistry much better
than one harvested from 5k training molecules.  If it does, the fix is cheap and
CPU-only, and it is a concrete argument for the pretraining half of the plan.

Also compares decompositions.  BRICS was chosen in E0-a on compression grounds
(14.4x vs 9.0x) before this ceiling was known; finer fragments recur more, so
rBRICS may buy ceiling back at the cost of compression.  That trade has to be
measured, not assumed.

Processes the largest corpus slice once and reads nested prefixes off it, so the
whole scaling curve costs one pass.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import RDLogger

from msfragfm.paths import MSG_PRETRAIN_MCES2, MSG_TSV, RESULTS

RDLogger.DisableLog("rdApp.*")

_CFG = {}


def _init(decomp):
    _CFG["decomp"] = decomp
    RDLogger.DisableLog("rdApp.*")


def _frags(smi):
    from fragfm.process import process_sample

    try:
        out = process_sample(
            {"smi": smi, "data_type": "guacamol", "decomp_method": _CFG["decomp"]}
        )
        return list(out["frag_smi_list"])
    except Exception:  # noqa: BLE001
        return None


def decompose(smis, decomp, procs, label):
    t0 = time.perf_counter()
    with mp.Pool(procs, initializer=_init, initargs=(decomp,)) as pool:
        recs = pool.map(_frags, smis, chunksize=64)
    ok = [r for r in recs if r is not None]
    print(
        f"  {label:28s} {len(ok):>7,}/{len(smis):,} ok  "
        f"{time.perf_counter() - t0:6.1f}s",
        flush=True,
    )
    return ok


def load_pretrain_smiles(n, seed):
    df = pd.read_csv(MSG_PRETRAIN_MCES2, sep="\t")
    col = "smiles" if "smiles" in df.columns else df.columns[0]
    if n < len(df):
        df = df.sample(n, random_state=seed)
    return df[col].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", nargs="+", default=["brics", "rbrics"])
    ap.add_argument("--corpus-n", type=int, default=200_000,
                    help="molecules drawn from the MCES-2-disjoint pretraining corpus")
    ap.add_argument("--test-n", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    msg = pd.read_csv(MSG_TSV, sep="\t", usecols=["smiles", "fold"]).drop_duplicates("smiles")
    train_smis = msg[msg.fold == "train"]["smiles"].tolist()
    test_df = msg[msg.fold == "test"]
    test_smis = test_df.sample(min(args.test_n, len(test_df)), random_state=args.seed)[
        "smiles"
    ].tolist()
    corpus_smis = load_pretrain_smiles(args.corpus_n, args.seed)
    print(
        f"train fold {len(train_smis):,} · test sample {len(test_smis):,} · "
        f"MCES-2-disjoint corpus {len(corpus_smis):,}"
    )

    sizes = [s for s in (5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)
             if s <= len(corpus_smis)]

    out = []
    for decomp in args.decomp:
        print(f"\n{decomp}:")
        test_sets = [set(f) for f in decompose(test_smis, decomp, args.procs, "test")]
        train_frags = decompose(train_smis, decomp, args.procs, "train fold (full)")
        corpus_frags = decompose(
            corpus_smis, decomp, args.procs, f"corpus ({len(corpus_smis):,})"
        )

        def ceiling(vocab):
            return float(np.mean([s <= vocab for s in test_sets]))

        train_vocab = set(f for fs in train_frags for f in fs)
        rows = [{
            "decomp": decomp, "source": "msg_train_fold",
            "n_molecules": len(train_frags), "vocab_size": len(train_vocab),
            "ceiling": ceiling(train_vocab),
        }]
        print(f"  {'source':28s} {'n_mol':>9} {'vocab':>8} {'ceiling':>8}")
        print(f"  {'msg train fold':28s} {len(train_frags):>9,} "
              f"{len(train_vocab):>8,} {rows[0]['ceiling']:>8.3f}")

        # Nested prefixes of one pass, plus the union with the train vocabulary
        # (which is what a real system would actually hold).
        for n in sizes:
            vocab = set(f for fs in corpus_frags[:n] for f in fs)
            for label, v in (("corpus", vocab), ("corpus+train", vocab | train_vocab)):
                c = ceiling(v)
                rows.append({
                    "decomp": decomp, "source": label, "n_molecules": n,
                    "vocab_size": len(v), "ceiling": c,
                })
                print(f"  {label + f' {n:,}':28s} {n:>9,} {len(v):>8,} {c:>8.3f}")
        out.extend(rows)

    path = RESULTS / "vocab_ceiling.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
