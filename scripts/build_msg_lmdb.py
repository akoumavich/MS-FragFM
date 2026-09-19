"""Decompose MassSpecGym into FragFM's LMDB format.

Emits keys prefixed train_/valid_/test_ from MassSpecGym's own folds, which is
what FragFM's downstream `process_fragment_from_lmdb.py` splits on, so their
fragment-bag builder runs unchanged afterwards.

Structures only: spectra are joined at training time by InChIKey, since one
structure carries many spectra and the fragment decomposition is per structure.

`data_type` selects FragFM's canonicalisation branch, not a dataset -- "guacamol"
is its relaxed branch, which E0-a showed recovers 2.6% of MassSpecGym structures
that the strict branch drops (charged species).
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import lmdb
import pandas as pd
from rdkit import RDLogger

from msfragfm.paths import MSG_TSV

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
FOLD_TO_PREFIX = {"train": "train", "val": "valid", "test": "test"}

_CFG = {}


def _init(decomp, canon):
    _CFG.update(decomp=decomp, canon=canon)
    RDLogger.DisableLog("rdApp.*")


def _one(item):
    from fragfm.process import process_sample

    key, smi, fold = item
    sample = {
        "smi": smi,
        "key": key,
        "fold": fold,
        "data_type": _CFG["canon"],
        "decomp_method": _CFG["decomp"],
    }
    try:
        sample.update(process_sample(dict(sample)))
        return sample
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", default="rbrics", choices=["brics", "rbrics"])
    ap.add_argument("--canon", default="guacamol", help="FragFM canonicalisation branch")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="debug: cap structures")
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))

    df = pd.read_csv(MSG_TSV, sep="\t", usecols=["smiles", "fold"]).drop_duplicates("smiles")
    if args.limit:
        df = df.head(args.limit)
    items, counts = [], {}
    for fold, grp in df.groupby("fold"):
        prefix = FOLD_TO_PREFIX[fold]
        counts[prefix] = len(grp)
        items += [(f"{prefix}_{i}", s, fold) for i, s in enumerate(grp["smiles"])]
    print(f"{len(items):,} structures {counts}")

    t0 = time.perf_counter()
    with mp.Pool(args.procs, initializer=_init, initargs=(args.decomp, args.canon)) as pool:
        recs = pool.map(_one, items, chunksize=32)
    ok = [r for r in recs if r is not None]
    print(f"decomposed {len(ok):,}/{len(items):,} in {time.perf_counter() - t0:.0f}s")

    out = FRAGFM / "data" / "processed" / f"msg_{args.decomp}_{args.tag}.lmdb"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=int(1e12))
    with env.begin(write=True) as txn:
        for r in ok:
            txn.put(str(r["key"]).encode(), pickle.dumps(r))
    env.close()
    print(f"wrote {out}")
    print(
        f"\nnext, from the FragFM checkout:\n"
        f"  cd {FRAGFM} && python process/process_fragment_from_lmdb.py "
        f"msg {args.decomp} {args.tag}"
    )


if __name__ == "__main__":
    main()
