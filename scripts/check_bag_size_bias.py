"""Is the 7-atom shortfall the bag's fault or the model's?

R18: generated molecules average 20.9 heavy atoms against a target of 27.9, one
atom short in every one of ~7 slots.  Two candidate causes, and they need
different fixes:

* **the bag** offers small fragments, so the model picks small ones;
* **the model** prefers small fragments even when larger ones are on offer.

There is a clean reference value.  Occurrence counts how many molecules contain
a fragment, so summing size x occurrence over the pool and dividing by total
occurrence gives exactly (total heavy atoms) / (total fragments) -- the true mean
fragment size, 3.94.  A draw that reproduced the occurrence distribution would
therefore be unbiased by construction.

But the draw is `replace=False`: it takes 384 *distinct* fragments.  The
highest-occurrence fragments -- which are the small common ones, benzene, methyl,
carbonyl -- are then included with near-certainty, and everything else is a thin
weighted sample of a long tail.  The *set* is not distributed like the
occurrences that generated it, and that is what the model sees.

So measure the mean size of an actual drawn bag against 3.94.  Below it, the bag
is the cause and the fix belongs in the draw.  At it, the model is the cause and
the fix belongs in conditioning or in a projection.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
RDLogger.DisableLog("rdApp.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--n-base-frag", type=int, default=384)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    import lmdb

    stem = Path(args.data).stem
    fenv = lmdb.open(f"data/processed/{stem}_fragment.lmdb", readonly=True,
                     lock=False, readahead=True, meminit=False, map_size=int(1e12))
    n = int(fenv.stat()["entries"])
    occ = np.zeros(n)
    size = np.zeros(n)
    with fenv.begin() as txn:
        for _, v in txn.cursor():
            r = pickle.loads(v)
            i = int(r["key"].split("_")[1])
            occ[i] = r["train_occurance"]
            m = Chem.MolFromSmiles(r["smi"])
            size[i] = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1) if m else 0

    live = occ > 0
    w = occ[live] / occ[live].sum()
    occ_weighted = float((w * size[live]).sum())
    unweighted = float(size[live].mean())

    print(f"pool {n:,} fragments, {int(live.sum()):,} drawable\n")
    print(f"  occurrence-weighted mean fragment size : {occ_weighted:.2f} heavy atoms")
    print(f"    (this is total heavy atoms / total fragments, the true value)")
    print(f"  unweighted mean over distinct fragments : {unweighted:.2f}")

    rng = np.random.default_rng(args.seed)
    idx = np.flatnonzero(live)
    p = w
    means = []
    for _ in range(args.draws):
        drawn = rng.choice(idx, size=args.n_base_frag, replace=False, p=p)
        means.append(size[drawn].mean())
    bag = float(np.mean(means))

    print(f"\n  mean size of a drawn bag ({args.n_base_frag} distinct, "
          f"{args.draws} draws): {bag:.2f}")
    print(f"  deviation from the true 3.94: {bag - occ_weighted:+.2f} heavy atoms")

    print("\n--- reading ---")
    if bag < occ_weighted - 0.3:
        print("  The bag is biased small. Sampling without replacement turns an")
        print("  unbiased occurrence distribution into a biased *set*, and the fix")
        print("  belongs in the draw.")
    elif bag > occ_weighted + 0.3:
        print("  The bag is biased large, which does not explain the shortfall.")
    else:
        print("  The bag is unbiased, so the model prefers small fragments among")
        print("  those offered. The fix belongs in conditioning or in a projection,")
        print("  not in the draw.")
    # The bag mean is not the target; the true mean is.  Comparing generated
    # against the bag would overstate the model's error by whatever the bag's
    # own skew happens to be.
    print(f"\n  generated 2.95 atoms/fragment (R18) against a true {occ_weighted:.2f}:")
    print(f"    model selects {occ_weighted - 2.95:+.2f} atoms/fragment below truth, "
          f"{(occ_weighted - 2.95) / occ_weighted:.0%} too small")
    print(f"    over 7.09 slots that is {(occ_weighted - 2.95) * 7.09:.2f} atoms, "
          f"against a measured shortfall of 6.97")


if __name__ == "__main__":
    main()
