"""Does peak explanation separate the true structure from a decoy?

Method C's conservation law is exact, but exactness is not the same as
discriminative power: a score every candidate satisfies is worthless as a signal
however sound its physics. So measure the separation before building anything on
it.

Three things:

* **enumeration cost** -- how many connected subtrees a real coarse tree has, to
  confirm the exact computation is as cheap as the tree property implies;
* **the true molecule's explained fraction** against its own spectrum;
* **a decoy's explained fraction** -- the same spectrum scored against a
  different molecule of similar size.

The gap between the last two is the whole value of the signal. If it is large,
peak explanation reranks candidates with no oracle and no training, and it says
how big the fragments should be -- which is exactly what R18 found missing.
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msfragfm.paths import MSG_TSV
from msfragfm.peak_explain import explained_fraction

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def _floats(s):
    return np.fromstring(str(s).strip("[]"), sep=",", dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--max-cuts", type=int, default=3,
                    help="MAGMa and ICEBERG model 1-3 bond cleavages")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    import lmdb

    env = lmdb.open(args.data, readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    by_smi = {}
    with env.begin() as txn:
        for _, val in txn.cursor():
            s = pickle.loads(val)
            by_smi[s["smi"]] = s

    df = pd.read_csv(MSG_TSV, sep="\t")
    df = df[(df.fold == args.fold) & (df.smiles.isin(by_smi))]
    df = df.sample(min(args.n, len(df)), random_state=args.seed)
    print(f"{len(df)} {args.fold} spectra, {args.ppm:.0f} ppm tolerance\n")

    pool = list(by_smi.values())
    rng = random.Random(args.seed)
    true_s, decoy_s, n_subs, n_frags = [], [], [], []
    for _, row in df.iterrows():
        mz, inten = _floats(row.mzs), _floats(row.intensities)
        if mz.size == 0:
            continue
        smp = by_smi[row.smiles]
        t, ns = explained_fraction(smp, mz, inten, row.adduct, args.ppm,
                                  max_cuts=args.max_cuts)
        true_s.append(t)
        n_subs.append(ns)
        n_frags.append(int(smp["n_frag"]))

        # Decoy: a different molecule of similar size, so the comparison is not
        # just "big molecules explain more peaks".
        target = int(smp["n_frag"])
        for _ in range(20):
            d = pool[rng.randrange(len(pool))]
            if d["smi"] != row.smiles and abs(int(d["n_frag"]) - target) <= 1:
                decoy_s.append(explained_fraction(
                    d, mz, inten, row.adduct, args.ppm,
                    max_cuts=args.max_cuts)[0])
                break

    t, d = np.array(true_s), np.array(decoy_s)
    print(f"connected subtrees per molecule: mean {np.mean(n_subs):.1f} "
          f"max {max(n_subs)} (over {np.mean(n_frags):.1f} fragments)")
    print(f"  2^k would be {2 ** np.mean(n_frags):.0f}; the tree structure cuts it\n")
    print(f"explained peak intensity, true structure : {t.mean():.4f} "
          f"(median {np.median(t):.4f})")
    print(f"explained peak intensity, size-matched decoy: {d.mean():.4f} "
          f"(median {np.median(d):.4f})")
    print(f"  separation {t.mean() - d.mean():+.4f}")
    if len(d):
        n = min(len(t), len(d))
        print(f"  true beats decoy on {np.mean(t[:n] > d[:n]):.1%} of spectra")
        pooled = np.sqrt((t.var() + d.var()) / 2)
        print(f"  Cohen's d {(t.mean() - d.mean()) / max(pooled, 1e-9):.2f}")


if __name__ == "__main__":
    main()
