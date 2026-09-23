"""Can the sampler even reach the right answer?

E3's first evaluation returned top-1 = 0 with only 0.4% of candidates carrying
the target formula.  Before tuning anything, establish whether the target was
reachable at all: the generator assigns a fragment only from the bag drawn at
that Euler step, and R4 measured that bag as 384 occurrence-weighted draws from
an 81,739-fragment pool.

Three numbers, in increasing strictness:

* `in_pool` -- are the molecule's fragments in the pool at all?  This is R5's
  coverage, recomputed here on the exact artefact the sampler loads.
* `in_one_draw` -- expected share present in a single step's bag.  An upper
  bound on what the model can assign at any one step.
* `all_in_traj` -- share of molecules for which *every* fragment appears in at
  least one bag across the trajectory.  A generous upper bound on exact top-1,
  since it ignores whether the fragment was available at the step it was needed.

If `all_in_traj` is near zero, no amount of training or sampling temperature
fixes the result, and the spectrum-filtered bag of Method A stops being an
optimisation and becomes a prerequisite.
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--n-mols", type=int, default=300)
    ap.add_argument("--n-base-frag", type=int, default=384)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--fold-prefix", default="test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    import lmdb

    stem = Path(args.data).stem
    frag_idx = pickle.loads(
        Path(f"data/processed/{stem}_fragment_to_idx.pkl").read_bytes())

    # Occurrence weights, exactly as random_select_frags_by_occurance uses them.
    fenv = lmdb.open(f"data/processed/{stem}_fragment.lmdb", readonly=True,
                     lock=False, readahead=True, meminit=False, map_size=int(1e12))
    n_all = int(fenv.stat()["entries"])
    occ = np.zeros(n_all)
    with fenv.begin() as txn:
        for k, v in txn.cursor():
            r = pickle.loads(v)
            occ[int(r["key"].split("_")[1])] = r["train_occurance"]
    p = occ / occ.sum()
    drawable = int((occ > 0).sum())
    print(f"pool {n_all:,} fragments, {drawable:,} drawable, "
          f"{args.n_base_frag} drawn per step, {args.steps} steps")

    env = lmdb.open(args.data, readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    mols = []
    with env.begin() as txn:
        for key, val in txn.cursor():
            if key.decode().startswith(args.fold_prefix):
                mols.append(pickle.loads(val)["frag_smi_list"])
    random.Random(args.seed).shuffle(mols)
    mols = mols[:args.n_mols]

    rng = np.random.default_rng(args.seed)
    in_pool, one_draw, all_traj, n_frag = [], [], [], []
    for frags in mols:
        ids = [frag_idx.get(f) for f in frags]
        n_frag.append(len(ids))
        in_pool.append(all(i is not None and occ[i] > 0 for i in ids))
        known = [i for i in ids if i is not None and occ[i] > 0]
        if not known:
            one_draw.append(0.0)
            all_traj.append(False)
            continue
        # Per-fragment inclusion probability in one occurrence-weighted draw
        # without replacement, approximated by 1 - (1 - p_i)^n, which is the
        # standard approximation and tight at n << pool.
        pin = 1 - (1 - p[known]) ** args.n_base_frag
        one_draw.append(float(np.mean(pin)))
        traj = 1 - (1 - pin) ** args.steps
        all_traj.append(float(np.prod(traj)))

    print(f"\n{len(mols)} {args.fold_prefix} molecules, "
          f"mean {np.mean(n_frag):.1f} fragments each\n")
    print(f"  in_pool      {np.mean(in_pool):.3f}   "
          f"(every fragment present in the loaded pool)")
    print(f"  in_one_draw  {np.mean(one_draw):.3f}   "
          f"(mean per-fragment chance of being offered at a given step)")
    print(f"  all_in_traj  {np.mean(all_traj):.3f}   "
          f"(all fragments offered at some point -- generous upper bound on top-1)")
    frac_hopeless = float(np.mean(np.array(all_traj) < 0.01))
    print(f"\n  {frac_hopeless:.1%} of molecules have below 1% chance of their "
          f"fragment set ever being offered")


if __name__ == "__main__":
    main()
