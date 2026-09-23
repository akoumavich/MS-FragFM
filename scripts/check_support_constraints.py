"""How much headroom is there in each exact constraint of section 9?

Section 9 names four constraints that belong in the support -- formula, valence,
connectivity, degree of unsaturation -- and R16 found we were enforcing none of
them.  The formula mask is now in.  This measures what the other three are worth
before any of them is built, because "exact and free" is an argument for
enforcing a constraint, not evidence that it binds.

The one worth checking first is structural.  BRICS cuts acyclic single bonds;
cutting a ring would take two cuts, which is why 7.7% of *rBRICS* coarse edges
carry two bonds and 0% of BRICS ones do (R11).  If BRICS coarse graphs are
therefore always trees, then |E_coarse| = k - 1 exactly, and the generator is
currently choosing freely among 2^(k(k-1)/2) edge configurations when the answer
must be one of the spanning trees.  That would be a far larger prune than the
formula mask.

Also decomposes the degree of unsaturation.  DBE is a property of the whole
molecule, but at fragment level it splits: each fragment carries a fixed internal
DBE, and the coarse graph contributes its own cycle count.  If the coarse graph
is a tree that second term is zero, and the fragments' internal DBEs must sum to
the molecule's -- a hard linear constraint on the fragment multiset, checkable
against the formula before a single edge is drawn.
"""

import argparse
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
RDLogger.DisableLog("rdApp.*")


def dbe_from_formula(smi):
    """DBE = #C - #H/2 - #X/2 + #N/2 + 1, halogens X."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    c = Counter(a.GetSymbol() for a in mol.GetAtoms())
    x = sum(c[e] for e in ("F", "Cl", "Br", "I"))
    return c["C"] - c["H"] / 2 - x / 2 + c["N"] / 2 + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--n-mols", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    import lmdb

    env = lmdb.open(args.data, readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    recs = []
    with env.begin() as txn:
        for key, val in txn.cursor():
            recs.append(pickle.loads(val))
    random.Random(args.seed).shuffle(recs)
    recs = recs[:args.n_mols]
    print(f"{len(recs):,} molecules from {args.data}\n")

    is_tree, n_frag, slots, extra_edges = [], [], [], []
    for r in recs:
        k = int(r["n_frag"])
        ce = np.asarray(r["coarse_e_index"])
        n_e = ce.shape[1] if ce.ndim == 2 else 0
        n_frag.append(k)
        is_tree.append(n_e == k - 1)
        extra_edges.append(n_e - (k - 1))
        slots.append(k * (k - 1) / 2)

    tree_frac = float(np.mean(is_tree))
    print("--- connectivity / coarse-graph structure ---")
    print(f"  coarse graph is a tree (|E| = k-1): {tree_frac:.4f}")
    print(f"  edge-count deviation from k-1: {Counter(extra_edges).most_common(5)}")
    if tree_frac > 0.999:
        free = np.array(slots)
        # A labelled tree on k nodes: Cayley gives k^(k-2) of them, against
        # 2^(k(k-1)/2) unconstrained edge configurations.
        ks = np.array(n_frag, dtype=float)
        log2_free = free
        log2_trees = np.where(ks > 2, (ks - 2) * np.log2(np.maximum(ks, 1)), 0.0)
        print(f"\n  unconstrained edge configurations: 2^{free.mean():.1f} on average")
        print(f"  spanning trees (Cayley k^(k-2)):   2^{log2_trees.mean():.1f}")
        print(f"  => the support shrinks by ~2^{(log2_free - log2_trees).mean():.1f}")
        print("  and connectivity comes for free, since every tree is connected.")

    print("\n--- degree of unsaturation ---")
    dbe_ok, dbe_diff = [], []
    for r in recs[:400]:
        try:
            total = dbe_from_formula(r["smi"])
        except Exception:  # noqa: BLE001
            continue
        internal = 0.0
        ok = True
        for f in r["frag_smi_list"]:
            m = Chem.MolFromSmiles(f)
            if m is None:
                ok = False
                break
            # Ring-bond and pi-bond count within the fragment, junctions ignored.
            internal += (m.GetRingInfo().NumRings()
                         + sum(b.GetBondTypeAsDouble() - 1
                               for b in m.GetBonds()
                               if b.GetBondTypeAsDouble() in (2.0, 3.0)))
        if not ok:
            continue
        dbe_diff.append(total - internal)
        dbe_ok.append(True)
    if dbe_diff:
        d = np.array(dbe_diff)
        print(f"  molecule DBE minus summed fragment-internal DBE:")
        print(f"    mean {d.mean():.2f}  std {d.std():.2f}  "
              f"median {np.median(d):.2f}")
        print("  A tight distribution means the fragment multiset is constrained by")
        print("  the formula's DBE before any edge is chosen; a wide one means")
        print("  aromatic bookkeeping is eating the signal and it needs kekulizing.")

    print(f"\n--- for reference ---")
    print(f"  mean fragments per molecule {np.mean(n_frag):.2f}  "
          f"(R2 measured 7.06 on a sample of all folds)")


if __name__ == "__main__":
    main()
