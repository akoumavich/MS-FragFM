"""The formula constraint, enforced in the support.

Proposal section 9 lists three places to put a constraint and says the support is
the strongest: mask the sampler so violations are unreachable, which is exact,
free, and trades nothing against likelihood.  R16 found we had none of it -- the
formula was a soft encoder input only -- and that 99.6% of generated candidates
carried a formula the target cannot have, every one of them knowable as
impossible before it was sampled.

The constraint used here is elementwise containment: a fragment may appear in a
molecule of formula F only if its heavy-atom counts are <= F's, element by
element.  That is exact rather than heuristic, because BRICS cuts bonds and never
removes atoms, so the heavy atoms of a molecule are exactly the union of its
fragments' heavy atoms.  A fragment carrying a chlorine cannot appear in a
chlorine-free target; a fragment with eighteen carbons cannot appear in a
seventeen-carbon one.

Hydrogen is excluded.  Junction atoms lose hydrogens when fragments bond, so a
fragment's hydrogen count is not conserved and containment would be wrong on it.
Dummy atoms marking junctions are excluded for the same reason.

This is a necessary condition per fragment, not a sufficient one for the
multiset: nothing here enforces that the chosen fragments *sum* to F.  The
stronger constraint needs a running budget across slots, which the flow does not
have because it updates every slot at once.  Pruning what is individually
impossible is the part that is exact and cheap.
"""

import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from msfragfm.spectrum import ELEMENTS, parse_formula

RDLogger.DisableLog("rdApp.*")

# Hydrogen is not conserved under fragmentation; everything else is.
HEAVY = tuple(e for e in ELEMENTS if e != "H")
_H_IDX = ELEMENTS.index("H")


def _counts_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    c = np.zeros(len(HEAVY) + 1, dtype=np.int16)  # last bin: anything unlisted
    for a in mol.GetAtoms():
        if a.GetAtomicNum() <= 1:  # H, and dummy junction markers
            continue
        s = a.GetSymbol()
        c[HEAVY.index(s) if s in HEAVY else len(HEAVY)] += 1
    return c


def fragment_counts(frag_lmdb, cache=None):
    """Heavy-atom counts per fragment, [n_frag, len(HEAVY)+1], indexed by frag id."""
    if cache and Path(cache).exists():
        return np.load(cache)

    import lmdb

    env = lmdb.open(str(frag_lmdb), readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    n = int(env.stat()["entries"])
    out = np.zeros((n, len(HEAVY) + 1), dtype=np.int16)
    # An unparseable fragment is marked impossible everywhere rather than
    # silently admissible: a large sentinel fails containment against any target.
    with env.begin() as txn:
        for k, v in txn.cursor():
            rec = pickle.loads(v)
            i = int(rec["key"].split("_")[1])
            c = _counts_from_smiles(rec["smi"])
            out[i] = c if c is not None else 9999
    env.close()
    if cache:
        np.save(cache, out)
    return out


def target_counts(formulas):
    """Heavy-atom counts per target formula, [n, len(HEAVY)+1]."""
    out = np.zeros((len(formulas), len(HEAVY) + 1), dtype=np.int16)
    for i, f in enumerate(formulas):
        full = parse_formula(f)
        out[i, :len(HEAVY)] = [full[ELEMENTS.index(e)] for e in HEAVY]
        out[i, len(HEAVY)] = full[-1]  # the "other" bin from parse_formula
    return out


def admissible(frag_counts_arr, target_counts_arr):
    """[n_targets, n_frag] bool: may this fragment appear in this formula?"""
    adm = (frag_counts_arr[None, :, :] <= target_counts_arr[:, None, :]).all(-1)
    return adm


def report(adm, frag_counts_arr):
    kept = adm.sum(1)
    return {
        "pool": int(adm.shape[1]),
        "admissible_mean": float(kept.mean()),
        "admissible_frac": float(kept.mean() / adm.shape[1]),
        "pruned_frac": float(1 - kept.mean() / adm.shape[1]),
        "min_admissible": int(kept.min()),
    }
