"""Conservation of composition under fragmentation, enforced exactly.

Method C states the law: every observed peak must be explainable as a connected
subgraph of the candidate whose formula mass matches the peak within tolerance,
allowing for hydrogen rearrangements.  It then says exact enumeration over
connected subgraphs is exponential and reaches for a differentiable relaxation,
deferred to the second paper.

**At fragment level the enumeration is tractable, but not for the reason I first
gave.**  Moving from ~28 heavy atoms to ~7 fragments takes the subset space from
2^28 to 2^7, and the tree structure (R17) restricts it further to connected
subtrees.  That argument is right on the mean and wrong on the tail: measured,
molecules average 17,101 connected subtrees and reach 553,030, because the cost
is mean(2^k) and not 2^(mean k), and the test fold runs to k ~ 20.

What makes it genuinely cheap is a restriction that is also more faithful to the
chemistry.  MAGMa and ICEBERG model fragments as the result of breaking one to
three bonds, not as arbitrary connected subgraphs.  On a tree, cutting j edges
yields exactly j+1 components, so bounding the cuts turns 2^k into O(k^d): at
k=20 that is 1,160 cut-sets rather than a million subsets.  The relaxation
remains what paper two needs for gradients and for atom-level work; the law
itself is usable exactly, here, now.

So the only physics that knows about the observation becomes available in the
support and as an exact score, in paper one, rather than as a soft loss in paper
two.  The proposal anticipates the direction -- "at fragment level this becomes
much cheaper, since s ranges over fragments rather than atoms" -- and stops one
step short of the consequence.

This addresses R18 directly.  Generated molecules run 6.97 heavy atoms short
because the occurrence-weighted bag favours small common fragments, and nothing
told the model how big the pieces should be.  The peaks say exactly that: an
MS/MS peak *is* a fragment mass.
"""

import itertools

import numpy as np

PROTON = 1.007276
ADDUCT_MASS = {"[M+H]+": PROTON, "[M+Na]+": 22.989218}
# Monoisotopic masses, keyed by atomic number.
_MASS = {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403,
         15: 30.973762, 16: 31.972071, 17: 34.968853, 35: 78.918338,
         53: 126.904473, 34: 79.916522, 14: 27.976927, 5: 11.009305,
         33: 74.921596}


def fragment_masses(sample):
    """Mass of each fragment, including the hydrogens it carries in the intact
    molecule.

    Taken from the stored atom list rather than the fragment SMILES: BRICS cuts
    bonds without removing atoms, so a subtree's atoms are exactly the union of
    its fragments' atoms, and hydrogens stay where they were.  Reconstructing
    from fragment SMILES would instead give the capped fragment, which is a
    different molecule.
    """
    h = np.asarray(sample["h"])
    fb = np.asarray(sample["h_frag_batch"])
    k = int(sample["n_frag"])
    out = np.zeros(k)
    for i in range(k):
        out[i] = sum(_MASS.get(int(z), 0.0) for z in h[fb == i])
    return out


def cleavage_subtrees(edge_index, k, max_cuts=3):
    """Fragments reachable by breaking at most `max_cuts` coarse bonds.

    Not every connected subtree: MAGMa and ICEBERG model fragmentation as one to
    three bond cleavages, and enumerating all connected subtrees both overcounts
    chemically and costs mean(2^k), measured at 17,101 per molecule and 553,030
    at the tail.  On a tree, cutting j edges yields exactly j+1 components, so
    this is O(k^max_cuts).
    """
    ei = np.asarray(edge_index)
    edges = list(zip(ei[0], ei[1])) if ei.ndim == 2 else []
    seen, out = set(), []
    for j in range(0, min(max_cuts, len(edges)) + 1):
        for cut in itertools.combinations(range(len(edges)), j):
            parent = list(range(k))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for e, (a, b) in enumerate(edges):
                if e in cut:
                    continue
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
            comps = {}
            for v in range(k):
                comps.setdefault(find(v), []).append(v)
            for c in comps.values():
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    out.append(np.array(c))
    return out


def explained_fraction(sample, mz, intensity, adduct="[M+H]+", ppm=20.0,
                       max_h_shift=2, intensity_weighted=True, max_cuts=3):
    """Share of observed peaks explainable by some connected subtree.

    Hydrogen rearrangement is the reason for `max_h_shift`: a fragment leaves
    with or without the hydrogen at the bond that broke, and homolytic versus
    heterolytic cleavage moves one either way, so a window of +/-2 H is the
    standard allowance rather than a fudge factor.
    """
    masses = fragment_masses(sample)
    subs = cleavage_subtrees(sample["coarse_e_index"], int(sample["n_frag"]),
                             max_cuts=max_cuts)
    cand = np.array([masses[s].sum() for s in subs])
    shifts = np.arange(-max_h_shift, max_h_shift + 1) * _MASS[1]
    cand = (cand[:, None] + shifts[None, :]).ravel() + ADDUCT_MASS.get(adduct, PROTON)

    mz = np.asarray(mz, dtype=float)
    hit = np.zeros(mz.shape[0], dtype=bool)
    for i, m in enumerate(mz):
        hit[i] = np.abs(cand - m).min() <= m * ppm * 1e-6
    if intensity_weighted and np.sum(intensity) > 0:
        w = np.asarray(intensity, dtype=float)
        return float((w * hit).sum() / w.sum()), len(subs)
    return float(hit.mean()), len(subs)


# --- applying the law to generated candidates -----------------------------
#
# R19 measured the separation and stopped there; the score was never wired into
# generation or ranking.  Everything hard we enforce -- containment, spanning
# tree, composition sum -- uses only the precursor formula, which comes from MS1.
# The MS/MS peaks constrain nothing, which is the gap this closes.
#
# A generated candidate needs no re-decomposition: the sampler hands us the
# coarse graph directly, so the subtree masses come straight off it.


def pool_fragment_masses(frag_lmdb, cache=None):
    """Monoisotopic mass each pool fragment contributes to a molecule.

    Not the mass of the capped fragment.  A junction atom bonds to a neighbouring
    fragment in the assembled molecule, so it carries one fewer hydrogen than the
    standalone fragment would.  RDKit already accounts for this when the dummy
    atoms are present, so summing atom masses plus `GetTotalNumHs` over the
    non-dummy atoms is exact; `ExactMolWt` of the capped fragment is not.
    """
    import pickle

    import numpy as np
    from rdkit import Chem

    if cache and __import__("pathlib").Path(cache).exists():
        return np.load(cache)

    import lmdb

    env = lmdb.open(str(frag_lmdb), readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    n = int(env.stat()["entries"])
    out = np.zeros(n)
    with env.begin() as txn:
        for _, v in txn.cursor():
            rec = pickle.loads(v)
            i = int(rec["key"].split("_")[1])
            m = Chem.MolFromSmiles(rec["smi"])
            if m is None:
                continue
            out[i] = sum(_MASS.get(a.GetAtomicNum(), 0.0)
                         + a.GetTotalNumHs() * _MASS[1]
                         for a in m.GetAtoms() if a.GetAtomicNum() > 0)
    env.close()
    if cache:
        np.save(cache, out)
    return out


def explained_from_coarse(frag_ids, edges, pool_masses, mz, intensity,
                          adduct="[M+H]+", ppm=20.0, max_h_shift=2, max_cuts=1):
    """Explained peak intensity for a generated coarse graph.

    frag_ids: [k] global fragment ids chosen for this molecule
    edges:    [2, n_edge] coarse edges, local indices
    """
    k = len(frag_ids)
    if k == 0 or len(mz) == 0:
        return 0.0
    masses = pool_masses[np.asarray(frag_ids)]
    subs = cleavage_subtrees(edges, k, max_cuts=max_cuts)
    cand = np.array([masses[s].sum() for s in subs])
    shifts = np.arange(-max_h_shift, max_h_shift + 1) * _MASS[1]
    cand = (cand[:, None] + shifts[None, :]).ravel() + ADDUCT_MASS.get(adduct, PROTON)

    mz = np.asarray(mz, dtype=float)
    hit = np.array([np.abs(cand - m).min() <= m * ppm * 1e-6 for m in mz])
    w = np.asarray(intensity, dtype=float)
    return float((w * hit).sum() / w.sum()) if w.sum() > 0 else float(hit.mean())
