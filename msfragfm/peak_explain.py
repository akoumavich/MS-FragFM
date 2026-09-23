"""Conservation of composition under fragmentation, enforced exactly.

Method C states the law: every observed peak must be explainable as a connected
subgraph of the candidate whose formula mass matches the peak within tolerance,
allowing for hydrogen rearrangements.  It then says exact enumeration over
connected subgraphs is exponential and reaches for a differentiable relaxation,
deferred to the second paper.

**At fragment level the enumeration is not exponential, it is trivial, and two
measurements make it so.**  A molecule has ~7 fragments rather than ~28 heavy
atoms, so the subset space is 2^7 = 128 rather than 2^28 = 268 million.  And the
BRICS coarse graph is always a tree (R17, 2000/2000), so only connected subtrees
count: 28 of them for a 7-node path, 70 for a 7-node star.  The relaxation is an
atom-level necessity; it is not needed here.

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


def connected_subtrees(edge_index, k):
    """Every connected vertex subset of the coarse tree, as boolean masks."""
    adj = [[] for _ in range(k)]
    ei = np.asarray(edge_index)
    if ei.ndim == 2:
        for a, b in zip(ei[0], ei[1]):
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
    out = []
    for r in range(1, k + 1):
        for s in itertools.combinations(range(k), r):
            ss = set(s)
            seen, stack = {s[0]}, [s[0]]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in ss and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if seen == ss:
                out.append(np.array(s))
    return out


def explained_fraction(sample, mz, intensity, adduct="[M+H]+", ppm=20.0,
                       max_h_shift=2, intensity_weighted=True):
    """Share of observed peaks explainable by some connected subtree.

    Hydrogen rearrangement is the reason for `max_h_shift`: a fragment leaves
    with or without the hydrogen at the bond that broke, and homolytic versus
    heterolytic cleavage moves one either way, so a window of +/-2 H is the
    standard allowance rather than a fudge factor.
    """
    masses = fragment_masses(sample)
    subs = connected_subtrees(sample["coarse_e_index"], int(sample["n_frag"]))
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
