"""Atom-level assembly that realises the coarse tree by construction.

R21: only 35.3% of assemblies come out in one piece, 2.37 components on average,
and 79% of the heavy-atom shortfall is mass thrown away by
`reconstruct_to_rdmol(get_largest=True)` after the pieces fail to join.

The cause is a mismatch of levels.  We constrain the *coarse* graph to a tree, so
the fragments are connected there.  The atom-level bonds then come from a
max-weight matching over junction slots, and **a matching is not required to
realise that tree**: it can bond A-B twice and leave A-C unbonded, or pair slots
of fragments that are not coarse-adjacent.  Either way the molecule falls apart,
and the truncation hides it behind a valid-looking SMILES.

The constraint that fixes it is exact: **every coarse edge is realised by exactly
one atom-level bond**.  Under BRICS that is not an approximation -- R11 measured
100.00% of BRICS coarse edges carrying exactly one bond, against 92.3% for
rBRICS, and R13 chose BRICS.

The bookkeeping already supports it.  A fragment's total `junction_count` is the
number of cut bonds incident to it, which is exactly its degree in the coarse
tree, so every fragment has precisely as many slots as it has edges to satisfy.
A feasible assignment therefore always exists; the only question is which one
scores best.

Greedy by score with a repair pass, rather than exact optimisation: the choice is
coupled across endpoints (an edge's score depends on the atom chosen at *both*
ends), which makes the exact problem quadratic, while the instance is tiny and
82.3% of junction atoms have a single slot and so no choice at all.  Returning
None on failure lets the caller keep the old path rather than emit something
worse.
"""

import numpy as np
import torch

# Set by the caller; off by default so the released path is untouched.
ENABLED = False


def assemble_tree(d):
    """Select one atom pair per coarse edge.

    d: a debatched fine-graph dict, carrying `frag_batch`, `ae_to_pred_index`,
       `ae_to_pred_prob` and `h_junction_count`.

    Returns a 0/1 tensor over `ae_to_pred_index`, or None if no assignment
    respects the slot capacities.
    """
    idx = d.ae_to_pred_index.detach().cpu().numpy()
    if idx.size == 0:
        return None
    prob = d.ae_to_pred_prob.detach().cpu().numpy()
    frag = d.frag_batch.detach().cpu().numpy()
    cap = d.h_junction_count.detach().cpu().numpy().astype(np.int64).copy()

    fa, fb = frag[idx[0]], frag[idx[1]]
    # A candidate only exists between coarse-adjacent fragments, so the distinct
    # fragment pairs appearing here *are* the coarse edges.
    lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
    key = lo * (frag.max() + 1) + hi
    edges = np.unique(key)

    chosen = {}
    for c in np.argsort(-prob):
        e = key[c]
        if e in chosen:
            continue
        i, j = idx[0, c], idx[1, c]
        if cap[i] > 0 and cap[j] > 0:
            chosen[e] = c
            cap[i] -= 1
            cap[j] -= 1
        if len(chosen) == edges.size:
            break

    # Repair: any coarse edge the greedy pass skipped because its best
    # candidates were already spent.
    for e in edges:
        if e in chosen:
            continue
        cands = np.flatnonzero(key == e)
        for c in cands[np.argsort(-prob[cands])]:
            i, j = idx[0, c], idx[1, c]
            if cap[i] > 0 and cap[j] > 0:
                chosen[e] = c
                cap[i] -= 1
                cap[j] -= 1
                break
        else:
            return None  # no capacity left; keep the caller's existing result

    out = np.zeros(idx.shape[1], dtype=np.int64)
    out[list(chosen.values())] = 1
    return torch.from_numpy(out).to(d.ae_to_pred_index.device)
