"""Coarse connectivity, enforced in the support: the answer must be a tree.

BRICS cuts acyclic single bonds; cutting a ring takes two cuts, which is why
7.7% of rBRICS coarse edges carry two bonds and 0.00% of BRICS ones do (R11).
The consequence, measured on 2,000 molecules with zero exceptions, is that a
BRICS coarse graph is **always** a tree: |E_coarse| = k - 1 exactly.

The generator was choosing freely among 2^(k(k-1)/2) edge configurations --
2^29.9 on average -- when the answer must be one of the k^(k-2) spanning trees,
2^16.3 on average. That is a 2^13.6 prune of the edge support, and it is the
largest exact constraint available at this level. Connectivity, section 9's
second bullet, comes with it: every tree is connected, so the disconnected
components that plague atom-level graph generation become unreachable rather
than merely penalised.

**This is the pattern the codebase already uses, one level up.** FragFM scores
attachment sites locally and then projects onto a valid global structure with
Blossom max-weight matching. Here the flow scores coarse edges locally and we
project onto a valid global structure with a max-weight spanning tree. Same
shape of solution, same place in the pipeline, one level coarser.

Applied at the final Euler step only. Intermediate states are not trees and
should not be: the flow needs to move through them.
"""

import numpy as np
import torch


def _kruskal(order, u, v, k):
    """Max-weight spanning tree by Kruskal; `order` is edge indices, best first."""
    parent = np.arange(k)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keep, n = [], 0
    for e in order:
        a, b = find(u[e]), find(v[e])
        if a != b:
            parent[a] = b
            keep.append(e)
            n += 1
            if n == k - 1:
                break
    return keep


def max_weight_spanning_tree(scores, edge_index, batch):
    """Boolean over edges: which coarse edges the spanning tree keeps.

    scores:     [n_edge] probability that this coarse edge is a bond
    edge_index: [2, n_edge] global node indices
    batch:      [n_node] molecule index per node

    A molecule whose edges cannot span it -- which should not happen, since the
    candidate set is fully connected -- keeps whatever Kruskal found rather than
    failing, leaving the downstream validity check to catch it.
    """
    dev = scores.device
    s = scores.detach().cpu().numpy()
    ei = edge_index.detach().cpu().numpy()
    b = batch.detach().cpu().numpy()
    edge_mol = b[ei[0]]

    out = np.zeros(s.shape[0], dtype=bool)
    for m in np.unique(edge_mol):
        idx = np.flatnonzero(edge_mol == m)
        nodes = np.flatnonzero(b == m)
        k = nodes.shape[0]
        if k < 2:
            continue
        local = {g: i for i, g in enumerate(nodes)}
        u = np.array([local[x] for x in ei[0, idx]])
        v = np.array([local[x] for x in ei[1, idx]])
        order = np.argsort(-s[idx])  # descending: max-weight
        out[idx[_kruskal(order, u, v, k)]] = True

    return torch.from_numpy(out).to(dev)
