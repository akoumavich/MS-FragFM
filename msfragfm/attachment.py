"""Discrete attachment sites.

FragFM scores each candidate attachment pair independently with a sigmoid and a
BCE loss, conditioned on a single per-molecule latent z.  E0-e (RESULTS.md R7,
R10) showed that z carries the attachment decision for roughly four molecules in
five, which leaves GRPO nothing per-fragment to assign attachment credit to.

The candidate set is already the right object.  For each coarse edge -- each
bonded fragment pair A-B -- the candidates are the (junction atom of A, junction
atom of B) pairs, and exactly one of them is the real bond.  That is a
categorical variable per coarse edge, not a set of independent binary ones.

Treating it as categorical does three things at once: it puts "exactly one bond
per coarse edge" in the support rather than leaving it to the loss, it gives each
coarse edge its own discrete variable for advantages to land on, and it removes
the need for z to carry attachment at all.

The group ids are derived at load time from fragment membership, so no
reprocessing of the LMDBs is needed.
"""

import torch
from torch_scatter import scatter_max

_MAX_FRAG = 64  # rBRICS tops out at 45 fragments per molecule (RESULTS.md R2.1)


def coarse_edge_groups(graph):
    """Contiguous group id per candidate attachment pair.

    Candidates sharing a group belong to the same fragment pair in the same
    molecule and compete for the one real bond between those fragments.
    """
    i, j = graph.ae_to_pred_index  # global atom indices (PyG offsets these)
    fa, fb = graph.h_frag_batch[i], graph.h_frag_batch[j]  # per-molecule, not offset
    lo, hi = torch.minimum(fa, fb), torch.maximum(fa, fb)
    key = graph.batch[i] * _MAX_FRAG * _MAX_FRAG + lo * _MAX_FRAG + hi
    _, group = torch.unique(key, return_inverse=True)
    return group


def check_one_hot(graph, group):
    """Does every coarse edge carry exactly one true bond?

    The categorical formulation is only valid if it does.  Returns counts of true
    bonds per group so a violation is visible rather than silently averaged away.
    """
    from torch_geometric.utils import scatter

    n_true = scatter(graph.ae_to_pred.long(), group, reduce="sum")
    n_cand = scatter(torch.ones_like(group), group, reduce="sum")
    return n_true, n_cand


def attachment_loss(logits, graph, group):
    """Cross-entropy of the true attachment against a softmax over each group."""
    from torch_geometric.utils import softmax

    p = softmax(logits, group)
    true = graph.ae_to_pred.bool()
    return -torch.log(p[true].clamp_min(1e-12)).mean()


def attachment_decode(logits, group):
    """One bond per coarse edge: the arg-max candidate in each group."""
    _, argmax = scatter_max(logits, group, dim=0)
    chosen = torch.zeros_like(logits, dtype=torch.bool)
    chosen[argmax[argmax < logits.numel()]] = True
    return chosen
