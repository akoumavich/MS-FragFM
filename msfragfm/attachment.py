"""Discrete attachment: per-atom choose-k over candidate partners.

FragFM scores each candidate attachment pair independently with a sigmoid and a
BCE loss, conditioned on one per-molecule latent z.  E0-e (RESULTS.md R7, R10,
R11) showed z carries the attachment decision even through the real Blossom
decode, which leaves GRPO nothing per-fragment to assign attachment credit to.

Three granularities were considered; the measurements picked the third.

*Per coarse edge* -- "which atom pair joins fragments A and B" -- is invalid.
rBRICS cuts rings, and a cut ring joins its two fragments **twice**: 7.7% of
rBRICS coarse edges carry two bonds, against 0% for BRICS.  A formulation that
only works for one decomposition is the wrong formulation.

*Per slot* -- expand each junction atom into one slot per open valence, as
Blossom already does -- always takes exactly one partner.  But slots on the same
atom are interchangeable, so a per-slot categorical is degenerate under
relabelling.

*Per atom, choosing exactly `junction_count` partners* has neither problem.  It
is decomposition-agnostic, has no label degeneracy, and is finer than the
fragment granularity the oracle attributes to.  82.3% of junction atoms have
count 1 and reduce to plain categorical; 17.2% have count 2.

The softmax with k targets is deliberate: its optimum puts mass 1/k on each true
partner, which ranks all k above every distractor.  That is exactly what the
max-weight matching at decode consumes, and it is the property BCE does not
provide, since BCE scores each candidate without reference to its competitors.

Training is local and decoding is global -- Blossom projects the per-atom scores
onto a consistent matching, the standard pattern.
"""

import torch
from torch_geometric.utils import scatter, softmax


def atom_candidates(graph):
    """Symmetrised per-atom view of the candidate list.

    `ae_to_pred_index` stores each candidate once, as (i, j) with i < j, so
    grouping by column 0 alone silently drops every atom that is always the
    larger index.  Each candidate is therefore emitted twice, once per endpoint.

    Returns (into_logits, group, is_true), all of length 2 * n_candidates.
    """
    i, j = graph.ae_to_pred_index
    n = i.numel()
    ar = torch.arange(n, device=i.device)
    into_logits = torch.cat([ar, ar])
    _, group = torch.unique(torch.cat([i, j]), return_inverse=True)
    true = graph.ae_to_pred.bool()
    return into_logits, group, torch.cat([true, true])


def attachment_loss(logits, graph):
    """Cross-entropy of each atom's true partners against a softmax over its
    candidates."""
    into, group, true = atom_candidates(graph)
    if into.numel() == 0:
        return logits.sum() * 0.0
    p = softmax(logits[into], group)
    return -torch.log(p[true].clamp_min(1e-12)).mean()


def partner_counts(graph):
    """True partners per atom, which must equal that atom's junction count."""
    _, group, true = atom_candidates(graph)
    return scatter(true.long(), group, reduce="sum")
