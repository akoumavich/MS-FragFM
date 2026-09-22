"""Decode attachment the way the deployed pipeline does: max-weight matching.

FragFM's generation path does not threshold the attachment scores.  It expands
each junction atom into as many slots as its junction count, runs Blossom
max-weight matching over the slots, and contracts back
(`genererate_utils.realize_single_fine_graph_dict`, lines 195-232).

E0-e evaluated `pred > 0.5` instead, so its numbers describe the scoring head
rather than the decode rule.  Matching depends only on the relative ordering of
scores, so it can be robust where a fixed threshold is not -- which is exactly
the difference that decides whether the latent z really costs us accuracy.

This mirrors that step on a ground-truth coarse graph, where the candidate list
is `ae_to_pred_index` rather than a generated one.
"""

import networkx as nx
import torch


def blossom_select(logits, graph):
    """Boolean over `ae_to_pred_index`: which candidate pairs the matching picks.

    Runs per molecule, on CPU via networkx, as the released code does.
    """
    device = logits.device
    i_all, j_all = graph.ae_to_pred_index
    cand_mol = graph.batch[i_all]
    chosen = torch.zeros_like(logits, dtype=torch.bool)

    for m in range(int(graph.batch.max()) + 1):
        sel = (cand_mol == m).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        atoms = (graph.batch == m).nonzero(as_tuple=True)[0]
        jc = graph.h_junction_count[atoms]
        junc = atoms[jc != 0]
        if junc.numel() == 0:
            continue
        pos = torch.full((int(atoms.max()) + 1,), -1, dtype=torch.long, device=device)
        pos[junc] = torch.arange(junc.numel(), device=device)

        n = junc.numel()
        score = torch.full((n, n), -999.0, device=device)
        a, b = pos[i_all[sel]], pos[j_all[sel]]
        score[a, b] = logits[sel]
        score[b, a] = logits[sel]

        # One slot per open valence, then a perfect matching over slots.
        dup = graph.h_junction_count[junc].long()
        exp = score.repeat_interleave(dup, 0).repeat_interleave(dup, 1)
        mask = _match(exp)
        # Contract: slot pairs summed back onto atom pairs, so a fragment pair
        # joined twice (a cut ring) reads as 2.
        rows = torch.stack([r.sum(0) for r in torch.split(mask, dup.tolist(), 0)])
        cont = torch.stack([c.sum(1) for c in torch.split(rows, dup.tolist(), 1)], 1)
        chosen[sel] = cont[a, b] > 0

    return chosen


def _match(score):
    s = score.detach().cpu().numpy()
    g = nx.Graph()
    n = s.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=float(s[i, j]))
    out = torch.zeros((n, n), dtype=torch.long)
    for i, j in nx.max_weight_matching(g, maxcardinality=True):
        out[i, j] = out[j, i] = 1
    return out
