"""The composition constraint: the chosen fragments must sum to the formula.

R17 measured the failure precisely. Generated molecules run **6.97 heavy atoms
short of target on average** -- 21 against 28, one atom too small in every one of
~7 slots -- and hit the exact heavy-atom count 4.5% of the time. Absolute error
9.85 against signed -6.97 makes it a systematic bias rather than symmetric noise.

The mechanism is the bag. Drawing it occurrence-weighted is what makes the right
fragments reachable at all (R16: 74% availability per step, against 0.5% if it
were uniform), but occurrence weighting favours *common* fragments, and common
fragments are *small* ones -- benzene, methyl, carbonyl. The same choice that
fixed reachability biases composition.

Elementwise containment (`formula_mask`) cannot fix this. It is necessary and not
sufficient: seven fragments that each fit inside C17H19NO3 will almost never be
C17H19NO3. Measured, it moved formula match 0.0034 -> 0.0050.

So project instead, which is the third instance of a pattern the pipeline already
runs twice: score locally, then project onto a globally valid structure. Blossom
max-weight matching for attachment, max-weight spanning tree for connectivity,
and here a beam search over slots carrying the element budget.

The budget prunes hard, which is what makes a subset-sum tractable here: a
partial assignment that has already overspent any element is dead, and so is one
whose remaining slots cannot supply what is left. With ~7 slots the search is
small.
"""

import numpy as np
import torch


def project_molecule(scores, cand, frag_counts, target, beam=64, top_k=32,
                     min_size=1, max_size=None):
    """Highest-scoring fragment assignment whose counts sum to `target`.

    scores: [k, n_cand] log-probabilities per slot over the candidate fragments
    cand:   [n_cand] global fragment ids
    beam:   partial assignments kept per slot
    top_k:  candidates considered per slot

    Vectorised over the whole beam at once.  The obvious implementation loops
    over states and then candidates in Python, which costs 20.5 ms per molecule
    and made the projection a third of evaluation wall-clock; expanding the beam
    as one [beam, top_k, n_elem] array instead is the same search, two orders of
    magnitude faster, and needs no worker processes.

    Returns a list of k indices into `cand`, or None if nothing is feasible.
    """
    k, n_cand = scores.shape
    counts = frag_counts[cand].astype(np.int32)
    if max_size is None:
        max_size = int(counts.sum(1).max()) if n_cand else 1

    # Slots filled most-confident first: committing the decisions the model is
    # surest about early gives the budget the most to prune against.
    order = np.argsort(-scores.max(1))
    rem = target.astype(np.int32)[None, :]
    sc = np.zeros(1)
    ch = np.full((1, k), -1, dtype=np.int32)

    for step, slot in enumerate(order):
        left = k - step - 1
        t = min(top_k, n_cand)
        top = np.argpartition(-scores[slot], t - 1)[:t] if t < n_cand             else np.arange(n_cand)

        new_rem = rem[:, None, :] - counts[top][None, :, :]     # [S, T, E]
        ok = (new_rem >= 0).all(-1)
        total = new_rem.sum(-1)
        # The remaining slots must be able to supply exactly what is left.
        ok &= (total == 0) if left == 0 else (
            (total >= left * min_size) & (total <= left * max_size))
        if not ok.any():
            return None

        si, ti = np.nonzero(ok)
        flat_sc = (sc[:, None] + scores[slot, top][None, :])[si, ti]
        flat_rem = new_rem[si, ti]

        o = np.argsort(-flat_sc)
        si, ti, flat_sc, flat_rem = si[o], ti[o], flat_sc[o], flat_rem[o]
        # Deduplicate on the remaining budget. Without this the beam fills with
        # variants that spent the same atoms by different routes, exploring one
        # budget path deeply instead of many shallowly -- which is how a
        # subset-sum beam starves. The array is score-sorted, so np.unique's
        # first occurrence of each budget is also its best-scoring one.
        _, first = np.unique(flat_rem, axis=0, return_index=True)
        keep = np.sort(first)[:beam]

        ch = ch[si[keep]].copy()
        ch[:, slot] = top[ti[keep]]
        rem, sc = flat_rem[keep], flat_sc[keep]

    return ch[int(np.argmax(sc))].tolist()


def project_batch(logits, batch, cand_ids, frag_counts, targets, beam=64,
                  top_k=32):
    """Apply the projection per molecule; fall back to arg-max where infeasible.

    Returns (choice per node, fraction of molecules projected successfully).
    """
    logp = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
    b = batch.detach().cpu().numpy()
    cand = cand_ids.detach().cpu().numpy()
    out = logp.argmax(1)
    n_ok = n_mol = 0
    for m in np.unique(b):
        slots = np.flatnonzero(b == m)
        n_mol += 1
        got = project_molecule(logp[slots], cand, frag_counts, targets[m],
                               beam=beam, top_k=top_k)
        if got is None:
            continue
        out[slots] = got
        n_ok += 1
    return torch.from_numpy(out).to(logits.device), n_ok / max(n_mol, 1)
