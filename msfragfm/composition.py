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


def _feasible_suffix(remaining, n_slots_left, min_size, max_size):
    """Can `n_slots_left` fragments still supply exactly `remaining`?"""
    if (remaining < 0).any():
        return False
    total = int(remaining.sum())
    if n_slots_left == 0:
        return total == 0
    return n_slots_left * min_size <= total <= n_slots_left * max_size


def project_molecule(scores, cand, frag_counts, target, beam=256, min_size=1,
                     max_size=None):
    """Highest-scoring fragment assignment whose counts sum to `target`.

    scores: [k, n_cand] log-probabilities per slot over the candidate fragments
    cand:   [n_cand] global fragment ids
    Returns a list of k indices into `cand`, or None if nothing is feasible.
    """
    k, n_cand = scores.shape
    counts = frag_counts[cand]                      # [n_cand, n_elem]
    sizes = counts.sum(1)
    if max_size is None:
        max_size = int(sizes.max()) if n_cand else 1

    # Slots are filled most-confident first: committing the decisions the model
    # is surest about early gives the budget the most to prune against.
    order = np.argsort(-scores.max(1))
    states = [(0.0, target.astype(np.int32), [])]
    for step, slot in enumerate(order):
        left = k - step - 1
        top = np.argsort(-scores[slot])[:beam]
        nxt = []
        for sc, rem, chosen in states:
            new_rem = rem[None, :] - counts[top]
            ok = (new_rem >= 0).all(1)
            for j in np.flatnonzero(ok):
                r = new_rem[j]
                if not _feasible_suffix(r, left, min_size, max_size):
                    continue
                nxt.append((sc + scores[slot, top[j]], r, chosen + [(slot, top[j])]))
        if not nxt:
            return None
        nxt.sort(key=lambda t: -t[0])
        states = nxt[:beam]

    best = states[0]
    if best[1].sum() != 0:
        return None
    out = [0] * k
    for slot, j in best[2]:
        out[slot] = int(np.argsort(-scores[slot])[:beam][j])
    return out


def project_batch(logits, batch, cand_ids, frag_counts, targets, beam=256):
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
        got = project_molecule(logp[slots], cand, frag_counts, targets[m], beam)
        if got is None:
            continue
        out[slots] = got
        n_ok += 1
    return torch.from_numpy(out).to(logits.device), n_ok / max(n_mol, 1)
