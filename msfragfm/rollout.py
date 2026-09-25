"""Roll the policy out and score it, for the self-conditioning and RL arms.

R25 through R31 closed every explanation for the 33% generated against 80%
teacher-forced per-fragment accuracy except one: the model is accurate at the
states teacher forcing puts it in and wrong at the states generation reaches.
Training corruption only ever masks *true* fragments, so it never produces a state
where an earlier slot holds a **wrong** fragment -- which is most of what a rollout
is made of after the first few Euler steps.

Every arm needs the same two things from a rollout, so they live here rather than
in the trainer:

* the realised coarse graph, which is what `sample_molecule_graph_dynamic` already
  returns -- global fragment ids per node, the full edge index, edge types, the
  latent and the batch vector. Substituting those into the training step reuses
  FragFM's corruption exactly rather than transcribing it.
* the reward, **per node as well as per sample**. c-DTM's target is per variable, so
  a per-fragment reward is its native form; GRPO and DMPO reduce the same vector to
  a scalar. Computing it once keeps the arms comparing like with like.

The reward is multiset fragment recall, which is the quantity R24 introduced and
every result since has tracked. Per node it is "did this slot's fragment appear in
the true multiset, counting multiplicity", and the per-node values average to the
per-sample recall by construction -- so the scalar the RL arms see and the metric
the evaluation reports are the same number.
"""

from collections import Counter

import torch


def rollout(sampler, n_frags):
    """One batch of candidates. Returns the realised coarse graph.

    The sampler must already carry its conditioning (`sampler.cond`, and the
    cross-attention memory when the checkpoint uses it) and any support
    constraints, exactly as the evaluation path sets them.
    """
    with torch.no_grad():
        h_type, e_index, e_type, z, batch = sampler.sample_molecule_graph_dynamic(
            n_frags=n_frags)
    return {"h_type": h_type, "e_index": e_index, "e_type": e_type,
            "z": z, "batch": batch}


def fragment_rewards(h_type, batch, true_ids):
    """Per-node and per-sample multiset recall.

    h_type:   [n_node] global fragment ids the rollout committed to
    batch:    [n_node] which sample each node belongs to
    true_ids: list of iterables of true global fragment ids, one per sample

    A node scores 1 if its fragment is still unclaimed in its sample's true
    multiset. Multiplicity is honoured by claiming: two nodes holding the same
    fragment both score only if the truth contains it twice. Per-sample recall is
    matched / len(true), so a sample with more nodes than the truth has is
    penalised through the denominator rather than by clipping.
    """
    dev = h_type.device
    n_sample = len(true_ids)
    per_node = torch.zeros(h_type.numel(), device=dev)
    per_sample = torch.zeros(n_sample, device=dev)

    h_cpu, b_cpu = h_type.tolist(), batch.tolist()
    slots = [[] for _ in range(n_sample)]
    for pos, (frag, b) in enumerate(zip(h_cpu, b_cpu)):
        if b < n_sample:
            slots[b].append((pos, frag))

    for s in range(n_sample):
        remaining = Counter(int(t) for t in true_ids[s])
        n_true = sum(remaining.values())
        matched = 0
        for pos, frag in slots[s]:
            if remaining.get(frag, 0) > 0:
                remaining[frag] -= 1
                per_node[pos] = 1.0
                matched += 1
        per_sample[s] = matched / n_true if n_true else 0.0
    return per_node, per_sample


def group_view(per_sample, group):
    """[n_groups, G], the shape the group-relative objectives expect."""
    return per_sample.view(-1, group)
