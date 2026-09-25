"""The fragment count handed to generation.

Generation needs n_frag per molecule and it is unknown at test time. The formula
pins the heavy-atom total exactly, so the count has been drawn from the empirical
p(n_frag | n_heavy) measured on the train fold.

R28 measured what that costs: the drawn count is exact 10% of the time with a mean
absolute error of 4.2 on molecules of about seven fragments, and feeding the true
count is worth 7.2 points of fragment recall. The prior's own best point estimate,
its conditional median, is exact only 16% with an error of 3.16 -- so the prior is
close to uninformative and no better *use* of it recovers the gap.
"""

import pickle
from collections import defaultdict

import numpy as np
from rdkit import Chem


def frag_count_prior(env):
    """p(n_frag | n_heavy) from the train fold, and the true n_frag per molecule.

    The second return value is the oracle arm: it separates "the model picks the
    wrong fragments" from "the model was told to pick the wrong *number* of
    fragments", which the marginal prior guarantees for most candidates.
    """
    prior = defaultdict(list)
    true_n = {}
    with env.begin() as txn:
        for key, val in txn.cursor():
            smp = pickle.loads(val)
            mol = Chem.MolFromSmiles(smp["smi"])
            if mol is None:
                continue
            true_n[Chem.MolToSmiles(mol)] = int(smp["n_frag"])
            if key.decode().startswith("train"):
                prior[mol.GetNumHeavyAtoms()].append(int(smp["n_frag"]))
    return prior, true_n


def frag_count_pool(prior, n_heavy):
    for d in (0, 1, 2, 3, 5, 8):  # widen the window until the bucket is populated
        pool = [v for k in range(n_heavy - d, n_heavy + d + 1) for v in prior.get(k, [])]
        if len(pool) >= 20:
            return pool
    return []


def frag_counts_for_group(prior, n_heavy, rng, mode, group, n_true):
    """The n_frag handed to each candidate in one spectrum's group.

    `sample` is 16 iid draws and is what every result before R28 used. `median`
    minimises absolute error. `spread` covers the same conditional distribution at
    even quantiles; R29 measured it as no better than `sample`, because the two
    share a marginal and so the same expected error per candidate -- stratifying
    removes variance in the group's composition, not error in any member.
    `oracle` is the bound.
    """
    if mode == "oracle" and n_true is not None:
        return [n_true] * group
    pool = frag_count_pool(prior, n_heavy)
    if not pool:
        return [max(2, round(n_heavy / 4))] * group
    if mode == "median":
        return [int(np.median(pool))] * group
    if mode == "spread":
        qs = np.quantile(pool, np.linspace(0.05, 0.95, group))
        return [int(min(n_heavy, max(2, round(q)))) for q in qs]
    return [int(rng.choice(pool)) for _ in range(group)]
