"""Collapse instrumentation for GRPO, logged from the first step.

E8 (proposal section 15).  Mode collapse is the reported default under
policy-gradient RL in RLVR, diffusion RL and molecular RL alike, so these are
measured from the start rather than reached for once numbers look wrong.

Everything here shares one x-axis, the training step, so the reward-hacking plot
(oracle score against held-out accuracy) and the collapse plot (top-1 against
top-10) are the same figure with different y-axes.

**The conditional setting inverts the usual reading, which is the trap.**
Property optimisation wants diverse hits, so concentration is bad.  Elucidation
has one right answer per spectrum, so concentrating *within* a spectrum is the
goal.  The failure is concentrating *across* spectra -- the policy emitting the
same fragments whatever it is shown, having stopped conditioning.  So
`mean_pairwise_tanimoto` falling is ambiguous on its own, and the number that is
not ambiguous is `across_within_ratio`.

Fingerprint is stated rather than implied: Morgan, radius 2, 2048 bits.  The
v1.5 audit flags that these diverge across papers and change conclusions.
"""

from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_RADIUS, FP_BITS = 2, 2048
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def _fp(smi):
    mol = Chem.MolFromSmiles(smi) if smi else None
    return _GEN.GetFingerprint(mol) if mol is not None else None


def _canon(smi):
    try:
        return Chem.CanonSmiles(smi)
    except Exception:  # noqa: BLE001
        return None


def group_metrics(candidates, scores, truth, ks=(1, 10)):
    """Metrics for one spectrum's group of G samples.

    candidates: G generated SMILES (invalid entries may be None or "X")
    scores:     G oracle scores, used to rank -- top-k is a ranked metric, so
                without the oracle's ordering "top-10" would just mean "any of 10"
    truth:      the reference SMILES
    """
    ref = _canon(truth)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda t: -t[0])]
    canon = [_canon(c) for c in ranked]
    valid = [c for c in canon if c is not None]

    out = {f"top{k}": float(ref is not None and ref in canon[:k]) for k in ks}
    out["validity"] = len(valid) / max(len(candidates), 1)
    # Unique *valid* molecules, not unique strings: the quantity that decides
    # whether another oracle call can buy anything.
    out["n_unique"] = len(set(valid))
    out["unique_frac"] = out["n_unique"] / max(len(candidates), 1)

    fps = [f for f in (_fp(c) for c in set(valid)) if f is not None]
    out["mean_pairwise_tanimoto"] = _mean_pairwise(fps)
    if ref is not None:
        rf = _fp(ref)
        sims = [DataStructs.BulkTanimotoSimilarity(rf, fps)] if fps else [[]]
        out["max_tanimoto_to_truth"] = max(sims[0]) if sims[0] else 0.0
    return out


def _mean_pairwise(fps):
    if len(fps) < 2:
        return 1.0  # a group of one is maximally concentrated
    sims = [s for i in range(1, len(fps))
            for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
    return float(np.mean(sims))


def across_within_ratio(groups):
    """Is the policy still conditioning on the spectrum?

    groups: list of per-spectrum candidate lists.

    Returns mean within-group similarity over mean across-group similarity.  A
    ratio near 1 means samples drawn for *different* spectra look as alike as
    samples drawn for the same one -- the policy has stopped listening.  Rising
    within-group similarity with the ratio held up is sharpening, which is what
    we want; both rising together is collapse.
    """
    fps = [[f for f in (_fp(c) for c in set(g) if c) if f is not None] for g in groups]
    fps = [g for g in fps if g]
    if len(fps) < 2:
        return float("nan")
    within = float(np.mean([_mean_pairwise(g) for g in fps if len(g) > 1] or [1.0]))
    rng = np.random.default_rng(0)
    across = []
    for _ in range(min(500, len(fps) * 10)):
        i, j = rng.choice(len(fps), size=2, replace=False)
        across.append(DataStructs.TanimotoSimilarity(
            fps[i][rng.integers(len(fps[i]))], fps[j][rng.integers(len(fps[j]))]))
    across = float(np.mean(across))
    return {"within": within, "across": across,
            "across_within_ratio": across / max(within, 1e-9)}


def fragment_usage(fragment_ids, n_fragments):
    """Effective fragment vocabulary, the diagnostic the representation gives us.

    String models cannot ask this question.  Collapse should show as the policy
    drawing on a shrinking set of fragments, which is also where a REINVENT-style
    diversity filter would intervene.
    """
    counts = Counter(int(i) for i in fragment_ids)
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    entropy = float(-(p * np.log(p)).sum())
    return {
        "n_fragments_used": len(counts),
        "fragment_entropy": entropy,
        # Perplexity: the size of a uniform vocabulary with the same entropy, so
        # it is comparable across pool sizes in a way raw entropy is not.
        "effective_vocab": float(np.exp(entropy)),
        "fragment_coverage": len(counts) / max(n_fragments, 1),
    }


def policy_entropy(logits, mask=None):
    """Mean entropy of the per-variable predictive distribution.

    The earliest warning in the RLVR literature: entropy falls in the first
    phase of training and keeps falling, well before accuracy shows anything.
    """
    import torch

    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return float(ent[mask].mean() if mask is not None else ent.mean())
