"""The three RL arms, side by side so swapping one is a config change.

| arm | what it optimises | why it is in the comparison |
| --- | --- | --- |
| `grpo` | trajectory log-likelihood surrogate, signed group-normalised advantages | the baseline the claim is made against |
| `dmpo` | forward-KL to a reward-tilted target defined *on the group* | cheapest retrofit; mode-covering |
| `c_dtm` | state-level local posteriors under progressive reward tilting | likelihood-free; built for this model class |

**The mechanism that separates them is worth being precise about**, because it is
the same mechanism the RLVR literature identifies as the cause of collapse.

GRPO's advantages are *signed*: `(r - mean) / std`. A correct solution that
scores below the group mean is actively pushed down. That is the concrete form
of "indifferent to how probability mass is distributed among correct solutions" —
mass concentrates on whichever correct mode happened to score highest, and the
concentration is self-reinforcing because the next group is drawn from the
sharpened policy.

DMPO's weights are a **softmax over the group**, so every member is positive and
the loss is a weighted maximum-likelihood fit rather than a push-pull. It never
suppresses a correct solution, it only allocates mass among them. That is the
mode-covering behaviour, and it is why this is a loss-function change rather than
an architecture change: it consumes exactly the group GRPO already samples.

c-DTM does not form a sequence likelihood at all. Method D names the obstacle —
the marginal probability of a generated graph needs a sum over every path
reaching it — and GRPO answers it with a surrogate and variance. DTM matches
local unmasking posteriors instead, which is well defined per state. Our node
prior is `mask`, so this is our setting with fragment slots in place of tokens.

**c-DTM also makes structured credit assignment native rather than grafted on.**
Its target is per-variable, with the reward entering as a factor on the realised
value at each variable independently, so a per-fragment reward is the objective's
own form. Under GRPO the same idea must be added by hand as
`A_i + kappa * (S_v - S_bar)`, with `kappa` trading global against local. Here
there is no `kappa`.

Adapted from `Graph-GRPO`, `DMPO/DMPO/dmpo_trainer.py` (arXiv:2510.08233) and
`dtm/dtm/lightning_module.py` (arXiv:2604.18739).
"""

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# arm 1: GRPO
# --------------------------------------------------------------------------


def grpo_advantages(rewards, eps=1e-4):
    """Signed, group-normalised. `rewards` is [n_groups, G]."""
    mu = rewards.mean(dim=-1, keepdim=True)
    sd = rewards.std(dim=-1, keepdim=True).clamp_min(eps)
    return (rewards - mu) / sd


def grpo_loss(logp, logp_old, advantages, clip=0.2):
    """Clipped surrogate on the trajectory log-likelihood.

    logp / logp_old: [n_groups, G] trajectory log-probabilities
    """
    ratio = (logp - logp_old.detach()).exp()
    a = advantages.detach()
    return -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()


# --------------------------------------------------------------------------
# arm 2: DMPO
# --------------------------------------------------------------------------


def dmpo_weights(rewards, log_ratio=None, alpha=1.0, coeff=1.0):
    """Group-level target distribution, normalised over the G samples.

        w_i  proportional to  exp(coeff * (log[p_pre/p_cur]_i + r_i / alpha))

    `log_ratio` corrects for sampling from the current policy when the target is
    anchored to the pretrained one. It is optional here in a way it is not for
    dLLMs: our trajectory likelihoods are intractable, and passing None drops to
    a pure reward softmax, which needs no likelihood at all. That is the cheap
    retrofit — it consumes the group GRPO already samples and nothing else.

    Returns weights summing to 1 per group, and the effective sample size, which
    is the diagnostic for whether the target has itself collapsed onto one
    member (ESS near 1/G) or stayed spread (near 1).
    """
    logits = rewards / alpha if alpha > 0 else rewards
    if log_ratio is not None:
        logits = logits + log_ratio
    w = (coeff * logits).softmax(dim=-1)
    ess = 1.0 / (w.square().sum(dim=-1) * w.size(-1))
    return w, ess


def dmpo_loss(per_sample_ce, weights):
    """Weighted maximum likelihood against the group target.

    per_sample_ce: [n_groups, G] negative log-likelihood of each sampled
    trajectory under the current policy, already normalised by its own length.
    """
    return (per_sample_ce * weights.detach()).sum(dim=-1).mean()


# --------------------------------------------------------------------------
# arm 3: c-DTM
# --------------------------------------------------------------------------


def c_dtm_target(old_probs, x1_onehot, reward, h, c=1.0):
    """The c-DTM cross-entropy target (Prop 3.3; c = 1 by default).

        target = c * pi_a(. | x_t) + onehot(x1) * (1 - c + expm1(h * r))

    `reward` broadcasts against the variable axis, so it may be

    * [n] — one scalar per sample, ordinary global reward; or
    * [n, V] — **per variable**, the oracle's per-fragment consistency score
      landing on the variable it refers to, with no aggregation step anywhere.
    """
    r = torch.as_tensor(reward, dtype=old_probs.dtype, device=old_probs.device)
    while r.dim() < x1_onehot.dim() - 1:
        r = r.unsqueeze(-1)
    bump = (1.0 - c) + torch.expm1(h * r)
    return c * old_probs + x1_onehot * bump.unsqueeze(-1)


def c_dtm_loss(curr_logits, old_logits, x1, reward, weights, h, c=1.0):
    """Weighted cross-entropy against the tilted target.

    curr_logits: [n, V, K] student predictions of the clean value per variable
    old_logits:  [n, V, K] frozen teacher pi_a
    x1:          [n, V] realised clean values from the rollout
    weights:     [n, V] which variables to score — the masked ones, since a
                 variable already revealed at x_t carries no signal
    """
    k = curr_logits.size(-1)
    x1_onehot = F.one_hot(x1.long(), num_classes=k).to(curr_logits.dtype)
    target = c_dtm_target(old_logits.softmax(-1).detach(), x1_onehot, reward, h, c)
    per_var = -(target * curr_logits.log_softmax(-1)).sum(-1)
    w = weights.to(per_var.dtype)
    return ((per_var * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)).mean()


class TiltSchedule:
    """Anneal the tilt a from 0 to a_max in steps of h.

    Aiming at the fully tilted distribution in one move is what the DTM authors
    identify as the collapse risk; each phase is instead a small step from the
    policy that is already in hand, with the teacher refreshed to the student at
    every phase boundary.

    This is a different defence from KL, which bounds how far the policy drifts
    but concentrates mass by construction (proposal section 15).
    """

    def __init__(self, h=0.1, a_max=1.0, steps_per_phase=100):
        self.h, self.a_max, self.steps_per_phase = h, a_max, steps_per_phase
        self.a, self._step = 0.0, 0

    def step(self):
        """True at a phase boundary, where the teacher should be refreshed."""
        self._step += 1
        if self._step % self.steps_per_phase:
            return False
        self.a = min(self.a + self.h, self.a_max)
        return True

    @property
    def done(self):
        return self.a >= self.a_max


ARMS = ("grpo", "dmpo", "c_dtm")
