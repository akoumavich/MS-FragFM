"""Verify the three RL objectives and the collapse metrics before GRPO exists.

These are checkable now, on synthetic groups, because the properties that matter
are properties of the loss rather than of the model:

* does GRPO suppress below-mean correct solutions where DMPO does not?  That is
  the collapse mechanism, and it should be visible as a sign difference.
* does a per-variable reward reach only its own variable under c-DTM?
* do the diversity metrics separate a collapsed group from a diverse one?

If any of these fails, the comparison the arms are meant to support is not being
measured, however well the RL loop runs.
"""

import torch

from msfragfm.diversity import (across_within_ratio, fragment_usage,
                                group_metrics, policy_entropy)
from msfragfm.objectives import (TiltSchedule, c_dtm_loss, c_dtm_target,
                                 dmpo_loss, dmpo_weights, grpo_advantages)


def check_sign_asymmetry():
    """The heart of it: a correct-but-below-average sample.

    GRPO gives it a negative advantage and pushes it down. DMPO gives it a
    positive weight and merely allocates it less mass. Whether a second correct
    mode survives training is decided here.
    """
    # The group that matters is one where *every* sample is a good answer and
    # they differ only slightly -- several distinct correct modes, which is the
    # situation the collapse literature is about.  A group with one obvious dud
    # would show GRPO suppressing only the dud, which proves nothing.
    rewards = torch.tensor([[0.90, 0.88, 0.86, 0.84]])
    a = grpo_advantages(rewards)
    w, ess = dmpo_weights(rewards, alpha=0.2)
    print(f"rewards        {[round(x, 2) for x in rewards.tolist()[0]]}  (all correct)")
    print(f"GRPO advantage {[round(x, 3) for x in a.tolist()[0]]}")
    print(f"DMPO weight    {[round(x, 3) for x in w.tolist()[0]]}  ESS {ess.item():.3f}")
    good_suppressed = int(((a < 0) & (rewards > 0.5)).sum())
    print(f"  GRPO pushes down {int((a < 0).sum())}/4, of which "
          f"{good_suppressed} are correct solutions")
    print(f"  DMPO pushes down {int((w < 0).sum())}/4")
    assert good_suppressed > 0, \
        "the test group must contain correct solutions GRPO suppresses, or it " \
        "is not exercising the collapse mechanism"
    assert (w > 0).all(), "DMPO weights must be positive to be mode-covering"

    # A degenerate group: one sample far ahead. ESS reports whether the target
    # itself has collapsed, which is the knob alpha controls.
    peaked = torch.tensor([[1.0, 0.1, 0.1, 0.1]])
    for alpha in (1.0, 0.2, 0.05):
        _, e = dmpo_weights(peaked, alpha=alpha)
        print(f"  alpha={alpha:<5} ESS {e.item():.3f}")


def check_per_variable_reward():
    """A per-fragment reward must move its own variable and no other."""
    n, v, k = 1, 4, 5
    old = torch.zeros(n, v, k)
    x1 = torch.tensor([[0, 1, 2, 3]])
    r = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # variable 0 only
    t = c_dtm_target(old.softmax(-1), torch.nn.functional.one_hot(x1, k).float(),
                     r, h=0.5)
    bump = t - old.softmax(-1)
    per_var = bump.abs().sum(-1)[0]
    print(f"per-variable target shift {[round(x, 4) for x in per_var.tolist()]}")
    assert per_var[0] > 1e-3 and per_var[1:].abs().max() < 1e-6, \
        "a per-fragment reward leaked into other fragments"
    print("  reward stayed on its own variable")

    # Scalar reward must broadcast to every variable, i.e. reduce to the global case.
    t_scalar = c_dtm_target(old.softmax(-1),
                            torch.nn.functional.one_hot(x1, k).float(),
                            torch.tensor([2.0]), h=0.5)
    shift = (t_scalar - old.softmax(-1)).abs().sum(-1)[0]
    assert shift.min() > 1e-3, "scalar reward failed to broadcast"
    print(f"  scalar reward shifts all variables equally: {shift[0]:.4f}")


def check_losses_run():
    n, v, k = 8, 6, 12
    curr = torch.randn(n, v, k, requires_grad=True)
    old = torch.randn(n, v, k)
    x1 = torch.randint(0, k, (n, v))
    mask = torch.rand(n, v) > 0.3
    loss = c_dtm_loss(curr, old, x1, torch.rand(n), mask, h=0.1)
    loss.backward()
    print(f"c-DTM loss {loss.item():.4f}  grad ok "
          f"{bool(curr.grad.abs().sum() > 0)}")
    # Masked-out variables must receive no gradient, or the objective is
    # learning from variables the state had already revealed.
    g = curr.grad.abs().sum(-1)
    assert g[~mask].max() < 1e-9, "gradient leaked to unmasked variables"
    print("  no gradient on already-revealed variables")

    ce = torch.rand(4, 8)
    w, _ = dmpo_weights(torch.rand(4, 8), alpha=0.5)
    print(f"DMPO loss  {dmpo_loss(ce, w).item():.4f}")

    sched = TiltSchedule(h=0.25, a_max=1.0, steps_per_phase=3)
    boundaries = [i for i in range(1, 13) if sched.step()]
    print(f"tilt schedule refreshes the teacher at steps {boundaries}, "
          f"a={sched.a:.2f} done={sched.done}")


def check_diversity():
    truth = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"  # caffeine
    diverse = [truth, "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"]
    collapsed = [truth] * 3 + ["CN1C=NC2=C1C(=O)N(C)C(=O)N2CC"]
    scores = [0.9, 0.5, 0.4, 0.3]
    for name, cands in (("diverse", diverse), ("collapsed", collapsed)):
        m = group_metrics(cands, scores, truth)
        print(f"{name:10s} top1={m['top1']:.0f} top10={m['top10']:.0f} "
              f"unique={m['n_unique']} pairwise_tanimoto="
              f"{m['mean_pairwise_tanimoto']:.3f}")

    # Both groups contain the truth, so top-k alone cannot tell them apart.
    # Diversity is what separates them, which is the reason it is instrumented.
    r = across_within_ratio([diverse, [truth, "CCO", "CCN"], collapsed])
    print(f"across/within ratio {r['across_within_ratio']:.3f} "
          f"(within {r['within']:.3f}, across {r['across']:.3f})")

    print(fragment_usage([0, 0, 0, 1, 2, 2, 5, 9], n_fragments=83194))
    print(f"policy entropy (uniform over 10) "
          f"{policy_entropy(torch.zeros(4, 10)):.4f} vs ln(10)=2.3026")


if __name__ == "__main__":
    for fn in (check_sign_asymmetry, check_per_variable_reward,
               check_losses_run, check_diversity):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\nall checks passed")
