# PROGRESS

Append-only log. Newest entry at the top.

---

## 2026-09-24 — Bag size ruled out. Cross-attention and exposure fix built; retraining next.

**Fragment recall is flat in bag size**: 0.2839 at 384, 0.2768 at 1024, 0.2734 at
2048. Widening the bag 5.3x changes nothing, so the 74% availability ceiling does
not bind -- it cannot, with recall at 28.5%. RESULTS.md R25.

That closes the last non-model explanation for the 80%-teacher-forced to
28.5%-generated gap. Everything remaining is training-side.

**Built, both training-side so one run tests both:**

*Cross-attention.* The encoder now exposes per-token memory -- peaks, one token
per element carrying its count as Method A specifies, then adduct, instrument and
precursor -- and the coarse GNN attends to it once before the backbone,
residually so an untrained attention is a no-op. 25.7M params, 726s/epoch against
662s.

*Exposure.* `frag_mask` guaranteed the molecule's own fragments were selectable at
every training step. With probability 0.25 that is now withheld, and the loss
skips nodes whose target became unselectable, since asking a model to predict what
it cannot choose gives an infinite cross-entropy.

R25 slightly weakens the second on its own terms: if availability does not bind
at generation, coping with absence matters less than the argument assumed. Still
worth testing -- training on a systematically easier task than the real one is a
defect either way -- but cross-attention is now the stronger hypothesis.

**Two backward-compatibility faults, one caught and one not.** I gated the
flow-side attention on its own flag so the existing checkpoint would still load,
then added modules to the encoder and broke loading there instead. The bag sweep
failed silently because my grep filter matched only the lines I expected, so a
traceback looked identical to a null result -- the second time that has happened.
Error patterns now go in the filter.

---

## 2026-09-24 — Diagnosis complete: 28.5% fragment recall. The model, not the pipeline.

**Fragment recall 28.5% mean, 53.5% best of 16, never all.** The model recovers
about 2 of 7 fragments. RESULTS.md R23, R24.

At n=2,000 top-1 = 0 became a measurement rather than an absence of resolution:
true accuracy below 0.15% at 95%. R23 predicted from that per-fragment accuracy
must be below ~37%; measured 28.5%, so the inference chain closes.

The signal is real -- 109x random selection from a 384-fragment bag -- and the gap
is large: teacher-forced perplexity 1.53 implies 80%+. Two of seven, not six of
seven, so no improvement to assembly or reranking reaches top-1 from here.

Tanimoto could not have told us this. 0.253 is consistent with "nearly right" and
with "mostly wrong", and it is mostly wrong. Worth remembering for the paper: the
benchmark's own similarity metrics hide the distinction that matters.

**The pipeline work is done and was worth doing.** Validity 99.9%, connectivity
94%, heavy-atom exact 74%, under one atom lost in assembly -- from a starting
point where 65% of molecules were being silently truncated. But none of it moved
structure identification, and I should say plainly that four of the five fixes
since R18 addressed symptoms of one mechanical fault.

**The most useful number is the spread.** Mean 28.5% against best-of-16 53.5%:
the group holds nearly twice as much of the answer as its average member. That
argues for fragment-level test-time search -- a scorer over the group assembling
something no single sample contains -- and the R19 peak explanation is an
oracle-free scorer already in hand. It also bears on C2, since structured credit
assignment acts precisely on that within-group variation.

**Three candidates for the gap, in the order I would test them:**

1. **Conditioning is one pooled vector.** Method A specifies per-element
   embedding injected by cross-attention; we deliver formula and peaks as a single
   256-d global vector shared across all seven slots. Identifying which fragment a
   peak implies is per-node retrieval, which is what cross-attention is for. R20
   flagged the deviation; it now has a measured cost.
2. **Training never shows a missing answer.** `frag_mask` always includes the
   molecule's own fragments, so the true fragment is guaranteed available at every
   training step. At generation it is available 74% of the time. Exposure bias
   with a specific, fixable cause.
3. **Neither arm converged** -- both losses still falling at epoch 50.

---

## 2026-09-24 — Valency constraint lands. Pipeline mechanically sound; model quality is next.

**Connectivity 38% to 93%, assembly atom loss 8.17 to 1.04, exact heavy-atom
count 26% to 74%, validity 99.94%.** RESULTS.md R22.

The prediction held exactly: a 14% per-node valency mismatch was compounding over
~7 nodes into a 62% molecule-level failure, and enforcing `junction_count ==
degree_tree(v)` in the support closed it. FragFM computed that mismatch as an
input feature all along and never constrained it.

The heavy-atom shortfall has changed hands. Assembly was 79% of it and is now
26%; fragment selection at -2.97 atoms is the dominant residual.

A detail I did not expect: constraining the choice made the model use a **wider**
fragment range, not narrower -- effective vocabulary 72.0 to 76.8 -- because the
valency requirement pushes it off the small common fragments R20 found it
defaulting to.

**The residual 6.7% disconnection is a bag problem.** 0.77% of nodes still take a
wrong-valency fragment because none with the right valency was in the drawn
384-fragment bag, and the guard keeps their unconstrained options rather than
producing a NaN softmax. Drawing the bag with required valencies in mind is the
fix -- the spectrum-conditioned bag arriving from a third direction.

**What has not moved is the point.** Tanimoto to truth changed by 0.0006.
Everything fixed since R18 was mechanical: the pipeline was destroying its own
output and has stopped. None of it made the model better at identifying a
structure. Four support constraints and what each bought:

| constraint | bought |
| --- | --- |
| formula containment | 55% of pool pruned; almost nothing downstream |
| spanning-tree coarse decode | validity 93% -> 99.5% |
| composition projection | formula match 6.9x, exact count 5.9x |
| **fragment valency** | **connectivity 38% -> 93%, assembly loss 8x** |

**Top-1 is not yet measurable.** Zero in 200 spectra bounds true accuracy below
1.49% at 95%, against a field at 18%. At 2,000 it bounds below 0.15%, which would
be informative. That is the first evaluation worth running at scale, and the
caching plus vectorisation make it affordable.

**After that, three candidates for model quality, in order of expected value:**
formula conditioning by cross-attention as Method A specifies rather than one
component of a global vector (R20 flagged the deviation and it now has a measured
cost); more training, since neither arm had converged at epoch 50; and the
attachment latent of R7-R11, which decides *which atoms* bond rather than which
fragments, and is now isolated from the connectivity failure that was masking it.

---

## 2026-09-23 — The bottleneck is atom-level assembly. Four hypotheses retracted.

**Only 35.3% of assemblies come out in one piece**, 2.37 components on average,
and 79% of the heavy-atom shortfall is mass discarded at assembly rather than
never chosen. RESULTS.md R21.

`reconstruct_to_rdmol(get_largest=True)` keeps the largest connected component
and the following "no dot in the SMILES" assert then passes, so **a broken
assembly is recorded as a valid molecule**. Validity 0.991 with top-1 zero: we
have been measuring successful truncation as success.

**The chain of hypotheses I ran through, and where each died:**

1. the bag does not offer the right fragments — refuted (R16, 74% availability)
2. the bag offers fragments that are too small — refuted (R20, it offers 9.52
   heavy atoms where 4.15 is needed)
3. the model picks fragments that are too small — true but minor, 21%
4. composition is unconstrained — fixed, and it made the signed error *worse*,
   because larger chosen fragments mean more mass to lose

Each was measured and each was wrong or secondary, and the answer was downstream
of all of them. What I should have noticed sooner: R13 measured the autoencoder
at 97.9% *on ground-truth coarse graphs with the encoded latent*, and generation
supplies neither. That gap was visible from R13 onward.

**The mechanism.** We enforce the coarse graph is a tree, so fragments are
connected *there*. Atom-level bonds come from Blossom max-weight matching over
junction slots, and a matching is not required to realise the coarse tree — it
can bond A-B twice and leave A-C unbonded, or pair non-adjacent fragments. The
spanning-tree constraint bought connectivity at the coarse level and nothing at
the atom level, which is where the molecule is.

**The fix is exact and the bookkeeping already supports it.** A fragment's
`junction_count` is the number of cut bonds incident to it, which is exactly its
degree in the coarse tree. So assigning a fragment's slots to its incident coarse
edges is a perfect matching, per fragment, at most 4x4. Then the atom graph
realises the coarse tree by construction and connectivity cannot fail.

This is the per-coarse-edge formulation R11 set aside because rBRICS breaks it
(7.7% two-bond edges). R13 then chose BRICS, where 100.00% carry exactly one —
so it was right to set aside at the time and wrong to leave aside afterwards.

**Two things it also explains.** Generated candidates explain 6.5% of peak
intensity against 35.4% for true structures, because truncated molecules cannot
explain evidence. And peak ranking changed top-k not at all, which is expected:
a reranker cannot find an answer that is not in the candidate set.

**Speed, meanwhile:** fragment-pool embeddings cached (107s -> 2s per run) and the
composition beam search vectorised (20.5 -> 2.5 ms/molecule, 8x, verified
behaviour-preserving on identical parameters).

---

## 2026-09-23 — Method C moves into paper one. Peak explanation separates at d=1.32.

Conservation of composition, computed exactly at fragment level. 300 test
spectra, 20 ppm, size-matched decoys. RESULTS.md R19.

| max cuts | candidates/mol | true | decoy | separation | Cohen's d |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **20.6** | 0.3544 | 0.0092 | **+0.345** | **1.32** |
| 3 | 271.1 | 0.4178 | 0.0233 | +0.395 | 1.44 |

**The law separates true from decoy with no oracle, no training and no learned
model** — arithmetic on the conservation law. True beats decoy on 75% of spectra
at one cleavage.

**Two corrections to my own cost claim, in sequence.** I said fragment-level
enumeration was trivial because 2^7 = 128. That used 2^(mean k) where the cost
is mean(2^k) — measured, all connected subtrees average **4,062,211** per
molecule, since the test fold reaches k ~ 20. Averaging the exponent instead of
the exponential hid the tail, and the script I wrote printed the misleading
comparison rather than the real one.

The fix is also the chemistry. MAGMa and ICEBERG model one to three bond
cleavages, not arbitrary connected subgraphs, and on a tree cutting j edges gives
exactly j+1 components — so bounding cuts turns 2^k into O(k^d). One cut is
**197,195x cheaper than full enumeration** and keeps 87% of the separation.
Returns collapse after that: 1->2 buys +0.041 for 3.9x cost, 2->3 buys +0.008 for
3.3x.

Enumerating everything was therefore both expensive *and* chemically wrong: the
decoy score rises with depth (0.0092 -> 0.0233) because more candidates mean more
accidental explanations. My original 0.436 was inflated on both arms.

**This is the missing signal from R18, in its most literal form.** Generated
molecules run 6.97 heavy atoms short because the occurrence-weighted bag favours
small common fragments and nothing told the model how big the pieces should be.
At one cleavage the candidates are exactly the two pieces a molecule splits into
at a single coarse bond, so the peaks constrain those sizes directly.

**Proposal updated: Method C's exact form moves into paper one.** Three uses
follow — rerank with no oracle call, select the bag from evidence rather than
occurrence (which is what Method A asks for), and supply a reward term immune to
the oracle-leakage problem the audit flags, since it is arithmetic and not a
learned simulator. The differentiable relaxation stays paper two's contribution
for what the exact version cannot do: flow gradients into the edge tensor, and
reach atom-level cleavages where enumeration really is exponential.

**A limit worth stating before a reviewer does.** The true structure explains
only 35-42% of peak intensity, and the remainder is structural rather than noise:
real fragmentation also breaks bonds *inside* BRICS fragments, which the coarse
graph cannot express. So this is a relative score between candidates, not an
absolute goodness of fit.

---

## 2026-09-23 — First de novo eval. Conditioning works; my bag hypothesis was wrong.

Top-1 is 0 on 200 test spectra. Two findings behind that, one of them a
retraction. RESULTS.md R16.

**Conditioning reaches generation, confirmed three ways.** Real vs shuffled vs
zero conditioning, same model, same everything else:

- formula match **12x higher** with the right spectrum than a wrong one (0.0036
  vs 0.0003), and zero with none;
- max Tanimoto to truth **+0.084** for the right spectrum over a wrong one;
- across/within ratio **0.99 at zero conditioning** -- samples for different
  spectra as alike as samples for the same one, i.e. not conditioning -- falling
  to 0.66 once any spectrum is present.

The third is the E8 instrument doing exactly its job, and `shuffled` matching
`real` there is correct rather than a null result: it still gets *a* spectrum, so
it differentiates just as much, only toward the wrong target. That is the
separation the metric was built for.

**Retraction: the fragment bag is not the bottleneck.** I predicted in R4, and
asserted here, that a 384-draw from an 81,739-fragment pool would rarely offer a
target's fragments. Measured on the real occurrence weights: 0.742 per fragment
per step against my estimated 0.0047, and a reachability bound on top-1 of 0.567
against my estimated 0.001. **My arithmetic assumed uniform draws; the bag is
occurrence-weighted**, which concentrates its 384 slots on exactly the fragments
real molecules are built from. I had the weights in hand and used a uniform
approximation anyway, then stated the conclusion with more confidence than the
estimate deserved.

**The real diagnosis inverts mine.** The bag does not fail to *offer* the right
fragments, it fails to *exclude* the wrong ones. 99.6% of candidates carry a
formula the target cannot have, and every one was knowable as impossible before
being sampled. Proposal section 9 says the support is the strongest place for an
exact constraint -- "mask the sampler so violations are unreachable, exact, free,
no tradeoff against likelihood" -- and **we have none of it**; the formula is a
soft encoder input and nothing else. The audit's thirtyfold formula-pruning
result is the same observation from the other side.

Same fix as R4 proposed, same code, opposite mechanism: prune the impossible
rather than include the necessary.

**On top-1 = 0.** Fragment perplexity 1.53 is teacher-forced, one variable at a
time; generation composes ~8 such decisions from a masked start with errors
compounding, which puts all-correct near 15% at 80% per-fragment accuracy and
near 5% at 70%. 200 spectra cannot resolve below ~1.5%. DiffMS reports 0% and
MADGEN 1.31% on this benchmark, so this is where unconstrained graph models sit
before constraints go in.

**Next:** the formula constraint in the support. `frag_mask` already exists in
the generator as a per-node mask over the current bag, so the hook is there.

**To check:** the reachability diagnostic reports 8.6 fragments per test
molecule, but R2 measured BRICS at 7.06 and *rBRICS* at 8.65. Probably the test
fold holding larger molecules, since it is MCES-disjoint rather than a random
sample, but close enough to the rBRICS figure to verify rather than assume.

---

## 2026-09-23 — E3 trained. Conditioning holds and strengthens. Resume earned itself.

50 epochs each, both arms, ~9 h in parallel. RESULTS.md R15.

**Fragment-type loss 0.4266 conditioned against 0.7310 blind, a 41.6%
reduction** — effectively choosing among 1.53 fragments rather than 2.08.

**The gap widens monotonically**: -0.136 at epoch 2, -0.250 at 10, -0.285 at 20,
-0.304 at 50. The model leans on the spectrum more as training proceeds, not
less. Conditional generative models quietly falling back on the marginal is
common enough to be worth ruling out, and monotone widening rules it out.

The decomposition matches what R14 predicted from a single epoch: fragment type
and latent both improve and keep improving, coarse edge is flat throughout. The
spectrum says which fragments are present and how their atoms sit, and says
nothing direct about which fragments bond to which. A signal that had improved
all three equally would have been suspicious rather than reassuring.

Neither arm has converged — both still falling ~0.009 per five epochs, in
parallel, so more training improves both without obviously changing the
comparison.

**The resume machinery was exercised rather than merely tested.** The spectrum
arm was preempted and resumed **four** times, the blind arm three. Loss is
continuous across every boundary and the restored iteration counts are consistent
(2,271 = 3 x 757). Without the checkpoint work these two runs would have
restarted from scratch seven times between them and never finished. That was
worth the detour.

**Correction to my own verification.** `check_objectives.py`'s sign-asymmetry
test used rewards `[0.9, 0.7, 0.6, 0.1]`, where only the obvious dud gets a
negative advantage — so it asserted `(a < 0).any()`, which passed trivially and
demonstrated nothing. The mechanism at issue is GRPO suppressing *correct*
solutions that merely score below the group mean, which needs a group where every
sample is good and they differ slightly. With `[0.90, 0.88, 0.86, 0.84]` GRPO
assigns `[+1.162, +0.387, -0.387, -1.162]` and pushes down two of four correct
solutions, while DMPO's weights are `[0.289, 0.261, 0.236, 0.214]`, ESS 0.988.
The test now asserts that correct solutions are suppressed, so it fails if the
group stops exercising the mechanism.

**These are training losses.** They show the conditioning path carries
information and is used. They say nothing about top-1 or top-10 on the test fold,
which needs generation — the next measurement, and the first one comparable to
anything in the literature.

---

## 2026-09-22 — E9 added: the objective is an arm. Collapse instruments built.

Read both repos rather than working from the abstracts, and the distinction
between the two methods is sharper than "alternatives to GRPO".

**DMPO** (`DMPO/DMPO/dmpo_trainer.py`, arXiv 2510.08233) builds a group-level
target proportional to reward and fits it by approximate forward-KL. The whole
mechanism is one line: GRPO's advantages are signed, `(r - mu)/sigma`, so a
**correct solution scoring below the group mean is actively pushed down**. That
is the RLVR literature's "indifference to how mass is distributed among correct
solutions" in concrete form. DMPO's weights are a softmax over the group, all
positive, so it allocates mass among correct solutions without suppressing any.
Cheapest possible retrofit: it consumes exactly the group GRPO already samples.
Its likelihood-ratio term is optional for us -- dropping it leaves a pure reward
softmax needing no trajectory likelihood at all, which matters because ours is
intractable.

**c-DTM** (`dtm/dtm/lightning_module.py`, arXiv 2604.18739) is built for this
model class. Method D concedes that the marginal probability of a generated graph
needs a sum over every path reaching it, and answers with a high-variance
surrogate. DTM removes the surrogate instead of taming it: likelihood-free
matching of local unmasking posteriors under progressive reward tilting. Our node
prior is `mask`, so this is our setting with fragment slots in place of tokens.

**The finding that changes the thesis rather than the recipe.** c-DTM's target is
per-variable, and the reward enters as a factor on the realised value at each
variable *independently*. So a per-fragment oracle score is the objective's own
form. Under GRPO the same idea has to be grafted on as `A_i + kappa(S_v - S_bar)`
with kappa trading global against local. **Under c-DTM there is no kappa.**

C2 restated accordingly: structured credit assignment *plus distribution
matching*, on a representation where both are natural. The fragment level is what
lets the oracle's per-fragment scores land without aggregation; c-DTM is what
lets them land without a mixing coefficient. Stronger and more coherent than
structured credit assignment alone, and it answers the reviewer who knows this
literature, for whom "we used KL" is not an answer.

**Built** (testable now, before GRPO exists, because these are properties of the
loss rather than of the model):

- `msfragfm/objectives.py` -- all three arms side by side so swapping is a config
  change: `grpo`, `dmpo`, `c_dtm` plus `TiltSchedule`.
- `msfragfm/diversity.py` -- E8 instruments: joint top-1/top-10, unique valid
  molecules per spectrum, mean pairwise Tanimoto within group, policy entropy,
  fragment-usage entropy and effective vocabulary, and the across/within ratio.
  Fingerprint stated explicitly (Morgan, radius 2, 2048 bits) since the v1.5
  audit flags that these diverge across papers.
- `scripts/check_objectives.py` -- verifies the sign asymmetry between GRPO and
  DMPO, that a per-fragment reward reaches only its own variable under c-DTM,
  that no gradient leaks to already-revealed variables, and that the diversity
  metrics separate a collapsed group from a diverse one where top-k cannot.

The last of those is the point of the diversity instruments in miniature: both
test groups contain the truth, so top-k alone scores them identically.

---

## 2026-09-22 — E8 added: mode collapse under GRPO, and a correction to Method D

Checked the literature rather than assuming. Mode collapse under policy-gradient
RL is **the reported default in three separate literatures**, not an edge case:

- **RLVR on LLMs.** GRPO raises Pass@1 while degrading Pass@k; entropy falls
  early and keeps falling. The mechanism transfers verbatim: the objective is
  *indifferent to how probability mass is distributed among correct solutions*,
  so mass concentrates on a narrow subset of them, self-reinforcingly.
- **RL on diffusion.** DDPO samples cluster around the highest-scoring outputs.
- **RL on molecular generators.** Known since REINVENT, and standard equipment
  rather than a finding: diversity filters with scaffold memory buckets.

**A correction to Method D.** The proposal said KL regularisation to the base
policy "prevents collapse". It does not. KL-regularised RL is variational
inference against a Gibbs posterior, which concentrates mass on high-reward
regions by construction -- forward and reverse KL both *induce* collapse rather
than resist it. KL bounds how far the policy drifts, which is a real defence
against reward hacking and forgetting, but it says nothing about spread at the
destination. Corrected, and collapse now has its own instrument.

**Why this threatens C2 rather than just sample quality.** The headline claim is
an accuracy-versus-oracle-calls curve that shifts left. A collapsed policy
returns near-duplicates within a group, so extra oracle calls buy nothing and the
curve **flattens instead of shifting**. Collapse can raise top-1 and destroy the
comparison the paper rests on simultaneously.

**Kept separate from reward hacking**, because each can hide the other. Reward
hacking is a high oracle score on a wrong molecule, caught by the held-out slice.
Collapse is loss of spread, invisible to that slice whenever the collapsed mode
happens to be correct.

**The conditional setting inverts the usual diagnostic**, and this is the part
that would have been easy to get wrong. Property optimisation wants diverse hits,
so concentration is bad. Elucidation has one right answer per spectrum, so
concentrating *within* a spectrum is the goal. The failure is concentration
*across* spectra -- the policy emitting the same fragments whatever it is shown,
i.e. having stopped conditioning. A falling within-spectrum diversity number is
therefore ambiguous, and the informative quantity is the ratio of across-spectrum
to within-spectrum variation. The spectrum-blind arm already gives the reference
value for that ratio.

**E8 instruments**, logged from the first RL step rather than added once
something looks wrong: policy entropy per generative variable; unique valid
molecules per group; top-k versus k against the base policy; accuracy versus
oracle calls against the base policy; and fragment-usage distribution against the
base, which is a diagnostic the representation gives us and string models cannot
-- collapse should show up as the effective fragment vocabulary shrinking.

Mitigations listed in order of evidence and applied only if the instruments fire:
a fragment-level diversity filter on the REINVENT pattern, entropy
regularisation, DAPO's clip-higher, and the choice of divergence.

No code yet: this lands when GRPO does. E3 is still training.

---

## 2026-09-22 — Decomposition settled: BRICS. Ceiling measured at 0.71

Fine-tuned the released NPGen autoencoder on MassSpecGym for both
decompositions, 60 epochs each (~9 min). Full test fold, Blossom decode:
**BRICS 0.9794, rBRICS 0.8389**. RESULTS.md R13.

Fine-tuning gains 3.0 points over the zero-shot checkpoint and 19.5 over
training from scratch, in a third of the epochs. On 25k molecules,
initialisation dominates everything else available.

**Decomposition: BRICS.** Both factors measured for the first time:

| | coverage | reconstruction | ceiling |
| --- | ---: | ---: | ---: |
| BRICS | 0.725 | 0.9794 | **0.7101** |
| rBRICS | 0.823 | 0.8389 | 0.6904 |

R9 fixed the criterion in advance: rBRICS needed reconstruction above 0.8628 to
pay for its coverage lead. It reached 0.8389.

This is the second reversal of this decision — E0-a chose BRICS on compression
with no coverage data, E0-c reversed to rBRICS on coverage with no reconstruction
data, R13 returns to BRICS with both measured. Each reversal followed a number
that did not exist before it. The cost was two days, and the alternative was
committing to rBRICS on half the evidence and finding out in week 9.

rBRICS reconstructs worse for reasons intrinsic to it: 8.6 fragments per molecule
against 7.1, more junction atoms each, and cut rings joining a fragment pair
twice. BRICS also leads on edge-slot compression (14.4x vs 9.0x), so coverage was
the only axis rBRICS ever won.

**The ceiling, stated plainly:** exact top-1 is bounded by **0.71** for this
representation, before the generator, the oracle or the RL are considered. Both
factors are floors — coverage was still climbing steeply at 200k of 4M available
corpus molecules, and the autoencoder is a fine-tune of someone else's chemistry.
Against ~18% SOTA that is roughly 4x headroom, which answers the question R3
raised and mis-framed.

Worth reporting in the paper regardless: no published de novo method states the
bound its own representation imposes.

**Next, and the critical path is now spectrum conditioning.** With speed
retired (R8) and the representation settled, the remaining unbuilt pieces are the
spectrum-conditioned flow (E3) and the sampled-attachment head that gives GRPO
per-atom credit. The second is small and depends on the first, so conditioning
goes first.

---

## 2026-09-22 — The three arms ran. Both of my hypotheses were wrong; the design got simpler

Results in RESULTS.md R12. Full MassSpecGym BRICS test fold, Blossom decode:
BCE+z **0.7845**, atom_ce+z 0.7636, atom_ce with z zeroed 0.4797, and the
released NPGen checkpoint zero-shot **0.9491**.

**Competition does not help, and the reason is instructive.** Per-atom
choose-k lands 2.1 points *below* BCE. Max-weight matching compares candidate
scores **across** atoms in one optimisation, and a per-atom softmax normalises
each atom separately — destroying exactly the comparability Blossom needs. BCE
calibrates each logit against an absolute probability, which is directly usable
as a matching weight. My argument was right about ranking within an atom and
wrong about what the decode consumes. The objective itself worked: atom_ce
plateaus at 0.199 against its 0.119 irreducible floor. It learned what I asked
for; I asked for the wrong thing.

**Training from scratch on MassSpecGym is a mistake.** Every arm is far below the
released checkpoint used zero-shot — 0.78 against 0.95. It saw 658,566 COCONUT
molecules; our train fold is 25,023. BCE's training loss reaches 0.0056 with test
at 0.78, which is overfitting. **Fine-tune, do not retrain** — cheaper and
better, and doing it for both decompositions also settles rBRICS.

**The architecture question is answered, and the answer is in between.** Trained
without z, the model reaches 0.4797 against 0.7636 with it. So ~63% of the
achievable accuracy comes from the coarse graph alone and ~28 points is
information z genuinely carries. Attachment cannot be dropped.

But it is less dire than R11 implied. The shuffled-latent gap was 73 points
because that decoder was *trained with z available* and leaned on it. Trained
without, the model recovers most of the way by itself. The honest measure of
what z uniquely contributes is the arm2-minus-arm3 gap, ~28 points, not 73. I
should have seen that the shuffled test necessarily overstates the case against a
variable the model was trained to depend on.

**A sampling bias in every `--limit 1K` number I have reported.** The test fold
is stored in an order where the first ~1,024 molecules score ~11 points above the
fold average — +11.8, +12.1, +9.7 across the three arms, so it is the data, not
noise. Paired comparisons within one subset (threshold vs Blossom, R11) are
unaffected; absolute levels from 1K runs are optimistic. The `--limit 10K` runs
covered the whole 3,160-molecule fold and stand. Evaluation now shuffles once
with a fixed seed.

**The design got smaller, and the failed experiment is what found it.** Credit
assignment does not need a new training loss — that is what arm 2 tried and what
the matching decode rejects. It needs attachment to be a **sampled** variable
with per-atom log-probabilities, and those two things are separable:

- train with BCE, which the decode rule prefers;
- at generation, sample each atom's partners from a temperature-controlled
  softmax over the BCE logits, then Blossom-project as now;
- the softmax supplies per-atom log-probs for GRPO, and the training objective is
  untouched, so no accuracy is paid for the credit signal.

**Next:** fine-tune the released checkpoint on MassSpecGym for both BRICS and
rBRICS. That gives the real second ceiling factor, settles the decomposition, and
supplies the base the sampled-attachment head sits on.

---

## 2026-09-22 — Discrete attachment: the design, after two wrong turns that the data caught

**My evaluation used the wrong decode rule.** FragFM's generation path never
thresholds attachment scores — it expands junction atoms into slots, runs Blossom
max-weight matching, and contracts back. E0-e used `pred > 0.5`, so R7 and R10
measured the scoring head, not the model that ships. Matching depends on the
*ordering* of scores and a threshold on their absolute values, so the two can
diverge badly. I built the evaluation from FragFM's `eval_ae.py`, which
thresholds because it is measuring the head in isolation — right for their
purpose, wrong for mine.

Paired on the same 1,000 molecules: Blossom lifts every arm, most where scores
are worst (prior 0.066 -> 0.238, shuffled 0.204 -> 0.257, encoded 0.950 ->
0.984). **It does not lift enough.** A wrong-but-valid latent still costs 73
points, 0.984 -> 0.257. **R7 and R10's conclusion survives; their numbers do
not** — 25.7%, not 19.5%. The BRICS ceiling factor is 0.984, so the BRICS
ceiling is 0.725 x 0.984 = 0.713.

**My first design was the wrong granularity, and the check killed it before
training.** Per-coarse-edge categorical — "which atom pair joins fragments A and
B" — is valid for BRICS (100.00% one bond per edge) and invalid for rBRICS
(7.67% carry two). The cause is rBRICS cutting rings: breaking a ring takes two
bonds, so the two fragments are joined twice. A formulation that works for one
decomposition and not the other is the wrong formulation.

**Per atom, choosing `junction_count` partners, is the right one.** Valid by
construction in any decomposition, no label degeneracy (unlike per-slot, where
slots on an atom are interchangeable), and finer than the fragment granularity
the oracle attributes to. 82.3% of junction atoms have count 1, 17.2% count 2 —
choose-k is common enough that it cannot be a special case.

Softmax with k targets is deliberate: its optimum puts 1/k on each true partner,
ranking all k above every distractor, which is exactly what max-weight matching
consumes. BCE cannot express that because it scores each candidate without
reference to its competitors. That competition is the whole content of the
change; **the network is untouched**, since it already emits one logit per
candidate.

**A measurement bug in my own granularity check.** `ae_to_pred_index` stores each
candidate once as (i, j) with i < j, so grouping by column 0 drops every atom
that is always the larger index — it reported 23.9% of atoms having zero true
partners, which is impossible for a junction atom. Symmetrised; true partners per
atom now equals junction count, as it must.

**Built:** `msfragfm/attachment.py` (per-atom choose-k loss),
`msfragfm/blossom.py` (the real decode, reusable), `scripts/train_ae.py` (three
arms), `scripts/check_attachment.py` (the granularity diagnostics, including the
retired coarse-edge grouping, kept because it is the evidence that retired it).

**Next: the three arms.** BCE+z is the fair retrained baseline; atom_ce+z asks
whether competition alone helps; **atom_ce with z zeroed is the one that
decides**. If it recovers atom_ce+z without the latent, attachment was
determined by the coarse graph all along, z goes, and Method D needs no further
work. If not, the choice has to become a variable of the flow rather than an
output of the decoder.

---

## 2026-09-19 — E0 complete. C1 is retired as a contribution; the attachment redesign is on

Week 1's blocking measurements are done, and between them they reshape the plan.

### The speed problem does not exist

Stock FragFM: **0.338 s/sample at 500 steps, 0.068 s at 20**, batch 256 on one
A100. That is 388x DiffMS and 19x FRIGID with nothing built. The GRPO epoch that
the proposal called impossible at 30 days is **23 minutes on 8 GPUs**. The week-6
gate of <2 s/sample was passed in week 1 with 6x margin. (RESULTS.md R8.)

This is good for the project and bad for C1 *as a contribution*. A speedup that
requires no work is an observation about the base model, not a result, and it
cannot carry a section. C1 now appears once, with the measurement, as the premise
that makes C2 possible. The weeks budgeted for speedup work move to C2.

**The step lever is capped at 5.8x, not the 25x budgeted**, because decode is a
flat 52-59 ms/sample floor that step count does not touch — 17% of cost at 500
steps, 90% at 10. Concretely: **do not do the distillation work.** FS-DFM, T3D
and Duo all attack flow-step count, worth ~2x from a sensible operating point.
ReMDM and correctors stay, for quality at low step counts, which is a different
argument. This is the finding the phase split was built to catch, and an
end-to-end number would have hidden it.

**One unbudgeted cost.** Generator startup embeds the whole fragment pool: 4m35s
for 133,823 fragments. Harmless for inference, fatal under GRPO where the
fragment embedder is part of the policy and any update invalidates every
embedding. Freeze the embedder and train only the coarse GNN, or embed the drawn
bag on demand. Decide before writing the RL loop.

### The attachment latent: R7 stands, magnitude corrected

The `shuffled` control — hand the decoder another molecule's *encoded* latent,
in-distribution by construction — gives **19.5%** exact reconstruction against
98.7% with the right one. So z is molecule-specific and carries the attachment
decision.

My prior arm *was* confounded, as suspected: prior draws are 4.4x wider than the
encoder's range (std 3.80 vs 0.869), because the min-max inverse is calibrated
for the flow's t=1 output and I fed it a standard normal. That inflated the
effect from 19.5% to 9.4%. **The conclusion is unchanged; the number was.** The
R9 z-search figures sampled the same bad distribution and are withdrawn.

Read cleanly: for about one molecule in five attachment is determined by the
coarse graph alone; for the other four in five z carries it, and z is a single
global continuous vector with no per-fragment structure. That is the architecture
case, and it arrived in week 1 rather than week 12. Weeks 5-6, freed by the speed
result, now go to discrete attachment sites.

### The decomposition choice has a decidable criterion

The MassSpecGym 64.6% was decomposition mismatch, not chemistry: the same
NPGen-trained autoencoder reconstructs **94.9%** of MassSpecGym zero-shot when
fed BRICS. Transfer across chemistry costs 3.8 points; transfer across
decomposition costs 30.

Both ceiling factors multiply: BRICS gives 0.725 x 0.949 = **0.688**, rBRICS
gives 0.823 x p. **rBRICS wins iff a retrained autoencoder reaches p > 0.836 on
rBRICS fragments** — not assured, since rBRICS means more fragments and more
junctions and attachment is the harder half of the job. Retraining on each
settles it, and the autoencoder is being retrained regardless.

### Next

1. Retrain the autoencoder on MassSpecGym, both decompositions -> settles rBRICS
   vs BRICS and gives the real second ceiling factor.
2. Discrete attachment sites.
3. Spectrum conditioning, which is now the critical path rather than speed.

---

## 2026-09-19 — E0-e: z carries attachment almost entirely. R6 confirmed, severe end

**Result.** Coarse graph held at ground truth, only z varied, FragFM's released
NPGen autoencoder. Exact graph reconstruction: **98.7% with the encoded z, 9.4%
with a prior z.** The coarse fragment graph does not determine attachment — z
carries essentially the whole decision. Method D's per-fragment advantages have
nothing to land on for attachment errors. This is the "pile B is large" case,
not the limitation-paragraph case. Full table in RESULTS.md R7.

The noised arms bound the flow's job: accuracy is intact at s=0.25 (97.9%), down
8 points at s=0.5, down 33 at s=1.0. z must be predicted to well under one
standard deviation in AE latent space.

**Correcting my own experiment.** The script's headline line read
"attachment information carried by z: 0.9872 -> 0.0940, drop = 0.8932", which
invites reading 9.4% as a forecast of generator accuracy. It is not. The flow
*evolves* z during sampling — `_calc_euler_step` integrates
`z_rate = (pred_z - gen_z)/(1-t)` — so the z reaching the decoder is a learned
function of the coarse-graph trajectory, not the prior draw it started from. The
prior arm answers "is z redundant" (no); the noised arms answer "how hard is the
flow's job". Headline corrected in the script.

**The MassSpecGym number is confounded and must not be quoted yet.** Encoded-z
accuracy is 64.6% there against 98.7% on NPGen, with two causes entangled: the
checkpoint was trained on BRICS and is being fed rBRICS, and it was trained on
COCONUT natural products and is being applied to broader chemistry. Running
MassSpecGym through BRICS separates them and costs two minutes.

This is the second factor of the ceiling, so it matters: 0.823 x 0.646 = 0.53 if
the zero-shot figure held. It should not — the autoencoder gets retrained on
MassSpecGym regardless, and 98.7% in-distribution is the target.

**The question that decides architecture vs limitation paragraph.** If attachment
is carried by z, is the true molecule recoverable by *searching* over z? Added
`--multiplicity K`: decode each ground-truth coarse graph under K prior draws,
count distinct molecules, check whether the true one appears.

- High recall at modest K -> attachment is a *search* problem. The oracle ranks z
  samples at test time, no architecture change, and the credit question reduces
  to how efficiently the policy proposes z.
- Low recall -> z is a genuine bottleneck. Fix, in order of preference: discrete
  attachment sites (make "which site of fragment A bonds to fragment B"
  categorical like every other node and edge, so it flows through the same
  machinery and takes credit like anything else, at the cost of a ragged
  categorical dimension), then a per-bond latent, which is the smaller code
  change and the weaker answer.

**Keep separate from C3 in the writeup.** Some isomer pairs are indistinguishable
because their simulated spectra genuinely are near-identical — verifier ceiling,
not credit assignment. A reviewer who conflates them reads an architecture fix as
papering over a fundamental limit. E0-e is clean by construction: the oracle is
never in the loop.

**Fixed:** E0-b built a fresh generator per step count, which LMDB rejects
(second open of one environment in a process) and which would also re-embed all
54k fragments each time. Now builds once and varies `n_euler_step`, passing
fragment counts explicitly so batch size varies without rebuilding the loader.

---

## 2026-09-19 — Fragment bag built; E0-e answers the attachment question without training

**Fragment bag is built.** 54,095 fragments, 53,220 with nonzero train
occurrence, from the MassSpecGym train fold plus 200k MCES-2-disjoint corpus
molecules. Zero dropped in `process_frag`. FragFM's NPGen checkpoints and
processed data are in place, so E0-b is unblocked.

**E0-e: measure the attachment question now, not after training.** The
disciplined version of R6 is to sort a trained model's errors into "wrong
fragment multiset" and "right multiset, wrong attachment" and count them — a
limitation paragraph at 20%, an architecture problem at 70%. That needs a trained
conditional model, which is weeks away. There is a training-free lower-bound
version available today, and it isolates the variable more cleanly:

> hold the coarse graph at ground truth, vary only z, watch reconstruction.

If reconstruction from a prior-sampled z matches reconstruction from the encoded
z, then attachment is determined by the coarse graph, z carries nothing, and the
concern is void. If it collapses, z carries attachment information that
per-fragment credit cannot reach. Either way the answer arrives in week 1 instead
of week 9, and it does not confound the question with model quality — the error
sort cannot separate "z is a bad carrier" from "the policy is undertrained",
whereas this holds everything else at ground truth.

The `prior` arm reproduces generation-time z exactly, including the min-max
inverse `store_smis_from_coarse_graph` applies: the generator does not feed a
standard normal to the decoder, and testing a raw normal would have measured the
wrong distribution.

Worth noting FragFM's own `exe/eval_ae.py` carries a commented-out
`z = z + torch.randn_like(z) * 0.2` probe. The authors asked this question and
did not publish the answer.

**If z does carry attachment information**, the fixes in preference order are
discrete attachment sites — make "which site of fragment A bonds to fragment B" a
categorical variable like every other node and edge, so it flows through the same
machinery and receives credit like anything else, at the cost of a ragged
categorical dimension — then a per-bond latent, which is a smaller change to
their code and less elegant.

**Keep this separate from C3 when writing up.** Some isomer pairs are
indistinguishable because their predicted spectra genuinely are nearly identical.
That is the verifier ceiling, not a credit-assignment failure, and a reviewer who
conflates them will read an architecture fix as papering over a fundamental
limit. The two are separable by construction: E0-e holds the oracle out of the
loop entirely.

**Next:** E0-e on NPGen (in-distribution, clean) then on MassSpecGym (transfer).
E0-b throughput now unblocked.

---

## 2026-09-19 — MassSpecGym decomposed; own fragment-bag builder; two tooling bugs

**MassSpecGym is decomposed.** 31,561 of 31,602 structures (99.87%) under
rBRICS+relaxed in 116s on 64 cores, written to FragFM's LMDB format with the
benchmark's own fold prefixes.

**We now build the fragment bag ourselves** (`scripts/build_fragment_bag.py`),
replacing FragFM's `process_fragment_from_lmdb.py`. Two reasons, and the second
is the real one:

1. Their script hard-codes a dataset whitelist to choose the strict/relaxed
   branch and raises `NotImplementedError` on anything else.
2. It reads exactly one source LMDB. E0-c says the pool must be harvested from a
   large external corpus, not from MassSpecGym alone — 54.4% to 82.3% test
   coverage. Pooling several sources is a requirement, not a convenience.

**A leakage trap in the bag mechanism, found while writing it.**
`random_select_frags_by_occurance` samples with `p = occurrence/sum` and
`replace=False`, so **a fragment with zero occurrence in the selected split is
unreachable**. Harvest a pool from the corpus while counting occurrences only
over MassSpecGym, and every corpus fragment is inert — the pool looks large and
behaves as if it were not. The coverage gain E0-c measured would have silently
failed to materialise, and the symptom would have been "the big pool did not
help", which is not obviously a bug.

So `train_occurance` is written as *everything the generator may draw from*: the
MassSpecGym train fold plus the external corpus. That is sound precisely because
the corpus is MassSpecGym's own MCES-2-disjoint release, built to be distant from
the test fold. MassSpecGym val/test counts are recorded for diagnostics and never
folded into the train bag.

**Two bugs in my own patch tool, both of the same family.** Deciding "already
applied" is fiddly in both directions, and I got it wrong in each:

- `new` a *prefix* of `old` (the Draw patch deletes a trailing import): checking
  `new in src` first marked every unpatched file as done. The tool reported
  success and changed nothing.
- `old` a *substring* of `new` (the `remove_h` body patch wraps its anchor):
  after fixing the above by testing the anchor first, this one re-applied on
  every run, stacking duplicate copies of an unreachable early-return block.

Now: a patch that wraps its anchor must declare an explicit `marker` unique to
the patched state; everything else is decided by anchor absence. Both failure
modes are named in a comment so the next patch does not rediscover them.

The second bug means `FragFM/fragfm/utils/mol_ops.py` on the cluster has a
duplicated block — harmless, since the duplicate is unreachable, but it must be
reset before it grows.

**Next:** fragment bag, then coarse-to-fine reconstruction accuracy — the
unmeasured second factor of the top-1 ceiling. E0-b throughput still pending on
the Drive assets.

---

## 2026-09-19 — rBRICS reverses the E0-a decision; the attachment latent is global

**Decomposition: rBRICS, not BRICS.** E0-a chose BRICS on edge-slot compression
(14.4x vs 9.0x) before coverage existed as a number. With the rBRICS arm in,
rBRICS reaches **higher coverage with a 35% smaller pool at every scale**: 82.3%
vs 72.5% of test structures at a 200k-molecule pool, from 53,220 fragments
against 81,739. Finer fragments recur across molecules more, so each one is worth
more. Full table in RESULTS.md R5.

Scoring all three axes: compression is *not* binding (9.0x still triples the
proposal budget of 3-6x), coverage is (it caps exact top-1), and pool fraction
per bag draw is (384/53,220 = 0.72% against 0.47%). rBRICS wins both binding axes
and loses the one with slack.

The pool-fraction column is what decided it, and it only exists because of
yesterday's finding that the bag is redrawn every Euler step: a smaller pool is
explored more thoroughly per draw, and that advantage compounds exactly when step
count is cut for speed. Compression could not have told us this. Cost: 8.6
fragments per molecule instead of 7.1, ~20% slower to decompose.

**The attachment-point latent is global — verified, and it dents C2.** Flagged by
a parallel review and confirmed in the code: `gen_z = torch.randn(bs,
latent_z_dim)` is one latent *per molecule*, and `AE.decode` turns it into a
single graph-level embedding (`ae.py:205`) from which inter-fragment bonds are
predicted. So the generative variables are coarse nodes, coarse edges, and one
global z. Fragment identity and fragment connectivity take per-variable
advantages cleanly; **attachment-point choice has only z**, so for that failure
mode Method D's structured advantage degrades to a scalar — the same aggregation
loss the proposal criticises FRIGID for, arriving through a different door. It
bites because attachment errors are the isomer confusions MS/MS resolves worst:
right fragments, wrong joins.

Not yet quantified, and it is quantifiable. The decoder is deterministic given
(coarse graph, z), so if coarse-to-fine reconstruction is near-lossless on
MassSpecGym chemistry then attachment is nearly determined by the coarse graph
and z carries little of the decision. Week-3 architecture decision either way,
per RESULTS.md R6.

**A quantity that gates everything, and is cheap.** The ceiling on exact top-1
factorises as P(all fragments of the target are in the pool) x P(coarse-to-fine
reconstruction correct) = 0.823 x (unmeasured) at a 200k rBRICS pool. That
product bounds every accuracy number the project can report, independent of
generator, oracle and RL. Promoted in the proposal to a week-1 item rather than
an ablation.

**Built:** `scripts/build_msg_lmdb.py` — decomposes MassSpecGym into FragFM's
LMDB format with train_/valid_/test_ key prefixes taken from the benchmark's own
folds, so FragFM's `process_fragment_from_lmdb.py` runs unchanged afterwards.
Prerequisite for measuring the second ceiling factor and for any training.

**Converging independently:** a parallel reading of FragFM reached the same
conclusion about spectrum-conditioning the bag. Their reweighting hook already
exists in code — `random_select_frags_by_occurance` multiplies the occurrence
prior by `exp(-bag_guide_strength * property_mse)` when a discriminator is loaded
— so the mechanism is a drop-in. Our version is stronger than a learned property
score: subformula-of-precursor is an exact constraint, not a prediction.

**Next:** build the MassSpecGym LMDB, then coarse-to-fine reconstruction accuracy
(the second ceiling factor). E0-b throughput still pending.

---

## 2026-09-19 — R3 retracted. The bag is swappable; the cost is coupled to step count

**Correction, prompted by Mohsen.** R3 called the unseen-fragment rate a
"representational ceiling", which implies a limit set by the training data. That
is wrong. FragFM has no fragment-identity embedding table: `FragToVect`
(`fragfm/model/flow.py:11`) encodes each candidate from its own graph — atomic
numbers, junction counts, bond types, message passing — so a fragment never seen
in training is embedded and scored like any other. The candidate pool is a data
artifact loaded at inference and swappable. This is precisely the fixed-
vocabulary trap FragFM was built to escape, and I wrote it up as though the trap
were still there. The R3 numbers are correct; the conclusion was not. Retracted
in RESULTS.md R4.

**Read correctly, E0-c is a design curve and it is encouraging.** Coverage of the
test fold: 44.5% from the full train fold (9,296 fragments), 72.5% from 200k
corpus molecules (81,739 fragments), still climbing steeply with 4M available.
R3's alarming 0.286 was an artifact of harvesting from only 5,000 molecules. The
pool is a knob, and a cheap CPU-only one.

**What reading the mechanism did surface — and this one is real.** The bag is
redrawn at **every Euler step** (`mol_generator.py:398-412`): 384 fragments by
occurrence weighting, unioned with whatever is already placed. A 500-step
generation is 500 independent draws, i.e. a stochastic search over the pool.

So **cutting 500 steps to 20 also cuts bag resampling 25-fold.** The proposal's
budget table treats step reduction and fragment-level representation as
independent multiplicative levers; they are not. Pool scale and step count pull
against each other — a larger pool raises the ceiling but needs more draws to
exploit, and cutting steps shrinks exploration exactly when the pool is largest.
FragFM never cuts steps, so its paper never had to confront this. This would have
surfaced in week 6 as an unexplained quality collapse under step reduction.

**The fix is on-thesis, which is why I think it is right rather than convenient.**
Stop drawing the bag by occurrence and select it from the spectrum. MS/MS peaks
*are* fragment masses: admissible fragments have formulas that are subformulas of
the precursor, and informative ones match observed peaks within tolerance. Both
constraints are exact, free to evaluate, and instance-specific — which is what
the conditional setting needs and what occurrence weighting cannot provide. A
spectrum-filtered bag raises coverage and removes the step-count dependence
together, because few draws suffice once the candidates are right.

That is the proposal's thesis applied one level lower: generate in the units the
evidence is expressed in, and *select* in them too. Added to Method A as a fourth
conditioning signal, flagged as conditioning the support rather than the network.

**Next:** E0-d — how far do formula and peak filtering prune an 81,739-fragment
pool, and what coverage survives. Then E0-b throughput, still pending.

---

## 2026-09-19 — Held-out vocabulary: a representational ceiling at 28.6%, and a decision reopened

Preflight 8/8 (`setuptools` fixed `torch.compile`).

**The finding: 71.4% of MassSpecGym test molecules contain at least one BRICS
fragment that never appears in the train fold.** The complete train-fold
vocabulary therefore caps exact top-1 at **28.6%**, and the top-V curve is flat
from V=2,000 — the missing fragments are not rare training fragments, they are
absent from the training molecules entirely. Details in RESULTS.md R3.

This matters because the field sits at ~18% top-1. Ten points of headroom on a
ceiling imposed by our own representation is not comfortable, and it is a
ceiling of the same kind as C3's verifier ceiling, sitting on the representation
instead of the oracle. The two stack. It caps *exact* top-1 only — MCES and
Tanimoto degrade gracefully — and MassSpecGym's MCES>=10 split is designed to
make held-out structures distant, so a large unseen-fragment share is the split
working as intended rather than a defect. It is still the binding number.

**I was wrong to close the BRICS-vs-rBRICS question yesterday.** I dropped
rBRICS on compression alone (9.0x vs 14.4x) before measuring held-out coverage.
Finer fragments recur more, so rBRICS should have a *higher* ceiling, and it
already showed better in-sample top-V coverage (0.138 vs 0.099 at V=100). That
is a trade between compression and ceiling, not a straight loss. Reopened; E0-c
measures both arms.

**New experiment E0-c (`scripts/vocab_ceiling.py`).** Does the ceiling lift with
vocabulary scale, and how fast? Harvest fragments from the benchmark's
4M-molecule MCES-2-disjoint corpus at 5k -> 1M molecules and re-measure test
coverage, for both decompositions. Fragments recur across molecules far more
than molecules recur, so this should lift substantially; if it does, the fix is
CPU-only and cheap, and it becomes a concrete argument for the pretraining half
of the plan rather than a scaling claim taken on faith. The script decomposes
the largest slice once and reads nested prefixes off it, so the whole curve
costs a single pass.

If the ceiling stays low even at 1M molecules, the representation needs an
atom-level escape hatch for unseen fragments, and that is a week-3 design change
rather than a week-9 surprise.

**Fixed:** `gdown` 5.x dropped `--id`; the bare file id still works.

**Next:** E0-c, then E0-b throughput once the FragFM checkpoints land.

---

## 2026-09-19 — E0-a done. Base decomposition settled; compression is 2-4x better than budgeted

Preflight is green apart from `torch.compile`, and that had nothing to do with
triton. **triton was installed the whole time; it imports `setuptools` at
runtime, and a uv venv ships without setuptools.** torch's `has_triton()`
swallows the `ImportError` and reports triton as "not installed or too old",
which sends you auditing the wrong package. Installing `setuptools` is the fix.
Noted because the error message is actively misleading.

**E0-a result (see RESULTS.md R2): BRICS with relaxed canonicalisation.**
99.8% of MassSpecGym decomposes, 7.1 fragments per molecule, 14.4x edge-slot
reduction, 3,485 distinct fragments over 5,000 structures, 2.1 mol/s/core.

Three things follow.

*FragFM stays the base.* The question E0-a existed to answer was whether
FragFM's decomposition survives MassSpecGym chemistry. At 99.8% it does, so the
FlowMS/DiffMS fallback is off the table and week 3-4 can commit to conditioning
FragFM rather than hedging.

*rBRICS is a straight loss here, which was not obvious.* It cuts more finely --
8.6 fragments instead of 7.1 -- and that makes compression *worse*, 9.0x against
14.4x, because more fragments means more fragment-level edge slots. It is also
slower to compute. Finer decomposition is not a free upgrade; this project wants
fewer, larger generative units, and that is also what keeps the GRPO horizon
short. Dropping rBRICS from consideration.

*Relaxed canonicalisation is the chemically correct setting, not a leniency
knob.* It differs from strict only in allowing a formal charge on N/O/S carrying
excess valence, and MassSpecGym contains genuinely charged species. It recovers
2.6% of structures that strict silently drops.

**The compression lever measures 14.4x against 3-6x budgeted.** Proposal updated,
but conservatively: the edge-slot ratio is a structural bound, not a speedup --
the coarse-to-fine autoencoder and everything linear in node count do not shrink
with it. 3-6x stays the planning figure until E0-b measures wall-clock. What the
number does say is that the 100x target has more headroom than the budget table
assumed.

**Adduct is a conditioning input.** v1.5 carries `[M+H]+` (195,237) and
`[M+Na]+` (35,867). Sodiated species fragment differently from protonated ones
and they are 15.5% of the data, so the model should be told which it is looking
at rather than inferring the charge carrier from the peaks. Added to Method A's
conditioning list; it is nearly free.

**Preprocessing is cheap enough not to plan around.** 134 mol/s on 64 cores puts
the 31,602 MassSpecGym structures at ~4 minutes and the benchmark's 4M-molecule
MCES-2-disjoint pretraining corpus at ~8.3 hours on one node.

**Still open:** the fragment vocabulary top-V coverage curve — 3,485 distinct
fragments over 5,000 molecules implies a long tail, and how long decides how
hard FragFM's stochastic fragment bag has to work. Measured, not yet read back.

**Next:** E0-b, wall-clock sampling throughput of unconditioned FragFM from the
released checkpoint. That is the number the proposal calls blocking, and it
sizes group size, training-set size and the whole RL schedule.

---

## 2026-09-19 — Preflight round 1: 4/8. Four fixes, one of them an upstream bug

Environment built on the A100 without incident (torch 2.6.0+cu124, numpy 2.4.6,
rdkit 2025.03.6, pyg 2.8.0, dgl 2.5.0+cu124, python 3.11.16).

**The single-environment gamble paid off.** `fragfm.process` and
`fragfm.model.flow` import clean under numpy 2 / rdkit 2025.03 despite FragFM
pinning numpy 1.26 / rdkit 2023.9. The GRPO reward can stay in-process.

**FragFM's released preprocessing is broken at HEAD.**
`fragfm/process.py` calls `reconstruct_to_rdmol(..., remove_h=True/False)` in
four places, but `fragfm/utils/mol_ops.py:13` defines no such parameter — it
unconditionally calls `Chem.RemoveHs`. Every call to `process_sample` raises
`TypeError`, so the published data-processing path cannot run as released.
Upstream commit `6b7e648` ("fix: make mol reconstruction work for custom
dataset") changed `mol_generator.py` only and left the callers in `process.py`
stranded. This is not a version-skew artifact; it is broken for everyone.

The repair matters for correctness, not just for getting past the exception.
`remove_h=False` callers feed the result to `remolstar.GetSubstructMatch(canmolh)`
and index into the match to build the fragment reordering map, so that path must
skip *both* `Chem.RemoveHs` *and* the SMILES round-trip inside
`valid_mol_can_with_seg` — the round-trip reorders atoms and would silently
produce a wrong permutation rather than an error. Patched accordingly in
`scripts/apply_patches.py`, and `preflight.py` now verifies the fix by
round-tripping: rebuild the molecule from the permuted `h`/`e_index`/`e` and
require canonical-SMILES equality with the input. Absence of an exception is not
evidence the permutation is right.

Worth reporting to the FragFM authors — it doubles as the contact the proposal
already schedules for week 1.

**Three environment fixes**

- *triton missing → `torch.compile` fails.* `--index-url .../whl/cu124` suppresses
  PyPI, and the cu124 index carries no triton, so torch's `triton==3.2.0`
  dependency was silently dropped. Installed explicitly.
- *`libXrender.so.1` missing → all of `ms_pred` unimportable.*
  `ms_pred/common/__init__.py` eagerly imports `plot_utils` → `rdkit.Chem.Draw`
  → `libXrender`. The cluster image lacks it and we have no root. Patched to a
  PEP-562 module `__getattr__` so the plotting helpers import on first use.
  Nothing on the training or RL path draws molecules. `cairosvg` was already
  lazy, so this is the only blocker of its kind in `common`.
- *bf16 benchmark read 9 TFLOP/s on an A100* — my measurement bug, not the
  hardware: the first matmul pays cuBLAS handle initialisation and kernel
  autotune, which dominated a 20-iteration timing. Added warmup.

**Decision: patch the reference checkouts, do not fork them.**
`scripts/apply_patches.py` applies exact-anchor string replacements, is
idempotent, and fails loudly if an anchor disappears. Each patch carries the
reason inline. This keeps the delta we depend on visible and lets it drop out
when upstream fixes things — a fork would bury both.

---

## 2026-09-19 — Repo bootstrap, environment strategy, week-1 scaffolding

**Done**

- Repo skeleton: `msfragfm/` package, `scripts/`, `cluster/`, `configs/`, `results/`.
- `scripts/setup_env.sh` — single environment for generator + oracle.
- `scripts/preflight.py` — eight checks gating everything downstream.
- `scripts/download_massspecgym.py` — targeted HF file fetch + split report.
- `scripts/fragment_coverage.py` — experiment **E0-a** (see below).

**Decision: one environment, not two.**
FragFM pins `numpy==1.26 / rdkit==2023.9 / torch==2.1`; ms-pred wants
`numpy>=2 / rdkit==2025.3 / torch==2.6 + dgl`. On paper these are irreconcilable.
Grepping FragFM's source finds no numpy-1-only aliases and no rdkit-2023-only
APIs, so the pins look conservative rather than load-bearing. We therefore
resolve on ms-pred's newer stack and install FragFM with `--no-deps`.
Why it matters: GRPO makes ~160k oracle calls per epoch, and a two-environment
split would force all of them through IPC. `preflight.py` is the empirical test
of this gamble; the fallback is a two-env split with the oracle behind a socket.

**New experiment E0-a, pulled forward from week 3.**
The proposal schedules "validate fragment vocabulary coverage" in week 3, but it
is a *base-model selection* question, not a tuning question: if FragFM's BRICS
decomposition fails on a large share of MassSpecGym, FragFM cannot be the base
and we fall back to FlowMS/DiffMS. It costs minutes on 64 cores, so it runs in
week 1, before any environment work is sunk into the wrong base.
`scripts/fragment_coverage.py` measures, over `brics`/`rbrics` x strict/relaxed
canonicalisation: decomposition success rate, fragment-count distribution,
**measured edge-slot reduction** (the proposal's 3-6x lever), fragment
vocabulary size, and preprocessing throughput.

**Findings while reading the reference code**

- MassSpecGym v1.5 is a single file, `data/MassSpecGym1.5.tsv`, inside a 52.8 GB
  HF repo — fetch per-file, never snapshot.
- The HF repo already ships
  `MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv` and
  `MassSpecGym1.5_pretraining_molecules_{2.5M,50M}_tani070.txt`. The proposal
  budgets work for MCES-filtering a pretraining corpus against the test fold;
  the benchmark authors have already done it. Proposal updated.
- GLACIER's `predict_smis_joint.py` documents that **featurization, not the GPU
  forward, is the inference bottleneck** — it already runs a CPU worker pool with
  a per-worker LRU cache and size-sorted batching. This changes the oracle
  throughput plan: the RL reward loop is CPU-bound, so it wants many CPU cores
  next to the GPU, and SMILES-level caching is worth more than a bigger batch.
  Proposal updated.
- `ms_pred/glacier/` and `ms_pred/iceberg/` ship without `__init__.py`; import
  their modules directly (`from ms_pred.glacier import joint_model`).

**Next (blocking on the interactive A100 session)**

1. `bash scripts/setup_env.sh && python scripts/preflight.py`
2. `python scripts/download_massspecgym.py --which main`
3. `python scripts/fragment_coverage.py --n 5000`

**Housekeeping**

- The `origin` remote URL embeds a GitHub PAT in plaintext. It is readable by
  anything that can run `git remote -v`. Rotate it and move it to a credential
  helper or `~/.git-credentials`.
