"""Train on the policy's own states: self-conditioning, then the group-relative arms.

R25 through R31 closed every explanation for 33% generated against 80%
teacher-forced per-fragment accuracy except one. Training corruption masks *true*
fragments, so it never builds a state where an earlier slot holds a **wrong**
fragment -- and after a few Euler steps that is most of what a rollout is. The
model is accurate where teacher forcing puts it and wrong where generation goes.

Three arms here, all consuming identical rollouts so the comparison is paired:

| arm | cross-entropy target | reward enters as |
| --- | --- | --- |
| `selfcond` | the truth | not at all |
| `grpo` | the truth | signed group-normalised advantage as a weight |
| `dmpo` | the truth | softmax weight over the group |

**`selfcond` is the diagnostic and it runs first.** It needs no reward at all:
corrupt from the rollout, supervise toward the truth. If the gap is the state
distribution then this closes it, and the RL arms must then be measured against it
rather than against `e3b-xattn`. If it does not move, the distribution-shift
diagnosis is wrong and the RL arms would have inherited that error silently.

**Two honest limitations, because the arm names invite more than is implemented.**

`grpo` here is REINFORCE with a group baseline, not PPO: the advantage multiplies
the cross-entropy directly, with no importance ratio and no clipping. A trajectory
likelihood would be needed for the ratio, and the marginal probability of a
generated graph requires a sum over every path reaching it -- the obstacle Method D
names. So this is the group-relative *credit assignment* of GRPO without its trust
region, and it should not be reported as GRPO proper.

`c_dtm` is deliberately absent. Its target is a tilt on the *teacher's* predictive
distribution at the same corrupted state, which needs the frozen teacher's logits
alongside the student's. The patch hands back the student's `h_logit` but not the
corruption inputs (`ht_onehot`, `et_onehot`, `zt`, `frag_zs`) a second forward pass
would need, so it wants one more patch. It is the arm best matched to this model
class and it should follow, not be faked with a weight.

**On-policy without weight syncing.** The optimiser is pointed at the sampler's own
`frag_embedder` and `coarse_gnn`, so rollouts and gradients share one set of weights
by construction rather than by a copy after each step.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from msfragfm import tracking
from msfragfm.diversity import fragment_usage, policy_entropy
from msfragfm.objectives import dmpo_weights, grpo_advantages
from msfragfm.paths import RESULTS
from msfragfm.rollout import fragment_rewards, group_view, rollout
from msfragfm.spectra_data import cond_inputs, make_spectrum_dataset
from msfragfm.spectrum import SpectrumEncoder

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
RDLogger.DisableLog("rdApp.*")
ARMS = ("selfcond", "grpo", "dmpo")


class OneBatch:
    """One step presented as a loader.

    `process_single_epoch` iterates its argument and reaches through `.dataset`
    for the fragment bag and the fragment graphs, so a shim is cheaper than
    restructuring it.
    """

    def __init__(self, dataset, batch):
        self.dataset, self._b = dataset, batch

    def __iter__(self):
        return iter([self._b])

    def __len__(self):
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RESULTS / "e3b-xattn.pt"))
    ap.add_argument("--ae", default=str(RESULTS / "ae_ft_brics.pt"))
    ap.add_argument("--nfrag-ckpt", default=str(RESULTS / "nfrag.pt"))
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--arm", default="selfcond", choices=ARMS)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--spectra-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--euler-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="dmpo softmax temperature on the reward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"rl_{args.arm}"

    sys.path.insert(0, str(FRAGFM))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(FRAGFM)
    from exe.train_flow import process_single_epoch
    from fragfm.distort_scheduler import DistortScheduler
    from fragfm.mol_generator import FragFMGenerator
    from fragfm.utils.file import read_yaml_as_easydict

    from eval_denovo import export_for_generator

    stem = Path(args.data).stem
    model_dir = Path("save/flow_model/msfragfm_rl_" + Path(args.ckpt).stem)
    ae_dir = Path("save/ae_model/msfragfm_ft")
    ae_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy("save/ae_model/npgen/cfg.yaml", ae_dir / "cfg.yaml")
    shutil.copy(args.ae, ae_dir / "model_best.pt")
    ck = export_for_generator(Path(args.ckpt), model_dir, ae_dir)

    gcfg = read_yaml_as_easydict("cfgs/generate/npgen.yaml")
    gcfg.data_dirn = args.data
    gcfg.frag_data_dirn = f"data/processed/{stem}_fragment.lmdb"
    gcfg.frag_smi_to_idx_fn = f"data/processed/{stem}_fragment_to_idx.pkl"
    gcfg.fm_model_dirn = str(model_dir)
    gcfg.n_euler_step = args.euler_steps
    gcfg.bs = args.spectra_per_step * args.group
    gcfg.n_sample = gcfg.bs
    # Also the split the generator's own FragFMDataset is built on, which the
    # training dataset adopts rather than opening the environments a second time.
    gcfg.fragment_bag = "train"
    gcfg.force_save_dirn = gcfg.save_dirn = str(RESULTS / "_rl_scratch")
    os.makedirs(gcfg.save_dirn, exist_ok=True)

    sampler = FragFMGenerator(gcfg)
    sampler.set_seed(args.seed)
    sampler.spanning_tree_decode = True
    sampler.enforce_valency = True

    ds = make_spectrum_dataset(args.data, gcfg.frag_data_dirn,
                               gcfg.frag_smi_to_idx_fn, "train",
                               base=sampler.test_set)

    # The policy is the sampler's own modules, so rollouts and gradients cannot
    # drift apart.
    frag_embedder, coarse_gnn = sampler.frag_embedder, sampler.coarse_gnn
    frag_embedder.train()
    coarse_gnn.train()
    cfg = sampler.fm_cfg
    cfg.latent_transform_param = sampler.cfg.latent_transform_param
    cond_model = SpectrumEncoder(out_dim=ck["cfg"]["embd_h_dim"]).cuda()
    cond_model.load_state_dict(ck["cond_model"], strict=False)
    use_xattn = bool(ck["cfg"].get("use_cross_attention"))
    print(f"{tag}: {len(ds):,} train spectra, arm {args.arm}, "
          f"cross-attention {use_xattn}")

    params = (list(frag_embedder.parameters()) + list(coarse_gnn.parameters())
              + list(cond_model.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr)
    cfg.lr, cfg.lr_warmup, cfg.n_iter_done = args.lr, False, 0
    scheds = (DistortScheduler(cfg.node_distort_schedule),
              DistortScheduler(cfg.edge_distort_schedule),
              DistortScheduler(cfg.latent_z_distort_schedule))

    # True fragment multiset per structure, for the reward. Only structures whose
    # every fragment resolves to a pool index are scorable, matching how
    # eval_denovo builds the same table.
    smi2idx = pickle.loads(Path(gcfg.frag_smi_to_idx_fn).read_bytes())
    true_frags = {}
    with sampler.test_set.env.begin() as txn:
        for k, v in txn.cursor():
            ids = [smi2idx.get(f) for f in pickle.loads(v)["frag_smi_list"]]
            if all(i is not None for i in ids):
                true_frags[k] = ids
    print(f"scorable structures: {len(true_frags):,}")

    nck = torch.load(args.nfrag_ckpt, map_location="cpu", weights_only=False)
    nfrag_model = SpectrumEncoder(out_dim=nck["max_n"]).cuda().eval()
    nfrag_model.load_state_dict(nck["model"])

    loader = DataLoader(ds, batch_size=args.spectra_per_step, shuffle=True,
                        drop_last=True)
    run = tracking.init(tag, {**vars(args), "n_train": len(ds)})

    history, it, unscored = [], iter(loader), 0
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        graph, coarse = batch
        coarse = coarse.to("cuda")

        # --- conditioning, exactly as the evaluation path sets it -------------
        ci = {k: v.cuda() for k, v in cond_inputs(coarse).items()}
        with torch.no_grad():
            sampler.cond = cond_model(ci).repeat_interleave(args.group, dim=0)
            if use_xattn:
                mem, keep = cond_model.memory(ci)
                sampler.cond_mem = mem.repeat_interleave(args.group, dim=0)
                sampler.cond_keep = keep.repeat_interleave(args.group, dim=0)
            p = nfrag_model(ci).softmax(-1)
            n_pred = ((p.cumsum(-1) < 0.5).sum(-1) + 1).clamp_min(2).cpu().tolist()

        # --- roll out and score ----------------------------------------------
        roll = rollout(sampler, [n for n in n_pred for _ in range(args.group)])
        keys = [ds.keys[i] for i in coarse.ds_idx.tolist()]
        missing = [k for k in keys if k not in true_frags]
        unscored += len(missing)
        truth = [true_frags.get(k, []) for k in keys]
        per_node, per_sample = fragment_rewards(
            roll["h_type"], roll["batch"],
            [t for t in truth for _ in range(args.group)])

        # --- one gradient step on the rollout's states ------------------------
        rl = {"state_h": roll["h_type"]}
        ess = None
        if args.arm == "dmpo":
            # Scaled by G so the mean weight is 1 and the step size matches
            # selfcond's, which makes the arms comparable at equal lr.
            w, ess = dmpo_weights(group_view(per_sample, args.group),
                                  alpha=args.alpha)
            rl["weight"] = w.reshape(-1) * args.group
        elif args.arm == "grpo":
            rl["weight"] = grpo_advantages(
                group_view(per_sample, args.group)).reshape(-1)

        res = process_single_epoch(
            cfg, sampler.ae_model, frag_embedder, coarse_gnn,
            OneBatch(ds, batch), scheds, frag_occurance_source="train",
            optimizer=opt, cond_model=cond_model, rl=rl)

        out = rl.get("out", {})
        row = {"reward": float(per_sample.mean()),
               "reward_std": float(per_sample.std()),
               "reward_best": float(group_view(per_sample, args.group)
                                    .max(-1).values.mean()),
               "loss": res["loss"], "frag_loss": res["fragment_type_loss"],
               "edge_loss": res["fragment_edge_loss"]}
        if out:
            # The collapse instruments, from step 1 rather than after the fact:
            # the RLVR warning is that entropy falls long before any metric moves.
            row["policy_entropy"] = policy_entropy(out["h_logit"].detach())
            row.update({k: v for k, v in
                        fragment_usage(roll["h_type"].tolist(),
                                       sampler.n_all_frag).items()
                        if k in ("effective_vocab", "fragment_entropy")})
        if ess is not None:
            row["dmpo_ess"] = float(ess.mean())
        history.append({"step": step, **row})
        if step == 1 or step % 10 == 0:
            print(f"  step {step:>4}  reward {row['reward']:.4f}  "
                  f"best {row['reward_best']:.4f}  frag {row['frag_loss']:.4f}  "
                  f"ent {row.get('policy_entropy', float('nan')):.3f}  "
                  f"vocab {row.get('effective_vocab', float('nan')):.1f}",
                  flush=True)
        tracking.log(run, {f"rl/{k}": v for k, v in row.items()}, step=step)

    if unscored:
        print(f"[warn] {unscored} samples had no resolvable true multiset and "
              f"scored 0")
    tracking.finish(run)
    torch.save({"frag_embedder": frag_embedder.state_dict(),
                "coarse_gnn": coarse_gnn.state_dict(),
                "cond_model": cond_model.state_dict(),
                "ema": {"frag_embedder": frag_embedder.state_dict(),
                        "coarse_gnn": coarse_gnn.state_dict()},
                "cfg": dict(cfg), "args": vars(args)}, RESULTS / f"{tag}.pt")
    (RESULTS / f"train_{tag}.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "unscored": unscored}, indent=2))
    print(f"\nwrote {RESULTS / (tag + '.pt')}")


if __name__ == "__main__":
    main()
