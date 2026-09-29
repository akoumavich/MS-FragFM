"""Self-conditioning: train on states where earlier slots hold the model's own errors.

R25 through R31 closed every explanation for 33% generated against 80%
teacher-forced per-fragment accuracy except one. Training corruption masks *true*
fragments, so the revealed slots always hold correct values -- and after a few Euler
steps a rollout's revealed slots mostly hold wrong ones. The model is accurate where
teacher forcing puts it and wrong where generation goes.

This trains on the second kind of state. One step is two forward passes:

1. the ordinary corrupted state, read for what the model predicts at each slot;
2. the same true graph, but the revealed slots carry those predictions instead of
   the true values, supervised toward the truth.

**Why not a generation rollout.** The first attempt corrupted from a full rollout and
supervised toward the truth, which does not typecheck as science: the rollout has
`n_pred` coarse nodes and the structure has `n_true`, they differ whenever the count
predictor is wrong (absolute error 2.45, R31), and there is no correspondence between
rollout slot *i* and true slot *i*. Keeping the true graph and substituting only the
*values* preserves node identity, still produces "an earlier slot is wrong", and
costs two forwards instead of a hundred Euler steps.

**The reward-weighted arms are not here yet, and not faked.** `grpo`, `dmpo` and
`c_dtm` supervise toward the policy's *own* sample rather than the truth, so they need
the rollout substituted as the data -- graph, edges and target together -- which the
current patch does not do. `msfragfm/rollout.py` has the scoring they need and
`msfragfm/objectives.py` the losses; the missing piece is building a coarse graph from
rollout tensors. Self-conditioning comes first regardless: it is the diagnostic, and
if it closes the gap the reward arms have to be measured against it rather than
against `e3b-xattn`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import RDLogger
from torch.utils.data import DataLoader

from msfragfm import tracking
from msfragfm.diversity import fragment_usage, policy_entropy
from msfragfm.paths import RESULTS
from msfragfm.spectra_data import cond_inputs, make_spectrum_dataset
from msfragfm.spectrum import SpectrumEncoder

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
REPO = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")


class OneBatch:
    """One batch presented as a loader.

    `process_single_epoch` iterates its argument and reaches through `.dataset` for
    the fragment bag and the fragment graphs, so a shim is cheaper than
    restructuring it.
    """

    def __init__(self, dataset, batch):
        self.dataset, self._b = dataset, batch

    def __iter__(self):
        return iter([self._b])

    def __len__(self):
        return 1


def repo_path(p):
    """Resolve against the MS-FragFM root.

    Both scripts chdir into FragFM so its own relative paths work, which means a
    relative --ckpt from the caller would otherwise resolve inside FragFM. `--data`
    is deliberately left alone: it *is* FragFM-relative.
    """
    q = Path(p)
    return str(q if q.is_absolute() else (REPO / q).resolve())

def predicted_ids(out, temperature):
    """The model's own choice per slot, as global fragment ids.

    Sampled rather than argmax: the states generation visits are sampled ones, and
    an argmax state is both narrower and easier than what the Euler loop produces.
    Masked-out columns are -inf and drop out of the softmax on their own.
    """
    logit = out["h_logit"][:, :out["cur_frag_idxs"].numel()]
    p = torch.softmax(logit.detach() / temperature, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    # A row with no admissible candidate falls back to its true value rather than
    # to fragment zero, which would teach the model to recover from a state the
    # sampler can never produce.
    dead = p.sum(-1) <= 0
    pick = torch.multinomial(p.clamp_min(1e-12), 1).squeeze(-1)
    ids = out["cur_frag_idxs"][pick]
    return torch.where(dead, out["cur_frag_idxs"][out["h_type"]], ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RESULTS / "e3b-xattn.pt"))
    ap.add_argument("--ae", default=str(RESULTS / "ae_ft_brics.pt"))
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--arm", default="selfcond", choices=("selfcond", "teacher"),
                    help="teacher is the control: identical code path, true values "
                         "in the revealed slots, so any difference is the states")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-power", type=float, default=1.0,
                    help="bias the flow-time draw toward the fully "
                         "masked end; 1.0 is uniform, 3.0 puts 53%% of "
                         "steps below t=0.15 where R34 measured "
                         "accuracy at 0.37")
    ap.add_argument("--sweep-t", action="store_true",
                    help="measure accuracy against mask fraction instead of "
                         "training: no optimizer, flow time held fixed")
    ap.add_argument("--resume", default="auto",
                    choices=("auto", "never"),
                    help="auto picks up results/<tag>.pt, which a "
                         "preempted job needs on restart")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    args.ckpt = repo_path(args.ckpt)
    args.ae = repo_path(args.ae)
    tag = args.tag or f"rl_{args.arm}"

    sys.path.insert(0, str(FRAGFM))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(FRAGFM)
    import exe.train_flow as tf
    from exe.train_flow import process_single_epoch
    from fragfm.dataset import collate_frag_fm_dataset
    from fragfm.distort_scheduler import DistortScheduler
    from fragfm.model.ae import FragJunctionAE
    from fragfm.model.flow import CoarseGraphPropagate, FragToVect
    from fragfm.utils.file import read_yaml_as_easydict
    from easydict import EasyDict as edict

    stem = Path(args.data).stem
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # read_yaml_as_easydict takes a path; the checkpoint's cfg is already a dict,
    # so it goes through EasyDict directly -- the same type train_flow.py expects,
    # since attribute access is how process_single_epoch reads every field.
    cfg = edict({k: v for k, v in ck["cfg"].items()
                 if k != "latent_transform_param"})
    cfg.latent_transform_param = ck["cfg"]["latent_transform_param"]

    ds = make_spectrum_dataset(args.data, f"data/processed/{stem}_fragment.lmdb",
                               f"data/processed/{stem}_fragment_to_idx.pkl",
                               "train")
    print(f"{tag}: {len(ds):,} train spectra, arm {args.arm}")

    ae_cfg = read_yaml_as_easydict("save/ae_model/npgen/cfg.yaml")
    ae = FragJunctionAE(ae_cfg).cuda().eval()
    ae.load_state_dict(torch.load(args.ae, map_location="cpu"))
    for p in ae.parameters():  # the latent target must not move
        p.requires_grad_(False)
    # Set in train_flow.py's main block, which we do not run.
    cfg.latent_z_dim = ae_cfg.latent_z_dim

    frag_embedder = FragToVect(cfg).cuda()
    coarse_gnn = CoarseGraphPropagate(cfg).cuda()
    frag_embedder.load_state_dict(ck["ema"]["frag_embedder"])
    coarse_gnn.load_state_dict(ck["ema"]["coarse_gnn"])
    cond_model = SpectrumEncoder(out_dim=cfg.embd_h_dim).cuda()
    cond_model.load_state_dict(ck["cond_model"], strict=False)

    # process_single_epoch writes module-level EMA globals; supply them rather
    # than let it fail on them.
    import copy

    tf.ema_frag_embedder = copy.deepcopy(frag_embedder).eval()
    tf.ema_coarse_gnn = copy.deepcopy(coarse_gnn).eval()
    tf.ema_cond_model = copy.deepcopy(cond_model).eval()
    for m in (tf.ema_frag_embedder, tf.ema_coarse_gnn, tf.ema_cond_model):
        for p in m.parameters():
            p.requires_grad_(False)

    params = (list(frag_embedder.parameters()) + list(coarse_gnn.parameters())
              + list(cond_model.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr)
    cfg.lr, cfg.lr_warmup, cfg.n_iter_done = args.lr, False, 0
    cfg.time_power = args.time_power
    scheds = (DistortScheduler(cfg.node_distort_schedule),
              DistortScheduler(cfg.edge_distort_schedule),
              DistortScheduler(cfg.latent_z_distort_schedule))

    torch.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=4, collate_fn=collate_frag_fm_dataset)
    run = tracking.init(tag, {**vars(args), "n_train": len(ds)})

    if args.sweep_t:
        # The Euler trajectory starts at t=0 with everything masked and ends at
        # t=1.  Training samples t uniformly, so if accuracy collapses as t falls,
        # the trajectory's earliest and most consequential commitments are made in
        # exactly the region training visits least -- and R33 ruled out the other
        # structural difference, wrong values in the revealed slots.
        print("     t  masked_frac  argmax_acc  frag_masked  entropy  n_slots")
        rows = []
        for t in (0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95):
            ag = fm = en = mf = 0.0
            n = slots = 0
            it2 = iter(loader)
            for _ in range(args.steps):
                try:
                    b = next(it2)
                except StopIteration:
                    break
                probe = {"model_t": t}
                with torch.no_grad():
                    process_single_epoch(
                        cfg, ae, frag_embedder, coarse_gnn, OneBatch(ds, b),
                        scheds, frag_occurance_source="train", optimizer=None,
                        cond_model=cond_model, rl=probe)
                o = probe["out"]
                m = o["crpt_h_mask"] & o["reachable"]
                if not bool(m.any()):
                    continue
                # Accuracy on slots that are still masked: what the model gets
                # right where it has to choose rather than copy.
                ag += float((o["h_logit"][m].argmax(-1)
                             == o["h_type"][m]).float().mean())
                fm += float(F.cross_entropy(o["h_logit"][m], o["h_type"][m]))
                en += policy_entropy(o["h_logit"].detach(), m)
                mf += float(o["crpt_h_mask"].float().mean())
                slots += int(m.sum())
                n += 1
            if n:
                # n_slots is the number of masked decisions the accuracy is
                # averaged over.  At t=0.95 barely anything is masked, so the row
                # rests on a handful of decisions and must not be read as a trend.
                rows.append({"t": t, "masked_frac": mf / n, "argmax_acc": ag / n,
                             "frag_masked": fm / n, "entropy": en / n,
                             "n_slots": slots})
                print(f"  {t:.2f}  {mf / n:11.3f}  {ag / n:10.4f}"
                      f"  {fm / n:11.4f}  {en / n:7.3f}  {slots:7d}", flush=True)
        (RESULTS / f"sweep_t_{tag}.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {RESULTS / ('sweep_t_' + tag + '.json')}")
        return

    # Resume. The cluster preempted tp-3 three times in twenty minutes and this
    # script had no resume, so each restart began again at step 1 and nothing
    # accumulated. train_flow_cond.py has had this since the E3 runs; train_rl did
    # not, which is the whole reason that job could never finish.
    ckpt_path = RESULTS / f"{tag}.pt"
    history, start_step = [], 1
    if args.resume == "auto" and ckpt_path.exists():
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "step" not in prev:
            print(f"{ckpt_path.name} has no step; starting over")
        else:
            frag_embedder.load_state_dict(prev["frag_embedder"])
            coarse_gnn.load_state_dict(prev["coarse_gnn"])
            cond_model.load_state_dict(prev["cond_model"])
            opt.load_state_dict(prev["optimizer"])
            torch.set_rng_state(prev["rng"].cpu())
            if prev.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(prev["cuda_rng"].cpu())
            history, start_step = prev.get("history", []), prev["step"] + 1
            drift = {k: (v, vars(args)[k]) for k, v in prev["args"].items()
                     if k in vars(args) and vars(args)[k] != v and k != "resume"}
            if drift:
                print(f"[warn] args changed since the checkpoint: {drift}")
            print(f"resumed at step {start_step}")
    if start_step > args.steps:
        print("already complete")
        tracking.finish(run)
        return

    def save(step):
        """Write then rename: a kill during torch.save would leave a truncated
        file and the next restart would have nothing."""
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"step": step,
                    "frag_embedder": frag_embedder.state_dict(),
                    "coarse_gnn": coarse_gnn.state_dict(),
                    "cond_model": cond_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "history": history,
                    "args": vars(args),
                    # The live weights, written under "ema" because that is what
                    # the generator loads (*_ema_best.pt) and eval_denovo reads.
                    # No EMA is maintained: this is a short fine-tune from an
                    # already-averaged checkpoint.
                    "ema": {"frag_embedder": frag_embedder.state_dict(),
                            "coarse_gnn": coarse_gnn.state_dict(),
                            "cond_model": cond_model.state_dict()},
                    "cfg": dict(cfg)}, tmp)
        os.replace(tmp, ckpt_path)

    it = iter(loader)
    for step in range(start_step, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # Pass 1: read what the model itself would put in each slot.  No optimizer,
        # so nothing is learned from this pass; it only produces the state.
        probe = {}
        with torch.no_grad():
            process_single_epoch(
                cfg, ae, frag_embedder, coarse_gnn, OneBatch(ds, batch), scheds,
                frag_occurance_source="train", optimizer=None,
                cond_model=cond_model, rl=probe)
        out1 = probe["out"]
        state_h = (predicted_ids(out1, args.temperature) if args.arm == "selfcond"
                   else out1["cur_frag_idxs"][out1["h_type"]])
        agree = float((state_h == out1["cur_frag_idxs"][out1["h_type"]])
                      .float().mean())

        # Pass 2: the same graph with those values in the revealed slots, supervised
        # toward the truth.
        rl = {"state_h": state_h}
        res = process_single_epoch(
            cfg, ae, frag_embedder, coarse_gnn, OneBatch(ds, batch), scheds,
            frag_occurance_source="train", optimizer=opt, cond_model=cond_model,
            rl=rl)

        # The comparison that isolates the hypothesis. `fragment_type_loss` runs
        # over every slot, and a revealed slot has its answer sitting in the input
        # -- trivial for `teacher`, actively misleading for `selfcond` -- so most
        # of the gap between the arms would be the revealed slots rather than the
        # model's ability to predict a masked one. Restricted to masked slots,
        # both arms are scored on the same question.
        o = rl["out"]
        m = o["crpt_h_mask"] & o["reachable"]
        frag_masked = (F.cross_entropy(o["h_logit"][m], o["h_type"][m]).item()
                       if bool(m.any()) else float("nan"))

        row = {"loss": res["loss"], "frag_loss": res["fragment_type_loss"],
               "frag_loss_masked": frag_masked,
               "masked_frac": float(m.float().mean()),
               "edge_loss": res["fragment_edge_loss"],
               "latent_loss": res["latent_loss"],
               # How often the model's own pick matches the truth at full
               # corruption: the per-slot accuracy that compounds into top-1.
               "state_agreement": agree,
               "policy_entropy": policy_entropy(o["h_logit"].detach(), m),
               "effective_vocab": fragment_usage(
                   state_h.tolist(), int(out1["cur_frag_idxs"].max()) + 1
               )["effective_vocab"]}
        history.append({"step": step, **row})
        if step == 1 or step % 25 == 0:
            print(f"  step {step:>5}  frag {row['frag_loss']:.4f}  "
                  f"masked {frag_masked:.4f}  agree {agree:.4f}  "
                  f"ent {row['policy_entropy']:.3f}  "
                  f"vocab {row['effective_vocab']:.1f}", flush=True)
        tracking.log(run, {f"sc/{k}": v for k, v in row.items()}, step=step)
        if step % 250 == 0:
            save(step)

    save(args.steps)
    tracking.finish(run)
    (RESULTS / f"train_{tag}.json").write_text(json.dumps(
        {"args": vars(args), "history": history}, indent=2))
    print(f"\nwrote {RESULTS / (tag + '.pt')}")


if __name__ == "__main__":
    main()
