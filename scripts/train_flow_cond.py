"""E3: train the spectrum-conditioned fragment-level flow.

Reuses FragFM's `process_single_epoch` rather than reimplementing it. Its
corruption logic -- antithetic time sampling, per-prior masking schedules, the
fragment-bag mask that keeps molecules independent within a batch -- is subtle,
and a transcription error would not raise, it would just train worse and we would
attribute that to conditioning.

The spectrum rides on the coarse graph as padded fixed-width tensors, so PyG
collation carries it through untouched.

`--cond none` trains the identical model with conditioning off. That is the
spectrum-blind control the v1.5 audit demands in the main table, and running it
from the same script means it cannot silently differ in anything else.
"""

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from msfragfm import tracking
from msfragfm.paths import RESULTS
from msfragfm.spectra_data import make_spectrum_dataset
from msfragfm.spectrum import SpectrumEncoder

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def load_train_flow():
    spec = importlib.util.spec_from_file_location(
        "fragfm_train_flow", FRAGFM / "exe" / "train_flow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--ae", default=None,
                    help="fine-tuned autoencoder; defaults to the MS-FragFM one")
    ap.add_argument("--arch", default="cfgs/train_flow/moses.yaml")
    ap.add_argument("--cond", default="spectrum", choices=["spectrum", "none"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-peaks", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=2000,
                    help="FragFM's 10000 was set for MOSES at 6.3k iters/epoch; "
                         "here an epoch is 757 iters, so it would not finish "
                         "warming up until epoch 13")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", default="auto", choices=["auto", "never"],
                    help="auto picks up the checkpoint if one is there, which is "
                         "what a preempted job needs on restart")
    args = ap.parse_args()
    tag = args.tag or f"flow_{args.cond}"

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    tf = load_train_flow()
    from fragfm.dataset import collate_frag_fm_dataset
    from fragfm.distort_scheduler import DistortScheduler
    from fragfm.model.ae import FragJunctionAE
    from fragfm.model.flow import CoarseGraphPropagate, FragToVect
    from fragfm.utils.file import read_yaml_as_easydict

    cfg = read_yaml_as_easydict(args.arch)
    cfg.use_spectrum_cond = args.cond == "spectrum"
    cfg.use_ema = False  # process_single_epoch reaches for module-level EMA globals
    cfg.is_resume = False
    cfg.n_iter_done = 0
    cfg.lr = args.lr
    cfg.lr_warmup_iter = args.warmup

    stem = Path(args.data).stem
    ds = {f: make_spectrum_dataset(
        args.data, f"data/processed/{stem}_fragment.lmdb",
        f"data/processed/{stem}_fragment_to_idx.pkl", fold=f, n_peaks=args.n_peaks)
        for f in ("train",)}
    loaders = {f: DataLoader(d, batch_size=args.bs, shuffle=True, num_workers=8,
                             collate_fn=collate_frag_fm_dataset, drop_last=True)
               for f, d in ds.items()}
    print(f"{tag}: {len(ds['train']):,} training spectra")

    ae_cfg = read_yaml_as_easydict("save/ae_model/npgen/cfg.yaml")
    ae = FragJunctionAE(ae_cfg)
    ae.load_state_dict(torch.load(
        args.ae or RESULTS / "ae_ft_brics.pt", map_location="cpu"))
    ae.cuda().eval()
    for p in ae.parameters():  # the latent target must not move under the flow
        p.requires_grad_(False)

    # Both are set in train_flow.py's main block, which we do not run.
    cfg.latent_z_dim = ae_cfg.latent_z_dim
    cache = RESULTS / f"latent_transform_{stem}.pkl"
    if cache.exists():
        cfg.latent_transform_param = pickle.loads(cache.read_bytes())
    else:
        # One pass over the training spectra to bound the latent range; cached,
        # because it depends only on the autoencoder and the structures.
        cfg.latent_transform_param = ae.get_min_max_transform(loaders["train"])
        cache.write_bytes(pickle.dumps(cfg.latent_transform_param))
    print(f"latent range [{cfg.latent_transform_param['min'].min():.2f}, "
          f"{cfg.latent_transform_param['max'].max():.2f}]")

    frag_embedder = FragToVect(cfg).cuda()
    coarse_gnn = CoarseGraphPropagate(cfg).cuda()
    cond_model = (SpectrumEncoder(out_dim=cfg.embd_h_dim).cuda()
                  if cfg.use_spectrum_cond else None)

    params = list(frag_embedder.parameters()) + list(coarse_gnn.parameters())
    if cond_model is not None:
        params += list(cond_model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=cfg.weight_decay)
    print(f"params: flow {sum(p.numel() for p in params) / 1e6:.1f}M"
          + (f" (of which spectrum encoder "
             f"{sum(p.numel() for p in cond_model.parameters()) / 1e6:.1f}M)"
             if cond_model else ""))

    run = tracking.init(tag, config={
        **vars(args),
        "n_train_spectra": len(ds["train"]),
        "n_fragments": ds["train"].n_total_frag,
        "params_total": sum(p.numel() for p in params),
        "params_spectrum_encoder":
            sum(p.numel() for p in cond_model.parameters()) if cond_model else 0,
        "n_base_frag": cfg.n_base_frag,
        "backbone": cfg.backbone_type,
    })

    scheds = (DistortScheduler(cfg.node_distort_schedule),
              DistortScheduler(cfg.edge_distort_schedule),
              DistortScheduler(cfg.latent_z_distort_schedule))

    ckpt_path = RESULTS / f"{tag}.pt"
    history, start_epoch = [], 1
    if args.resume == "auto" and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "optimizer" in ck:
            frag_embedder.load_state_dict(ck["frag_embedder"])
            coarse_gnn.load_state_dict(ck["coarse_gnn"])
            if cond_model is not None and ck.get("cond_model"):
                cond_model.load_state_dict(ck["cond_model"])
            opt.load_state_dict(ck["optimizer"])
            cfg.n_iter_done = ck["n_iter_done"]
            history = ck.get("history", [])
            start_epoch = ck["epoch"] + 1
            print(f"resumed from epoch {ck['epoch']} "
                  f"({cfg.n_iter_done:,} iters done)")
        else:
            print(f"{ckpt_path.name} predates resume support; starting over")
    if start_epoch > args.epochs:
        print("already complete")
        tracking.finish(run)
        return

    t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        r = tf.process_single_epoch(
            cfg, ae, frag_embedder, coarse_gnn, loaders["train"], scheds,
            frag_occurance_source="train", optimizer=opt, cond_model=cond_model)
        history.append({"epoch": epoch, **{k: v for k, v in r.items()}})
        print(f"  ep {epoch:>3}  loss {r['loss']:.4f}  frag {r['fragment_type_loss']:.4f}"
              f"  edge {r['fragment_edge_loss']:.4f}  z {r['latent_loss']:.4f}"
              f"  [{time.perf_counter() - t0:.0f}s]", flush=True)
        tracking.log(run, {
            "epoch": epoch,
            "loss/total": r["loss"],
            "loss/fragment_type": r["fragment_type_loss"],
            "loss/coarse_edge": r["fragment_edge_loss"],
            "loss/latent_z": r["latent_loss"],
            "lr": opt.param_groups[0]["lr"],
            "iters_done": cfg.n_iter_done,
            "epoch_seconds": r["time"],
        }, step=epoch)
        # Write then rename: a preemption during torch.save would otherwise
        # leave a truncated file, and the next restart would have nothing.
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"epoch": epoch,
                    "frag_embedder": frag_embedder.state_dict(),
                    "coarse_gnn": coarse_gnn.state_dict(),
                    "cond_model": cond_model.state_dict() if cond_model else None,
                    "optimizer": opt.state_dict(),
                    "n_iter_done": cfg.n_iter_done,
                    "history": history,
                    "cfg": dict(cfg)}, tmp)
        os.replace(tmp, ckpt_path)

    (RESULTS / f"train_{tag}.json").write_text(json.dumps(
        {"tag": tag, "args": vars(args), "history": history}, indent=2))
    tracking.finish(run)
    print(f"\nwrote {RESULTS / (tag + '.pt')}")


if __name__ == "__main__":
    main()
