"""Retrain the coarse-to-fine autoencoder, three arms.

| arm | `--loss` | `--z` | question |
| --- | --- | --- | --- |
| 1 | bce | encoded | FragFM's objective, retrained in-distribution: the fair baseline |
| 2 | atom_ce | encoded | does making candidates compete help on its own? |
| 3 | atom_ce | zero | **does the coarse graph determine attachment at all?** |

Arm 3 is the one that decides the architecture. R10/R11 showed that under
FragFM's objective a wrong-but-valid z costs ~73 points of exact reconstruction
-- but that was measured on a decoder *trained to lean on z*. If arm 3 recovers
arm 2's accuracy without z, attachment was determined by the coarse graph all
along, z can be dropped, and Method D's per-fragment credit assignment needs no
further work. If arm 3 is far worse, attachment is genuinely ambiguous and the
choice has to become a variable of the flow rather than an output of the decoder.

Evaluation is exact-molecule reconstruction under the **Blossom** decode, since
that is what the generation path uses; a 0.5 threshold flatters or punishes the
head for reasons the pipeline never sees.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from msfragfm.attachment import attachment_loss
from msfragfm.blossom import blossom_select
from msfragfm.paths import RESULTS

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def latents(model, graph, z_mode):
    z, logvar = model.encode(
        graph.h, graph.h_junction_count, graph.h_in_frag_label,
        graph.h_aux_frag_label, graph.e_index, graph.e, graph.batch,
    )
    if z_mode == "zero":
        return torch.zeros_like(z), None
    return z, logvar


def logits_for(model, graph, z):
    return model.decode(
        z, graph.h, graph.h_junction_count, graph.h_in_frag_label,
        graph.h_aux_frag_label, graph.decomp_e_index, graph.decomp_e,
        graph.ae_to_pred_index, graph.batch,
    )


@torch.no_grad()
def evaluate(model, loader, z_mode, max_batches=None):
    model.eval()
    n_g = ok_g = n_e = ok_e = 0
    for bi, graph in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        graph.to("cuda")
        logits = logits_for(model, graph, latents(model, graph, z_mode)[0])
        picked = blossom_select(logits, graph)
        wrong = (picked != graph.ae_to_pred.bool()).int()
        pb = graph.batch[graph.ae_to_pred_index[0]]
        from torch_geometric.utils import scatter

        wrong_g = (scatter(wrong, pb, reduce="sum") != 0).int()
        n_e += wrong.numel()
        ok_e += (1 - wrong).sum().item()
        n_g += wrong_g.numel()
        ok_g += (1 - wrong_g).sum().item()
    model.train()
    return {"edge_acc": ok_e / max(n_e, 1), "graph_acc": ok_g / max(n_g, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--loss", default="atom_ce", choices=["bce", "atom_ce"])
    ap.add_argument("--z", default="encoded", choices=["encoded", "zero"])
    ap.add_argument("--arch", default="cfgs/train_ae/moses.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-batches", type=int, default=4,
                    help="Blossom is CPU-bound; keep periodic eval short")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{Path(args.data).stem}_{args.loss}_z{args.z}"

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    from fragfm.dataset import FragJunctionAEDataset, collate_frag_junction_ae_dataset
    from fragfm.model.ae import FragJunctionAE
    from fragfm.utils.file import read_yaml_as_easydict

    cfg = read_yaml_as_easydict(args.arch)
    sets = {s: FragJunctionAEDataset(args.data, data_split=s, debug=False)
            for s in ("train", "test")}
    loaders = {
        s: DataLoader(d, batch_size=args.bs, shuffle=(s == "train"),
                      collate_fn=collate_frag_junction_ae_dataset,
                      num_workers=8, drop_last=(s == "train"))
        for s, d in sets.items()
    }
    print(f"{tag}: train {len(sets['train']):,} · test {len(sets['test']):,}")

    model = FragJunctionAE(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=cfg.weight_decay)

    history, best, it = [], 0.0, 0
    warmup = cfg.lr_warmup_iter
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tot = 0.0
        for graph in loaders["train"]:
            graph.to("cuda")
            z, logvar = latents(model, graph, args.z)
            logits = logits_for(model, graph, z)
            if args.loss == "bce":
                loss = F.binary_cross_entropy_with_logits(
                    logits, graph.ae_to_pred.float())
            else:
                loss = attachment_loss(logits, graph)
            if logvar is not None:  # VAE regulariser, as in FragFM
                loss = loss + cfg.reg_loss * (
                    -0.5 * torch.mean(1 + logvar - z.pow(2) - logvar.exp()))

            it += 1
            if it <= warmup:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * it / warmup
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item()

        if epoch % 10 == 0 or epoch == args.epochs:
            ev = evaluate(model, loaders["test"], args.z, args.eval_batches)
            history.append({"epoch": epoch, "loss": tot / len(loaders["train"]), **ev})
            print(f"  ep {epoch:>4}  loss {tot / len(loaders['train']):.4f}  "
                  f"test edge {ev['edge_acc']:.4f}  graph {ev['graph_acc']:.4f}  "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)
            if ev["graph_acc"] >= best:
                best = ev["graph_acc"]
                torch.save(model.state_dict(), RESULTS / f"ae_{tag}.pt")

    final = evaluate(model, loaders["test"], args.z)  # full test fold
    print(f"\n{tag} final (full test): edge {final['edge_acc']:.4f}  "
          f"graph {final['graph_acc']:.4f}")
    (RESULTS / f"train_ae_{tag}.json").write_text(json.dumps(
        {"tag": tag, "args": vars(args), "history": history, "final": final}, indent=2))


if __name__ == "__main__":
    main()
