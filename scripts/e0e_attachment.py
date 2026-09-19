"""E0-e: how much of the attachment decision does the global latent z carry?

R6 flagged that FragFM's only per-molecule generative variable is a single
continuous z, which the coarse-to-fine decoder turns into inter-fragment bonds.
If attachment errors route only to z, Method D's per-fragment advantages degrade
to a scalar for exactly the isomer confusions MS/MS resolves worst.

The disciplined version of this question is to sort a trained model's errors into
"wrong fragment multiset" and "right multiset, wrong attachment".  That needs a
trained conditional model.  This is the training-free lower-bound version, and it
is available now:

    hold the coarse graph at ground truth, vary only z, and watch reconstruction.

If reconstruction from a prior-sampled z matches reconstruction from the encoded
z, then attachment is determined by the coarse graph and z carries nothing --
the concern is void and no architecture change is needed.  If it collapses, z
carries real attachment information that per-fragment credit cannot reach, and
the fix (discrete attachment sites, or a per-bond latent) belongs in week 3.

The `prior` arm reproduces generation-time z exactly, including the min-max
inverse that `mol_generator.store_smis_from_coarse_graph` applies -- the
generator does not feed a standard normal to the decoder.

FragFM's own eval_ae.py carries a commented-out `z + randn*0.2` probe, so the
authors asked this question and did not publish the answer.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter

from msfragfm.paths import RESULTS

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def evaluate(model, loader, z_mode, transform, sigma, seed):
    torch.manual_seed(seed)
    n_edge = n_graph = n_ok_edge = n_ok_graph = 0
    for graph in loader:
        graph.to("cuda")
        z, _ = model.encode(
            graph.h, graph.h_junction_count, graph.h_in_frag_label,
            graph.h_aux_frag_label, graph.e_index, graph.e, graph.batch,
        )
        if z_mode == "noised":
            z = z + torch.randn_like(z) * sigma
        elif z_mode == "prior":
            z = torch.randn_like(z)
            if transform is not None:
                lo, hi = transform["min"].to(z.device), transform["max"].to(z.device)
                z = (z + 1) * 0.5 * (hi - lo) + lo
        pred = torch.sigmoid(model.decode(
            z, graph.h, graph.h_junction_count, graph.h_in_frag_label,
            graph.h_aux_frag_label, graph.decomp_e_index, graph.decomp_e,
            graph.ae_to_pred_index, graph.batch,
        ))
        wrong = ((pred > 0.5) != graph.ae_to_pred).int()
        pred_batch = graph.batch[graph.ae_to_pred_index[0]]
        wrong_graph = (scatter(wrong, pred_batch, reduce="sum") != 0).int()
        n_edge += wrong.numel()
        n_graph += wrong_graph.numel()
        n_ok_edge += (1 - wrong).sum().item()
        n_ok_graph += (1 - wrong_graph).sum().item()
    return {"edge_acc": n_ok_edge / n_edge, "graph_acc": n_ok_graph / n_graph,
            "n_graph": n_graph}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="save/ae_model/npgen")
    ap.add_argument("--flow", default="save/flow_model/npgen",
                    help="only for latent_transform_param.pkl")
    ap.add_argument("--data", default="data/processed/npgen_brics_all.lmdb")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", default="10K", choices=["1K", "10K", "false"])
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tag", default="npgen")
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    from fragfm.dataset import FragJunctionAEDataset, collate_frag_junction_ae_dataset
    from fragfm.model.ae import FragJunctionAE
    from fragfm.utils.file import read_yaml_as_easydict

    cfg = read_yaml_as_easydict(os.path.join(args.ae, "cfg.yaml"))
    ds = FragJunctionAEDataset(
        args.data, data_split=args.split,
        debug=False if args.limit == "false" else args.limit,
    )
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        collate_fn=collate_frag_junction_ae_dataset, num_workers=8)
    model = FragJunctionAE(cfg)
    model.load_state_dict(torch.load(os.path.join(args.ae, "model_best.pt")))
    model.cuda().eval()

    tp_path = Path(args.flow) / "latent_transform_param.pkl"
    transform = pickle.loads(tp_path.read_bytes()) if tp_path.exists() else None
    if transform is None:
        print("[warn] no latent_transform_param.pkl; prior arm uses a raw normal")

    arms = [("encoded", None)] + [("noised", s) for s in (0.1, 0.25, 0.5, 1.0)] + [("prior", None)]
    rows = []
    print(f"{len(ds)} molecules from {args.data} [{args.split}]\n")
    print(f"{'z source':<14} {'edge acc':>9} {'graph acc':>10}")
    with torch.no_grad():
        for mode, sigma in arms:
            r = evaluate(model, loader, mode, transform, sigma or 0.0, seed=0)
            label = f"{mode}" + (f" s={sigma}" if sigma else "")
            rows.append({"z_source": label, **r})
            print(f"{label:<14} {r['edge_acc']:>9.4f} {r['graph_acc']:>10.4f}")

    enc = rows[0]["graph_acc"]
    pri = rows[-1]["graph_acc"]
    print(f"\nattachment information carried by z: "
          f"{enc:.4f} encoded -> {pri:.4f} prior, drop = {enc - pri:.4f}")

    path = RESULTS / f"e0e_attachment_{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
