"""E0-e: how much of the attachment decision does the global latent z carry?

R6 flagged that FragFM's only per-molecule generative variable is a single
continuous z, which the coarse-to-fine decoder turns into inter-fragment bonds.
If attachment errors route only to z, Method D's per-fragment advantages degrade
to a scalar for exactly the isomer confusions MS/MS resolves worst.

The disciplined version of this question sorts a trained model's errors into
"wrong fragment multiset" and "right multiset, wrong attachment".  That needs a
trained conditional model.  This is the training-free version, available now, and
it isolates the variable more cleanly -- the error sort cannot separate "z is a
bad carrier" from "the policy is undertrained":

    hold the coarse graph at ground truth, vary only z, watch reconstruction.

**Read the `prior` arm as a floor, not as a forecast of generator accuracy.** The
flow evolves z during sampling (`_calc_euler_step` sets
`z_rate = (pred_z - gen_z) / (1 - t)`), so the z reaching the decoder is a
learned function of the coarse-graph trajectory, not the prior draw it started
from.  `prior` answers "how much does the coarse graph alone determine", i.e.
whether z is redundant.  The *noised* arms answer the question that bounds the
generator: how precisely must z be predicted.

`--multiplicity K` asks the practically decisive follow-up: if attachment is
carried by z, is the true molecule recoverable by *searching* over z?  If K draws
usually contain it, attachment is a search problem the oracle can rank its way
out of rather than a representational one.

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
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter

from msfragfm.paths import RESULTS

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def draw_z(like, transform):
    """Generation-time z: a normal mapped through the min-max inverse that
    store_smis_from_coarse_graph applies.  The decoder never sees a raw normal."""
    z = torch.randn_like(like)
    if transform is not None:
        lo, hi = transform["min"].to(z.device), transform["max"].to(z.device)
        z = (z + 1) * 0.5 * (hi - lo) + lo
    return z


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
            z = draw_z(z, transform)
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


def decode_to_smiles(model, graph, z):
    """Decode one batch at the given z; canonical SMILES per molecule, None on
    reconstruction failure."""
    from rdkit import Chem

    from fragfm.utils.mol_ops import reconstruct_to_rdmol

    pred = torch.sigmoid(model.decode(
        z, graph.h, graph.h_junction_count, graph.h_in_frag_label,
        graph.h_aux_frag_label, graph.decomp_e_index, graph.decomp_e,
        graph.ae_to_pred_index, graph.batch,
    )) > 0.5
    pred_batch = graph.batch[graph.ae_to_pred_index[0]]
    decomp_batch = graph.batch[graph.decomp_e_index[0]]
    out, n_cum = [], 0
    for i in range(graph.batch.max().item() + 1):
        h = graph.h[graph.batch == i].cpu().numpy()
        sel = pred[pred_batch == i]
        new_idx = graph.ae_to_pred_index[:, pred_batch == i][:, sel] - n_cum
        base_idx = graph.decomp_e_index[:, decomp_batch == i] - n_cum
        base_e = graph.decomp_e[decomp_batch == i]
        n_cum += h.shape[0]
        try:
            e_index = torch.cat([base_idx, new_idx], dim=1).cpu().numpy()
            ones = torch.ones(int(sel.sum()), dtype=base_e.dtype, device=base_e.device)
            e = torch.cat([base_e, ones]).cpu().numpy()
            out.append(Chem.MolToSmiles(reconstruct_to_rdmol(h, e_index, e)))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def multiplicity(model, loader, transform, k, max_mols, seed=0):
    from rdkit import Chem

    torch.manual_seed(seed)
    distinct, recovered, scored, n = [], 0, 0, 0
    for graph in loader:
        if n >= max_mols:
            break
        graph.to("cuda")
        z0, _ = model.encode(
            graph.h, graph.h_junction_count, graph.h_in_frag_label,
            graph.h_aux_frag_label, graph.e_index, graph.e, graph.batch,
        )
        cands = [set() for _ in range(graph.batch.max().item() + 1)]
        for _ in range(k):
            for i, smi in enumerate(decode_to_smiles(model, graph, draw_z(z0, transform))):
                if smi is not None:
                    cands[i].add(smi)
        for i, cand in enumerate(cands):
            if n >= max_mols:
                break
            n += 1
            distinct.append(len(cand))
            try:
                truth = Chem.CanonSmiles(graph.smi[i])
            except Exception:  # noqa: BLE001
                continue
            scored += 1
            recovered += truth in cand
    return {
        "z_source": f"z_search_k{k}",
        "n_mols": n,
        "mean_distinct_molecules": sum(distinct) / max(len(distinct), 1),
        "max_distinct_molecules": max(distinct) if distinct else 0,
        "recall_at_k": recovered / max(scored, 1),
    }


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
    ap.add_argument("--multiplicity", type=int, default=0, help="K z-draws per molecule")
    ap.add_argument("--mult-n", type=int, default=512)
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

    tp = Path(args.flow) / "latent_transform_param.pkl"
    transform = pickle.loads(tp.read_bytes()) if tp.exists() else None
    if transform is None:
        print("[warn] no latent_transform_param.pkl; z draws use a raw normal")

    arms = ([("encoded", None)]
            + [("noised", s) for s in (0.1, 0.25, 0.5, 1.0)]
            + [("prior", None)])
    rows = []
    print(f"{len(ds)} molecules from {args.data} [{args.split}]\n")
    print(f"{'z source':<14} {'edge acc':>9} {'graph acc':>10}")
    with torch.no_grad():
        for mode, sigma in arms:
            r = evaluate(model, loader, mode, transform, sigma or 0.0, seed=0)
            label = mode + (f" s={sigma}" if sigma else "")
            rows.append({"z_source": label, **r})
            print(f"{label:<14} {r['edge_acc']:>9.4f} {r['graph_acc']:>10.4f}")

        enc, pri = rows[0]["graph_acc"], rows[-1]["graph_acc"]
        print(f"\nz is redundant iff prior ~= encoded: {pri:.4f} vs {enc:.4f}")
        print("Noised arms bound how precisely the flow must predict z; `prior` "
              "is a floor, not a generator forecast, since the flow evolves z.")

        if args.multiplicity:
            m = multiplicity(model, loader, transform, args.multiplicity, args.mult_n)
            rows.append(m)
            print(f"\nz-search, K={args.multiplicity} draws over {m['n_mols']} molecules:")
            print(f"  distinct molecules per coarse graph : "
                  f"{m['mean_distinct_molecules']:.1f} mean, "
                  f"{m['max_distinct_molecules']} max")
            print(f"  true molecule recovered within K    : {m['recall_at_k']:.3f}")

    path = RESULTS / f"e0e_attachment_{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
