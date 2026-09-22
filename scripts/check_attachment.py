"""Which granularity is attachment actually categorical at?

Three candidates, checked against the data rather than assumed:

* **per coarse edge** -- "which atom pair joins fragments A and B" -- is valid
  only if every coarse edge carries exactly one bond;
* **per atom, choose junction_count partners** -- valid by construction, and what
  `msfragfm.attachment` implements;
* the candidate-set sizes, which say how hard the choice is at all.

The coarse-edge grouping lives here rather than in the library because the
measurement retired it: rBRICS cuts rings, and a cut ring joins its fragments
twice.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter

from msfragfm.attachment import atom_candidates

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
_MAX_FRAG = 64  # rBRICS tops out at 45 fragments per molecule (RESULTS.md R2.1)


def coarse_edge_groups(graph):
    i, j = graph.ae_to_pred_index
    fa, fb = graph.h_frag_batch[i], graph.h_frag_batch[j]
    lo, hi = torch.minimum(fa, fb), torch.maximum(fa, fb)
    key = graph.batch[i] * _MAX_FRAG * _MAX_FRAG + lo * _MAX_FRAG + hi
    return torch.unique(key, return_inverse=True)[1]


def show(title, counts, note=""):
    tot = sum(counts.values())
    print(f"\n{title}:")
    for k in sorted(counts):
        print(f"  {k:>3}: {counts[k]:>9,}  ({counts[k] / tot:.4f})")
    if note:
        print(f"  {note}")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", default="10K", choices=["1K", "10K", "false"])
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    from fragfm.dataset import FragJunctionAEDataset, collate_frag_junction_ae_dataset

    ds = FragJunctionAEDataset(
        args.data, data_split=args.split,
        debug=False if args.limit == "false" else args.limit,
    )
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        collate_fn=collate_frag_junction_ae_dataset, num_workers=8)

    edge_true, edge_cand, atom_true, atom_cand, jc = (Counter() for _ in range(5))
    n_mols = 0
    for graph in loader:
        n_mols += int(graph.batch.max()) + 1
        j = graph.h_junction_count
        jc.update(j[j != 0].tolist())
        if graph.ae_to_pred_index.numel() == 0:
            continue

        g = coarse_edge_groups(graph)
        edge_true.update(scatter(graph.ae_to_pred.long(), g, reduce="sum").tolist())
        edge_cand.update(scatter(torch.ones_like(g), g, reduce="sum").tolist())

        _, ag, at = atom_candidates(graph)
        atom_true.update(scatter(at.long(), ag, reduce="sum").tolist())
        atom_cand.update(scatter(torch.ones_like(ag), ag, reduce="sum").tolist())

    print(f"{n_mols:,} molecules, {sum(edge_true.values()):,} coarse edges")

    show("true bonds per coarse edge", edge_true)
    print(f"\nper-coarse-edge categorical valid: {set(edge_true) == {1}}")

    tot = show("candidates per coarse edge", edge_cand)
    print(f"  {edge_cand[1] / tot:.4f} have one candidate (free); "
          f"uniform guess would score "
          f"{sum(v / k for k, v in edge_cand.items()) / tot:.4f}")

    show("junction count per junction atom", jc)
    at = show("true partners per atom", atom_true)
    print(f"\nper-atom choose-one exact for {atom_true[1] / at:.4f} of atoms; "
          f"the rest need choose-k")
    print(f"true partners per atom == junction count: "
          f"{dict(atom_true) == dict(jc)}")
    show("candidates per atom", atom_cand)


if __name__ == "__main__":
    main()
