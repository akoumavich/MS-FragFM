"""Verify the assumption the categorical attachment head rests on.

If any coarse edge carries something other than exactly one true bond, attachment
is not a categorical variable per coarse edge and msfragfm/attachment.py is the
wrong formulation.  Checked before anything is trained on it.

Also reports how many candidates each coarse edge has, which is the real measure
of how hard attachment is: groups of size one are free.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from msfragfm.attachment import check_one_hot, coarse_edge_groups

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/msg_rbrics_all.lmdb")
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

    true_counts, cand_counts, jc_counts = Counter(), Counter(), Counter()
    atom_true, atom_cand = Counter(), Counter()
    n_groups = n_mols = 0
    for graph in loader:
        jc = graph.h_junction_count
        jc_counts.update(jc[jc != 0].tolist())
        # Per-source-atom grouping: an atom with junction count k needs exactly k
        # partners, so "pick one" only generalises if k is almost always 1.
        src = graph.ae_to_pred_index[0]
        if src.numel():
            _, agrp = torch.unique(src, return_inverse=True)
            from torch_geometric.utils import scatter
            atom_true.update(scatter(graph.ae_to_pred.long(), agrp, reduce="sum").tolist())
            atom_cand.update(scatter(torch.ones_like(agrp), agrp, reduce="sum").tolist())

        group = coarse_edge_groups(graph)
        if group.numel() == 0:
            n_mols += int(graph.batch.max()) + 1
            continue
        n_true, n_cand = check_one_hot(graph, group)
        true_counts.update(n_true.tolist())
        cand_counts.update(n_cand.tolist())
        n_groups += n_true.numel()
        n_mols += int(graph.batch.max()) + 1

    print(f"{n_mols:,} molecules, {n_groups:,} coarse edges\n")
    print("true bonds per coarse edge:")
    for k in sorted(true_counts):
        print(f"  {k}: {true_counts[k]:>9,}  ({true_counts[k] / n_groups:.5f})")

    ok = set(true_counts) == {1}
    print(f"\ncategorical formulation valid: {ok}")
    if not ok:
        print("  -> some coarse edge does not carry exactly one bond; "
              "attachment is not a per-edge categorical and the head needs rework")

    print("\ncandidates per coarse edge:")
    tot = sum(cand_counts.values())
    cum = 0
    for k in sorted(cand_counts):
        cum += cand_counts[k]
        print(f"  {k:>3}: {cand_counts[k]:>9,}  ({cand_counts[k] / tot:.4f})"
              f"  cumulative {cum / tot:.4f}")
    trivial = cand_counts[1] / tot
    mean_c = sum(k * v for k, v in cand_counts.items()) / tot
    print(f"\n{trivial:.4f} of coarse edges have a single candidate (free), "
          f"mean {mean_c:.2f} candidates")
    print("Chance accuracy of a uniform guess per edge: "
          f"{sum(v / k for k, v in cand_counts.items()) / tot:.4f}")

    # Slot-level view: a junction atom with count k occupies k slots and takes k
    # partners.  Slots always take exactly one, whatever the decomposition, so
    # they are the granularity that is decomposition-agnostic -- but the candidate
    # list is atom pairs, so k > 1 needs a choose-k head rather than choose-one.
    jt = sum(jc_counts.values())
    print("\njunction count per junction atom:")
    for k in sorted(jc_counts):
        print(f"  {k}: {jc_counts[k]:>9,}  ({jc_counts[k] / jt:.4f})")

    at = sum(atom_true.values())
    print("\ntrue partners per source atom:")
    for k in sorted(atom_true):
        print(f"  {k}: {atom_true[k]:>9,}  ({atom_true[k] / at:.4f})")
    print(f"\nchoose-one per atom is exact for {atom_true[1] / at:.4f} of atoms")


if __name__ == "__main__":
    main()
