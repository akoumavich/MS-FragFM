"""Verify the spectrum pipeline before anything trains on it.

Checks the join (do spectra reach the right structure?), the parsing (do peaks
and formulas come out sane?), and the encoder (does it run and produce finite,
non-degenerate output?).
"""

import argparse
import sys
from pathlib import Path

import lmdb
import torch
from torch.utils.data import DataLoader

from msfragfm.spectra_data import MassSpecGymSpectra, collate_spectra
from msfragfm.spectrum import ELEMENTS, SpectrumEncoder, parse_formula

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FRAGFM / "data/processed/msg_brics_all.lmdb"))
    ap.add_argument("--fold", default="train")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()
    sys.path.insert(0, str(FRAGFM))

    print("formula parsing:")
    for f in ("C8H10N4O2", "C27H44O3", "C6H12O6", "C22H19Cl2NO3"):
        c = parse_formula(f)
        named = {e: int(n) for e, n in zip(ELEMENTS, c) if n}
        print(f"  {f:16s} -> {named}  other={c[-1]:.0f}")

    env = lmdb.open(args.data, readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    ds = MassSpecGymSpectra(env, fold=args.fold)
    print(f"\n{len(ds):,} spectra in fold '{args.fold}'")

    # The join is the part that fails silently, so check it against the source.
    bad = 0
    import pickle
    with env.begin() as txn:
        for i in range(0, len(ds), max(1, len(ds) // 200)):
            want = ds.df.iloc[i].smiles
            got = pickle.loads(txn.get(ds.keys[i]))["smi"]
            bad += (want != got)
    print(f"join check: {bad} mismatches in 200 sampled spectra")

    loader = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        collate_fn=collate_spectra, num_workers=4)
    b = next(iter(loader))
    npk = b["peak_mask"].sum(1)
    print(f"\npeaks per spectrum: min {npk.min()} mean {npk.float().mean():.1f} "
          f"max {npk.max()}")
    print(f"m/z range {b['mz'][b['peak_mask']].min():.2f} - "
          f"{b['mz'][b['peak_mask']].max():.2f}")
    print(f"precursor m/z {b['precursor_mz'].min():.1f} - {b['precursor_mz'].max():.1f}")
    # A fragment cannot outweigh its precursor, but M+2 isotopes of Cl/Br/S
    # compounds legitimately sit a few Da above it.  Report the excess, not a
    # count: a few Da is chemistry, tens of Da is a parsing error.
    excess = (b["mz"] - b["precursor_mz"].unsqueeze(1))[b["peak_mask"]]
    over = excess[excess > 1.0]
    print(f"peaks above precursor+1 Da: {over.numel()}/{b['peak_mask'].sum()}"
          + (f", excess max {over.max():.2f} Da median {over.median():.2f} Da"
             if over.numel() else ""))
    print(f"adducts {b['adduct'].bincount().tolist()}  "
          f"instruments {b['instrument'].bincount().tolist()}")

    enc = SpectrumEncoder(out_dim=args.dim).cuda()
    dev = {k: v.cuda() for k, v in b.items() if torch.is_tensor(v)}
    with torch.no_grad():
        out = enc(dev)
    print(f"\nencoder out {tuple(out.shape)}  finite={torch.isfinite(out).all().item()}"
          f"  std={out.std():.4f}")
    # Distinct spectra must not collapse to one vector, or conditioning is inert.
    d = torch.cdist(out, out)
    print(f"pairwise distance mean {d[d > 0].mean():.4f} min {d[d > 0].min():.4f}")
    print(f"params: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    main()
