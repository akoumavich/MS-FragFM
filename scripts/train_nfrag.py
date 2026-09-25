"""Predict the BRICS fragment count from the spectrum.

R28 measured the count handed to generation as exact 10% of the time, and R29 the
conditional median of p(n_frag | n_heavy) -- the best point estimate that prior
admits -- as exact 16% with a mean absolute error of 3.16, on molecules of about
seven fragments. Feeding the true count is worth 7.2 points of fragment recall,
so the prior itself is the binding part and no better use of it recovers the gap.

The spectrum should do better on mechanism rather than by capacity: a BRICS
fragment boundary is a cleavable bond, and MS/MS peaks arise from cleaving exactly
those bonds, so the peak pattern carries direct evidence about how many there are.
The heavy-atom total carries none -- a twenty-atom molecule can be three large
fragments or eight small ones.

This trains the smallest thing that tests it: the existing spectrum encoder with a
categorical head over the count. Categorical rather than a regression because the
predictive distribution is what the generation group needs -- a mode to commit to
and a spread to hedge over -- and because absolute error is minimised by the
predictive median, not the mean.

The baseline is recomputed on the same spectra, so the comparison does not depend
on R29 having used the same subset.
"""

import argparse
import json
import pickle

import lmdb
import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch import nn
from torch.utils.data import DataLoader

from msfragfm import tracking
from msfragfm.nfrag import frag_count_pool, frag_count_prior
from msfragfm.paths import RESULTS
from msfragfm.spectra_data import MassSpecGymSpectra, collate_spectra
from msfragfm.spectrum import SpectrumEncoder

RDLogger.DisableLog("rdApp.*")
MAX_N = 40  # BRICS counts run well under this; anything above is clipped in


def targets(batch):
    return torch.tensor([min(s["n_frag"], MAX_N) - 1 for s in batch["samples"]])


def metrics(logits, y):
    """Argmax and predictive-median error.

    The median minimises absolute error under the predicted distribution, so it is
    the estimate to use when the metric is absolute error; the argmax is what a
    group that commits to one count would take.
    """
    p = logits.softmax(-1)
    med = (p.cumsum(-1) < 0.5).sum(-1)
    am = logits.argmax(-1)
    return {
        "mae_argmax": (am - y).abs().float().mean().item(),
        "mae_median": (med - y).abs().float().mean().item(),
        "exact_argmax": (am == y).float().mean().item(),
        "signed_argmax": (am - y).float().mean().item(),
    }


def evaluate(model, loader, dev):
    model.eval()
    L, Y = [], []
    with torch.no_grad():
        for batch in loader:
            d = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
            L.append(model(d).cpu())
            Y.append(targets(batch))
    model.train()
    return metrics(torch.cat(L), torch.cat(Y))


def prior_baseline(ds, prior):
    """p(n_frag | n_heavy) on these exact spectra, for a paired comparison."""
    err, exact = [], []
    with ds.env.begin() as txn:
        for i in range(len(ds)):
            mol = Chem.MolFromSmiles(ds.df.iloc[i].smiles)
            pool = frag_count_pool(prior, mol.GetNumHeavyAtoms()) if mol else None
            if not pool:
                continue
            true = min(pickle.loads(txn.get(ds.keys[i]))["n_frag"], MAX_N)
            med = int(np.median(pool))
            err.append(abs(med - true))
            exact.append(med == true)
    return {"mae_median": float(np.mean(err)),
            "exact_median": float(np.mean(exact)), "n": len(err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="molecule LMDB; defaults to FragFM's msg_brics_all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-peaks", type=int, default=60)
    ap.add_argument("--tag", default="nfrag")
    args = ap.parse_args()

    data = args.data or str(
        RESULTS.parents[1] / "FragFM/data/processed/msg_brics_all.lmdb")
    # One env for every split: py-lmdb refuses a second open of one environment.
    env = lmdb.open(data, readonly=True, lock=False, readahead=True,
                    meminit=False, map_size=int(1e12))
    ds = {f: MassSpecGymSpectra(env, fold=f, n_peaks=args.n_peaks)
          for f in ("train", "val", "test")}
    for f, d in ds.items():
        print(f"{f}: {len(d):,} spectra")

    prior, _ = frag_count_prior(env)
    base = prior_baseline(ds["test"], prior)
    print(f"prior baseline on test: mae_median {base['mae_median']:.3f}  "
          f"exact {base['exact_median']:.3f}  (n={base['n']:,})")

    loaders = {f: DataLoader(d, batch_size=args.bs, shuffle=(f == "train"),
                             num_workers=8, collate_fn=collate_spectra,
                             drop_last=(f == "train"))
               for f, d in ds.items()}

    dev = "cuda"
    model = SpectrumEncoder(out_dim=MAX_N).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    run = tracking.init("nfrag", {**vars(args),
                                 "prior_mae_median": base["mae_median"]})

    history = []
    for epoch in range(1, args.epochs + 1):
        tot = k = 0
        for batch in loaders["train"]:
            d = {k_: v.to(dev) for k_, v in batch.items() if torch.is_tensor(v)}
            loss = nn.functional.cross_entropy(model(d), targets(batch).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            k += 1
        val = evaluate(model, loaders["val"], dev)
        history.append({"epoch": epoch, "loss": tot / k, **val})
        print(f"  ep {epoch:>3}  loss {tot / k:.4f}  "
              f"val mae_argmax {val['mae_argmax']:.3f}  "
              f"mae_median {val['mae_median']:.3f}  exact {val['exact_argmax']:.3f}",
              flush=True)
        tracking.log(run, {"epoch": epoch, "loss/ce": tot / k,
                           **{f"val/{k_}": v for k_, v in val.items()}}, step=epoch)

    test = evaluate(model, loaders["test"], dev)
    print("\ntest:")
    for k_, v in test.items():
        print(f"  {k_:<16s} {v:.4f}")
    print(f"  {'prior mae_median':<16s} {base['mae_median']:.4f}  (to beat)")
    tracking.log(run, {f"test/{k_}": v for k_, v in test.items()}, step=args.epochs)
    tracking.finish(run)

    torch.save({"model": model.state_dict(), "max_n": MAX_N, "args": vars(args)},
               RESULTS / f"{args.tag}.pt")
    (RESULTS / f"train_{args.tag}.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "test": test,
         "prior_baseline": base}, indent=2))
    print(f"\nwrote {RESULTS / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()
