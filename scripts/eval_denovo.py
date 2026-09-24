"""E3 evaluation: generate candidates per test spectrum and score top-k.

The first number here comparable to anything in the literature. Three things
about how it is measured, stated up front because each one can move the result
by more than the model does.

**Ranking is by sample frequency, not by an oracle.** Top-k is a ranked metric,
and no reranker is wired yet, so candidates are ordered by how often the model
produced them across the G samples -- the natural proxy for model probability.
When GLACIER goes in, this becomes oracle-ranked and the numbers are not
comparable across that change. Stated in the output so it cannot be mistaken.

**Both formula-free and formula-filtered numbers are reported.** The v1.5 audit
found formula pruning lifts a random baseline roughly thirty-fold, 0.43% to
12.98%, so a single number without saying which it is means little. The gap
between the two is also the most direct read on how much the formula is doing
versus the spectrum.

**Fragment count comes from the formula.** Generation needs n_frag per molecule,
which is unknown at test time. The formula gives the heavy-atom count exactly,
so n_frag is drawn from the empirical p(n_frag | n_heavy) measured on the train
fold -- the concrete form of what Method A means by the formula constraining the
fragment bag before generation begins.
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from msfragfm import assemble as _assemble
from msfragfm import tracking
from msfragfm.diversity import across_within_ratio, fragment_usage, group_metrics
from msfragfm.formula_mask import (admissible, fragment_counts, report,
                                   target_counts)
from msfragfm.peak_explain import explained_from_coarse, pool_fragment_masses
from msfragfm.paths import RESULTS
from msfragfm.spectrum import SpectrumEncoder

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
RDLogger.DisableLog("rdApp.*")


def export_for_generator(ckpt, out_dir, ae_dir):
    """Write our checkpoint in the layout FragFMGenerator expects.

    Exporting rather than patching the loader keeps the released generation path
    -- Euler loop, per-step bag resampling, Blossom assembly -- exactly as tested.
    EMA weights, since that is what the generator loads and what R13's autoencoder
    was paired with.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ema = ck["ema"]
    torch.save(ema["frag_embedder"], out_dir / "frag_embedder_ema_best.pt")
    torch.save(ema["coarse_gnn"], out_dir / "coarse_propagate_ema_best.pt")
    (out_dir / "latent_transform_param.pkl").write_bytes(
        pickle.dumps(ck["cfg"]["latent_transform_param"]))
    import yaml

    cfg = {k: v for k, v in ck["cfg"].items() if k != "latent_transform_param"}
    cfg["ae_model_dirn"] = str(ae_dir)
    (out_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return ck


def frag_count_prior(env):
    """p(n_frag | n_heavy) from the train fold."""
    prior = defaultdict(list)
    with env.begin() as txn:
        for key, val in txn.cursor():
            if not key.decode().startswith("train"):
                continue
            smp = pickle.loads(val)
            mol = Chem.MolFromSmiles(smp["smi"])
            if mol is not None:
                prior[mol.GetNumHeavyAtoms()].append(int(smp["n_frag"]))
    return prior


def draw_n_frag(prior, n_heavy, rng):
    for d in (0, 1, 2, 3, 5, 8):  # widen the window until the bucket is populated
        pool = [v for k in range(n_heavy - d, n_heavy + d + 1) for v in prior.get(k, [])]
        if len(pool) >= 20:
            return int(rng.choice(pool))
    return max(2, round(n_heavy / 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(RESULTS / "flow_spectrum.pt"))
    ap.add_argument("--ae", default=str(RESULTS / "ae_ft_brics.pt"))
    ap.add_argument("--data", default="data/processed/msg_brics_all.lmdb")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--n-spectra", type=int, default=500)
    ap.add_argument("--group", type=int, default=16, help="candidates per spectrum")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--spectra-per-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tree-assembly", default="on", choices=["on", "off"],
                    help="realise one atom-level bond per coarse edge")
    ap.add_argument("--rank", default="frequency", choices=["frequency", "peaks"],
                    help="peaks ranks by explained peak intensity -- the "
                         "conservation law of Method C, applied exactly")
    ap.add_argument("--composition", default="off", choices=["on", "off"],
                    help="project the final assignment onto sum(counts)==formula")
    ap.add_argument("--tree-decode", default="on", choices=["on", "off"],
                    help="project coarse edges onto a spanning tree at the last "
                         "step; BRICS coarse graphs are trees without exception")
    ap.add_argument("--formula-mask", default="on", choices=["on", "off"],
                    help="enforce the formula in the support (section 9)")
    ap.add_argument("--cond-mode", default="real",
                    choices=["real", "shuffled", "zero"],
                    help="shuffled gives each spectrum another one's embedding: "
                         "in-distribution by construction, so if the metrics do "
                         "not move, conditioning is inert at generation time")
    ap.add_argument("--tag", default="eval_spectrum")
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    from fragfm.mol_generator import FragFMGenerator
    from fragfm.utils.file import read_yaml_as_easydict

    stem = Path(args.data).stem
    model_dir = Path("save/flow_model/msfragfm_" + Path(args.ckpt).stem)
    ae_dir = Path("save/ae_model/msfragfm_ft")
    ae_dir.mkdir(parents=True, exist_ok=True)
    # The generator loads the autoencoder from a directory, so mirror ours there.
    import shutil

    shutil.copy("save/ae_model/npgen/cfg.yaml", ae_dir / "cfg.yaml")
    shutil.copy(args.ae, ae_dir / "model_best.pt")
    ck = export_for_generator(Path(args.ckpt), model_dir, ae_dir)

    gcfg = read_yaml_as_easydict("cfgs/generate/npgen.yaml")
    gcfg.data_dirn = args.data
    gcfg.frag_data_dirn = f"data/processed/{stem}_fragment.lmdb"
    gcfg.frag_smi_to_idx_fn = f"data/processed/{stem}_fragment_to_idx.pkl"
    gcfg.fm_model_dirn = str(model_dir)
    gcfg.n_euler_step = args.steps
    gcfg.bs = args.spectra_per_batch * args.group
    gcfg.n_sample = gcfg.bs
    gcfg.fragment_bag = "train"
    gcfg.force_save_dirn = gcfg.save_dirn = str(RESULTS / "_eval_scratch")
    os.makedirs(gcfg.save_dirn, exist_ok=True)

    # Before the generator: it opens the fragment LMDB and py-lmdb refuses a
    # second open of one environment.  Cached, so later runs pay nothing.
    fcounts = None
    if args.formula_mask == "on":
        fcounts = fragment_counts(gcfg.frag_data_dirn,
                                  cache=RESULTS / f"frag_counts_{stem}.npy")
        print(f"fragment element counts: {fcounts.shape[0]:,} fragments")

    pool_masses = None
    if args.rank == "peaks":
        pool_masses = pool_fragment_masses(
            gcfg.frag_data_dirn, cache=RESULTS / f"frag_masses_{stem}.npy")

    sampler = FragFMGenerator(gcfg)
    sampler.set_seed(args.seed)
    sampler.spanning_tree_decode = args.tree_decode == "on"
    _assemble.ENABLED = args.tree_assembly == "on"
    sampler.track_components = True
    sampler.component_counts = []

    cond_model = SpectrumEncoder(out_dim=ck["cfg"]["embd_h_dim"]).cuda().eval()
    cond_model.load_state_dict(ck["ema"]["cond_model"] or ck["cond_model"])

    # FragFMGenerator already opened the molecule LMDB, and py-lmdb refuses a
    # second open of one environment, so share its handle.
    env = sampler.test_set.env
    prior = frag_count_prior(env)
    rng = np.random.default_rng(args.seed)

    from msfragfm.spectra_data import MassSpecGymSpectra, collate_spectra

    spec_ds = MassSpecGymSpectra(env, fold=args.fold)
    # Shuffle before truncating: the folds are not stored in a representative
    # order, and a prefix scored ~11 points high on the autoencoder (R12).
    idx = list(range(len(spec_ds)))
    random.Random(args.seed).shuffle(idx)
    idx = idx[:args.n_spectra]
    print(f"{len(idx)} of {len(spec_ds):,} {args.fold} spectra "
          f"x G={args.group} at {args.steps} steps")

    rows, groups, all_frags = [], [], []
    B = args.spectra_per_batch
    for start in range(0, len(idx), B):
        chunk = idx[start:start + B]
        batch = collate_spectra([spec_ds[i] for i in chunk])
        dev = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            cond = cond_model(dev)
        if args.cond_mode == "shuffled" and cond.size(0) > 1:
            cond = cond[torch.randperm(cond.size(0), device=cond.device)]
        elif args.cond_mode == "zero":
            cond = torch.zeros_like(cond)
        sampler.cond = cond.repeat_interleave(args.group, dim=0)

        if fcounts is not None:
            adm = admissible(fcounts,
                             target_counts([spec_ds.df.iloc[i].formula for i in chunk]))
            if start == 0:
                print(f"  formula mask: {report(adm, fcounts)}")
            sampler.frag_admissible = torch.from_numpy(
                adm).cuda().repeat_interleave(args.group, dim=0)
        if args.composition == "on":
            tc = target_counts([spec_ds.df.iloc[i].formula for i in chunk])
            sampler.composition_counts = fcounts
            sampler.composition_targets = np.repeat(tc, args.group, axis=0)

        ns = []
        for i in chunk:
            mol = Chem.MolFromSmiles(spec_ds.df.iloc[i].smiles)
            nh = mol.GetNumHeavyAtoms()
            ns += [draw_n_frag(prior, nh, rng) for _ in range(args.group)]

        x = sampler.sample_molecule_graph_dynamic(n_frags=ns)
        # Heavy atoms the chosen fragments carry, before assembly. The projection
        # constrains this; every later metric is measured on the molecule that
        # comes out of the coarse-to-fine decode, and reconstruct_to_rdmol keeps
        # only the largest connected component. The gap between the two is
        # therefore mass lost in assembly rather than mass never chosen.
        chosen, cbatch = x[0].cpu().numpy(), x[-1].cpu().numpy()
        # Explained peak intensity per candidate, from the generated coarse graph
        # directly -- no re-decomposition needed, the sampler already has it.
        peak_scores = None
        if pool_masses is not None:
            ce_index = x[1].cpu().numpy()
            ce_type = x[2].cpu().numpy()
            peak_scores = []
            for n_, m in enumerate(np.unique(cbatch)):
                nodes = np.flatnonzero(cbatch == m)
                lo = nodes.min()
                sel = (ce_type == 1) & np.isin(ce_index[0], nodes)
                edges = ce_index[:, sel] - lo
                row_ = spec_ds.df.iloc[chunk[n_ // args.group]]
                peak_scores.append(explained_from_coarse(
                    chosen[nodes], edges, pool_masses,
                    _floats_mz(row_.mzs), _floats_mz(row_.intensities),
                    row_.adduct, max_cuts=1))
            peak_scores = np.array(peak_scores)
        intended = None
        if fcounts is not None:
            intended = np.array([fcounts[chosen[cbatch == m]].sum()
                                 for m in np.unique(cbatch)])

        cands = sampler.store_smis_from_coarse_graph(*x)
        all_frags += [int(t) for t in x[0].cpu().tolist()]

        for j, i in enumerate(chunk):
            row = spec_ds.df.iloc[i]
            g = cands[j * args.group:(j + 1) * args.group]
            groups.append(g)
            ps = (peak_scores[j * args.group:(j + 1) * args.group]
                  if peak_scores is not None else None)
            r = score_group(g, row, peak_scores=ps)
            if ps is not None:
                r["explained_peaks_mean"] = float(np.mean(ps))
                r["explained_peaks_max"] = float(np.max(ps))
            if intended is not None:
                sl = slice(j * args.group, (j + 1) * args.group)
                n_true = Chem.MolFromSmiles(row.smiles).GetNumHeavyAtoms()
                r["intended_heavy_err"] = float(np.mean(intended[sl] - n_true))
                r["assembly_atom_loss"] = float(
                    np.mean(intended[sl]) - (n_true + r["heavy_atom_signed_err"]))
            rows.append(r)
        print(f"  {start + len(chunk):>5}/{len(idx)}  "
              f"top1={np.mean([r['top1'] for r in rows]):.4f} "
              f"top1_f={np.mean([r['top1_formula'] for r in rows]):.4f}",
              flush=True)
        sampler.cond = None

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    summary["n_spectra"] = len(rows)
    summary.update(across_within_ratio(groups))
    summary.update(fragment_usage(all_frags, sampler.n_all_frag))
    cc = np.array([c for c in sampler.component_counts if c > 0])
    if cc.size:
        summary["assembly_components_mean"] = float(cc.mean())
        summary["assembly_connected_frac"] = float((cc == 1).mean())
    summary["tree_assembly"] = args.tree_assembly
    summary["ranking"] = args.rank
    summary["formula_mask"] = args.formula_mask
    summary["tree_decode"] = args.tree_decode
    summary["composition"] = args.composition
    if args.composition == "on":
        summary["projection_rate"] = float(
            getattr(sampler, "last_projection_rate", float("nan")))
    summary["cond_mode"] = args.cond_mode

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:<34} {v if isinstance(v, str) else round(v, 4)}")
    path = RESULTS / f"{args.tag}.json"
    path.write_text(json.dumps({"summary": summary, "args": vars(args)}, indent=2))
    print(f"\nwrote {path}")

    run = tracking.init(args.tag, config=vars(args))
    tracking.log(run, {f"eval/{k}": v for k, v in summary.items()
                       if not isinstance(v, str)})
    tracking.finish(run)


def _floats_mz(s):
    return np.fromstring(str(s).strip("[]"), sep=",", dtype=np.float64)


def score_group(cands, row, peak_scores=None):
    """Score top-k with and without the formula filter.

    Ranked by explained peak intensity when `peak_scores` is given, otherwise by
    how often the model produced each candidate.  Frequency is a proxy for model
    probability; peak explanation is the evidence itself, and it needs no oracle.
    """
    target = row.formula
    counts, best = Counter(), {}
    for i, c in enumerate(cands):
        if not c or c == "X":
            continue
        try:
            m = Chem.MolFromSmiles(c)
            if m is None:
                continue
            cs = Chem.MolToSmiles(m)
        except Exception:  # noqa: BLE001
            continue
        counts[cs] += 1
        if peak_scores is not None:
            best[cs] = max(best.get(cs, -1.0), float(peak_scores[i]))
    if peak_scores is not None and best:
        ranked = sorted(best, key=lambda k: -best[k])
    else:
        ranked = [s for s, _ in counts.most_common()]
    on_formula = [s for s in ranked
                  if CalcMolFormula(Chem.MolFromSmiles(s)).rstrip("+-") == target]
    truth = Chem.CanonSmiles(row.smiles)

    # How far off is the composition?  Containment being necessary but not
    # sufficient, the question is whether the candidates are the right *size*
    # with the wrong elements, or the wrong size entirely -- those want
    # different constraints.
    n_target = Chem.MolFromSmiles(row.smiles).GetNumHeavyAtoms()
    heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in ranked]
    out = {"validity": sum(counts.values()) / max(len(cands), 1),
           "n_unique": len(ranked),
           "formula_match_rate": len(on_formula) / max(len(ranked), 1),
           "heavy_atom_abs_err": float(np.mean([abs(h - n_target) for h in heavy]))
                                 if heavy else 0.0,
           "heavy_atom_signed_err": float(np.mean([h - n_target for h in heavy]))
                                    if heavy else 0.0,
           "heavy_atom_exact": float(np.mean([h == n_target for h in heavy]))
                               if heavy else 0.0}
    for k in (1, 10):
        out[f"top{k}"] = float(truth in ranked[:k])
        out[f"top{k}_formula"] = float(truth in on_formula[:k])
    m = group_metrics(cands, [counts.get(c, 0) for c in cands], row.smiles)
    out["mean_pairwise_tanimoto"] = m["mean_pairwise_tanimoto"]
    out["max_tanimoto_to_truth"] = m.get("max_tanimoto_to_truth", 0.0)
    return out


if __name__ == "__main__":
    main()
