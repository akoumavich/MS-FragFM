"""Patch the reference checkouts in place, idempotently.

We keep FragFM and ms-pred as pristine upstream clones and repair them here, so
the exact delta we depend on stays visible and drops out the moment upstream
fixes it.  Re-running is a no-op.
"""

import argparse
import os
import sys
from pathlib import Path

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[2]))

PATCHES = [
    dict(
        name="fragfm-reconstruct-remove-h",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "utils" / "mol_ops.py",
        why=(
            "Upstream bug at HEAD (dcc6652): process.py calls "
            "reconstruct_to_rdmol(..., remove_h=...) but the function has no such "
            "argument, so fragfm's entire preprocessing path raises TypeError. "
            "remove_h=False callers match the result against AddHs() output by atom "
            "index, so that path must skip both RemoveHs and the SMILES round-trip "
            "in valid_mol_can_with_seg (which reorders atoms)."
        ),
        old="""def reconstruct_to_rdmol(h, e_index, e, is_relaxed=False, get_largest=True, fix=False):""",
        new="""def reconstruct_to_rdmol(
    h, e_index, e, is_relaxed=False, get_largest=True, fix=False, remove_h=True
):""",
    ),
    dict(
        name="fragfm-reconstruct-remove-h-body",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "utils" / "mol_ops.py",
        why="Second half of the above: honour remove_h at the return site.",
        # `old` survives inside `new`, so anchor absence cannot detect this one.
        marker="if not remove_h:",
        old="""    if get_largest:
        # get largest connected component
        mol = valid_mol_can_with_seg(mol, largest_connected_comp=get_largest)
    else:
        mol = mol.GetMol()

    assert mol is not None
    mol = Chem.RemoveHs(mol)

    return mol""",
        new="""    if not remove_h:
        # Callers passing remove_h=False index into this molecule to build a
        # reordering map, so neither RemoveHs nor the SMILES round-trip inside
        # valid_mol_can_with_seg may run: both would change the atom order.
        mol = mol.GetMol()
        assert mol is not None
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol

    if get_largest:
        # get largest connected component
        mol = valid_mol_can_with_seg(mol, largest_connected_comp=get_largest)
    else:
        mol = mol.GetMol()

    assert mol is not None
    mol = Chem.RemoveHs(mol)

    return mol""",
    ),
    dict(
        name="fragfm-spectrum-conditioning",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why=(
            "Add an optional conditioning vector to the coarse GNN.  g_embd is "
            "built once from the fragment bag, the timestep and the latent, then "
            "carried through every layer, so a fourth slot reaches every node at "
            "every depth.  Off by default, so unconditional FragFM is unchanged."
        ),
        marker="use_spectrum_cond",
        old="""        # global feature embedder
        self.merge_embd_g = MLP(
            [cfg.embd_h_dim * 3, cfg.embd_h_dim, cfg.embd_h_dim],""",
        new="""        # global feature embedder
        self.merge_embd_g = MLP(
            [cfg.embd_h_dim * (4 if cfg.get("use_spectrum_cond", False) else 3),
             cfg.embd_h_dim, cfg.embd_h_dim],""",
    ),
    dict(
        name="fragfm-spectrum-conditioning-sig",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why="Second part: accept the vector.",
        marker="cond=None,",
        old="""        timestep,
        frag_zs,
        coarse_h_valency=None,
    ):""",
        new="""        timestep,
        frag_zs,
        coarse_h_valency=None,
        cond=None,
    ):""",
    ),
    dict(
        name="fragfm-spectrum-conditioning-cat",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "model" / "flow.py",
        why="Third part: concatenate it into the global feature.",
        marker="g_embd_parts",
        old="""        g_embd = torch.cat([frag_bag_embd_, timestep_embd, latent_z_embd], dim=1)""",
        new="""        g_embd_parts = [frag_bag_embd_, timestep_embd, latent_z_embd]
        if cond is not None:
            g_embd_parts.append(cond)
        g_embd = torch.cat(g_embd_parts, dim=1)""",
    ),
    dict(
        name="fragfm-single-lmdb-open",
        path=THIRD_PARTY / "FragFM" / "fragfm" / "mol_generator.py",
        why=(
            "FragFMGenerator opens the fragment LMDB, then hands the same path to "
            "FragFMDataset, which opens it again.  py-lmdb 2.x refuses a second "
            "open of one environment in a process; the lmdb==1.5.1 that FragFM "
            "pins allowed it.  Build the dataset first and share its handle."
        ),
        marker="Reuse the dataset's handle",
        old="""        # get fragment lmdb env
        self.frag_env = lmdb.open(
            cfg.frag_data_dirn,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            map_size=100000000,
        )
        self.n_all_frag = int(self.frag_env.stat()["entries"])  # exc. mask

        # get test set and loader
        self.test_set = FragFMDataset(
            lmdb_fn=self.cfg.data_dirn,
            frag_lmdb_fn=self.cfg.frag_data_dirn,
            frag_smi_to_idx_fn=self.cfg.frag_smi_to_idx_fn,
            data_split=self.cfg.fragment_bag,
            debug=self.cfg.debug,
        )""",
        new="""        # get test set and loader
        self.test_set = FragFMDataset(
            lmdb_fn=self.cfg.data_dirn,
            frag_lmdb_fn=self.cfg.frag_data_dirn,
            frag_smi_to_idx_fn=self.cfg.frag_smi_to_idx_fn,
            data_split=self.cfg.fragment_bag,
            debug=self.cfg.debug,
        )
        # Reuse the dataset's handle rather than opening a second environment on
        # the same path, which py-lmdb 2.x rejects.
        self.frag_env = self.test_set.frag_env
        self.n_all_frag = int(self.frag_env.stat()["entries"])  # exc. mask""",
    ),
    dict(
        name="fragfm-drop-unused-draw-import",
        paths=[
            THIRD_PARTY / "FragFM" / "process" / "process_fragment_from_lmdb.py",
            THIRD_PARTY / "FragFM" / "process" / "process_to_lmdb.py",
            THIRD_PARTY / "FragFM" / "exe" / "eval_ae.py",
            THIRD_PARTY / "FragFM" / "exe" / "train_ae.py",
        ],
        why=(
            "These four import rdkit.Chem.Draw and never use it.  Draw needs "
            "libXrender.so.1, absent from the cluster image and un-installable "
            "without root, so an unused import blocks the entire preprocessing "
            "and autoencoder path."
        ),
        old="from rdkit.Chem import QED, Crippen, Descriptors, Draw",
        new="from rdkit.Chem import QED, Crippen, Descriptors",
    ),
    dict(
        name="mspred-lazy-plot-import",
        path=THIRD_PARTY / "ms-pred" / "src" / "ms_pred" / "common" / "__init__.py",
        why=(
            "ms_pred.common eagerly imports plot_utils, which imports "
            "rdkit.Chem.Draw, which needs libXrender.so.1.  The cluster image "
            "lacks it and we cannot apt-install without root.  Nothing on the "
            "training or RL path draws molecules, so import it on demand."
        ),
        old="""from .plot_utils import plot_mol_as_vector, plot_compare_ms, plot_ms""",
        new='''_PLOT_NAMES = ("plot_mol_as_vector", "plot_compare_ms", "plot_ms")


def __getattr__(name):
    # plot_utils -> rdkit.Chem.Draw -> libXrender.so.1, absent on headless
    # cluster images.  Import only if a caller actually plots.
    if name in _PLOT_NAMES:
        from . import plot_utils

        return getattr(plot_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")''',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    n_applied = n_already = 0
    for p in PATCHES:
        for path in p.get("paths", [p.get("path")]):
            src = path.read_text(encoding="utf-8")
            label = f"{p['name']}:{path.name}" if "paths" in p else p["name"]
            # Deciding "already applied" is fiddly in both directions: `new` can
            # be a prefix of `old` (deleting a trailing import), and `old` can be
            # a substring of `new` (wrapping a block, which leaves the anchor in
            # place and invites re-application on every run).  A patch that wraps
            # its anchor must therefore give an explicit `marker` unique to the
            # patched state; the rest are decided by anchor absence.
            marker = p.get("marker")
            applied = marker in src if marker else (p["old"] not in src and p["new"] in src)
            if applied:
                print(f"  already  {label}")
                n_already += 1
                continue
            if p["old"] not in src:
                print(f"  STALE    {label}: anchor not found in {path}")
                sys.exit(2)
            if args.check:
                print(f"  pending  {label}")
                continue
            path.write_text(src.replace(p["old"], p["new"], 1), encoding="utf-8")
            print(f"  applied  {label}")
            n_applied += 1

    print(f"{n_applied} applied, {n_already} already present")


if __name__ == "__main__":
    main()
