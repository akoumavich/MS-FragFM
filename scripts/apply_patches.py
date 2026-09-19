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
