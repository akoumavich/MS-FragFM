"""Fetch the MassSpecGym v1.5 files we actually need, and report the split.

The HF repo is 52.8 GB; we pull single files rather than snapshot it.
"""

import argparse

import pandas as pd
from huggingface_hub import hf_hub_download

from msfragfm.paths import RAW

REPO = "roman-bushuiev/MassSpecGym"
FILES = {
    "main": "data/MassSpecGym1.5.tsv",
    # MCES-2-disjoint from the test fold: the pretraining corpus filter the
    # proposal calls for, already built by the benchmark authors.
    "pretrain": "data/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv",
    "retrieval_formula": "data/molecules/MassSpecGym1.5_retrieval_candidates_formula.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["main"], choices=[*FILES, "all"])
    args = ap.parse_args()
    which = list(FILES) if "all" in args.which else args.which

    for key in which:
        path = hf_hub_download(
            REPO, FILES[key], repo_type="dataset", local_dir=RAW.parent / "_hf"
        )
        dest = RAW / FILES[key].split("/")[-1]
        if dest.is_symlink():
            dest.unlink()  # may be stale/broken, which .exists() reports as False
        if not dest.exists():
            dest.symlink_to(path)
        print(f"{key:18s} -> {dest}")

    if "main" in which:
        df = pd.read_csv(RAW / "MassSpecGym1.5.tsv", sep="\t")
        print(f"\nrows={len(df):,}  cols={list(df.columns)}")
        for col in ("fold", "adduct", "instrument_type"):
            if col in df:
                print(f"\n{col}:\n{df[col].value_counts().to_string()}")
        if "smiles" in df:
            print(f"\nunique smiles: {df['smiles'].nunique():,}")
        if "inchikey" in df:
            print(f"unique inchikey14: {df['inchikey'].str[:14].nunique():,}")


if __name__ == "__main__":
    main()
