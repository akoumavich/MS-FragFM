"""MassSpecGym spectra joined to the fragment decomposition.

One structure carries many spectra -- 231,104 spectra over 31,602 structures --
so the decomposition is stored per structure and joined here by SMILES.  The
join is built by scanning the molecule LMDB once, rather than by re-deriving the
key assignment that `build_msg_lmdb.py` used, so the two cannot drift apart.

Spectra whose structure failed decomposition (0.13%) are dropped, and the count
is reported rather than swallowed.
"""

import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from msfragfm.paths import MSG_TSV
from msfragfm.spectrum import ADDUCTS, INSTRUMENTS, parse_formula

FOLD_TO_PREFIX = {"train": "train", "val": "valid", "test": "test"}


def _floats(s):
    return np.fromstring(str(s).strip("[]"), sep=",", dtype=np.float32)


class MassSpecGymSpectra(Dataset):
    def __init__(self, mol_lmdb, fold="train", n_peaks=60, min_intensity=0.0):
        self.env = mol_lmdb  # an already-open lmdb env; py-lmdb forbids reopening
        self.n_peaks = n_peaks

        prefix = FOLD_TO_PREFIX[fold]
        smi_to_key = {}
        with self.env.begin() as txn:
            for key, val in txn.cursor():
                if key.decode().startswith(prefix):
                    smi_to_key[pickle.loads(val)["smi"]] = key

        df = pd.read_csv(MSG_TSV, sep="\t")
        df = df[df.fold == fold]
        n_all = len(df)
        df = df[df.smiles.isin(smi_to_key)].reset_index(drop=True)
        if n_all != len(df):
            print(f"[{fold}] dropped {n_all - len(df):,}/{n_all:,} spectra whose "
                  f"structure failed decomposition")
        self.df = df
        self.keys = [smi_to_key[s] for s in df.smiles]
        self.min_intensity = min_intensity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mz, inten = _floats(row.mzs), _floats(row.intensities)
        if inten.max() > 0:
            inten = inten / inten.max()
        keep = inten >= self.min_intensity
        mz, inten = mz[keep], inten[keep]
        # Keep the most intense peaks, which is what the truncation is for.
        if len(mz) > self.n_peaks:
            top = np.argsort(inten)[-self.n_peaks:]
            mz, inten = mz[top], inten[top]

        with self.env.begin() as txn:
            sample = pickle.loads(txn.get(self.keys[i]))

        return {
            "sample": sample,
            "mz": torch.from_numpy(mz),
            # sqrt compresses the dynamic range without discarding small peaks,
            # which for fragment evidence are often the informative ones.
            "intensity": torch.from_numpy(np.sqrt(inten)),
            "formula": torch.tensor(parse_formula(row.precursor_formula), dtype=torch.float),
            "precursor_mz": torch.tensor(float(row.precursor_mz)),
            "collision_energy": torch.tensor(
                float(row.collision_energy) if pd.notna(row.collision_energy) else 0.0),
            "adduct": torch.tensor(
                ADDUCTS.index(row.adduct) if row.adduct in ADDUCTS else len(ADDUCTS)),
            "instrument": torch.tensor(
                INSTRUMENTS.index(row.instrument_type)
                if row.instrument_type in INSTRUMENTS else len(INSTRUMENTS)),
        }


def collate_spectra(items):
    n = max(x["mz"].numel() for x in items)
    out = {
        "mz": torch.zeros(len(items), n),
        "intensity": torch.zeros(len(items), n),
        "peak_mask": torch.zeros(len(items), n, dtype=torch.bool),
    }
    for i, x in enumerate(items):
        k = x["mz"].numel()
        out["mz"][i, :k] = x["mz"]
        out["intensity"][i, :k] = x["intensity"]
        out["peak_mask"][i, :k] = True
    for k in ("formula", "precursor_mz", "collision_energy", "adduct", "instrument"):
        out[k] = torch.stack([x[k] for x in items])
    out["samples"] = [x["sample"] for x in items]
    return out
