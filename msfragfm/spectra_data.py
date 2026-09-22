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


# --- conditioning the flow -------------------------------------------------
#
# The spectrum rides on the coarse graph rather than in a parallel batch, so it
# survives PyG collation and reaches `process_single_epoch` without changing its
# signature beyond one optional model.  Peaks are padded to a fixed width so PyG
# concatenates [1, n_peaks] per graph into [bs, n_peaks].

COND_FIELDS = ("mz", "intensity", "peak_mask", "formula", "precursor_mz",
               "collision_energy", "adduct", "instrument")


def cond_inputs(coarse_graph):
    """Pull the spectrum fields back off a batched coarse graph."""
    return {k: getattr(coarse_graph, k) for k in COND_FIELDS}


def make_spectrum_dataset(lmdb_fn, frag_lmdb_fn, frag_smi_to_idx_fn, fold,
                          n_peaks=60):
    """FragFMDataset indexed by spectrum instead of by structure.

    One structure carries many spectra, so `keys` is rewritten to one entry per
    spectrum; the parent class then loads the right molecule for each.
    """
    from fragfm.dataset import FragFMDataset

    class SpectrumConditioned(FragFMDataset):
        def __init__(self):
            super().__init__(lmdb_fn, frag_lmdb_fn, frag_smi_to_idx_fn,
                             data_split=FOLD_TO_PREFIX[fold], debug=False)
            spec = MassSpecGymSpectra(self.env, fold=fold, n_peaks=n_peaks)
            smi_to_key = {}
            with self.env.begin() as txn:
                for k in self.keys:
                    smi_to_key[pickle.loads(txn.get(k))["smi"]] = k
            self.spec = spec
            self.keys = [smi_to_key[s] for s in spec.df.smiles]
            self.length = len(self.keys)
            self.n_peaks = n_peaks

        def __len__(self):
            return self.length

        def __getitem__(self, i):
            graph, coarse = super().__getitem__(i)
            s = self.spec[i]
            k = s["mz"].numel()
            mz = torch.zeros(1, self.n_peaks)
            inten = torch.zeros(1, self.n_peaks)
            mask = torch.zeros(1, self.n_peaks, dtype=torch.bool)
            mz[0, :k], inten[0, :k], mask[0, :k] = s["mz"], s["intensity"], True
            coarse.mz, coarse.intensity, coarse.peak_mask = mz, inten, mask
            coarse.formula = s["formula"].unsqueeze(0)
            coarse.precursor_mz = s["precursor_mz"].view(1)
            coarse.collision_energy = s["collision_energy"].view(1)
            coarse.adduct = s["adduct"].view(1)
            coarse.instrument = s["instrument"].view(1)
            return graph, coarse

    return SpectrumConditioned()
