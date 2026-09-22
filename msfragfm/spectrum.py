"""Spectrum encoder: raw peaks, formula, adduct.

Raw peaks rather than a MIST fingerprint, for three reasons. The fingerprint is
FRIGID's own stated limitation and it has a "direct spectra conditioning"
ablation for exactly that. A public MIST checkpoint is also one of the four
leakage vectors the v1.5 audit names, and a frozen encoder trained on
overlapping data contaminates a clean split. And it is less code. The fingerprint
path stays available as an ablation.

Two design points that are domain-specific rather than architectural:

**Neutral losses are encoded alongside peaks.** A fragment is identified as much
by precursor_mz - peak_mz as by peak_mz itself, and the two are not recoverable
from each other by a permutation-invariant set encoder without the precursor.
Both are embedded per peak, which costs nothing.

**m/z is embedded with Fourier features spanning sub-Da to whole-spectrum
scales.** Mass *differences* carry the structural signal, and a linear or
normalised scalar makes small differences between large masses numerically
invisible.
"""

import math
import re

import torch
from torch import nn

# Elements covered by MassSpecGym formulas; anything else falls in "other".
ELEMENTS = ("C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Se", "Si", "B", "As")
ADDUCTS = ("[M+H]+", "[M+Na]+")
INSTRUMENTS = ("Orbitrap", "QTOF")

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str):
    counts = [0.0] * (len(ELEMENTS) + 1)
    for sym, num in _FORMULA_RE.findall(formula or ""):
        if not sym:
            continue
        n = int(num) if num else 1
        idx = ELEMENTS.index(sym) if sym in ELEMENTS else len(ELEMENTS)
        counts[idx] += n
    return counts


class FourierMass(nn.Module):
    """Fourier features over m/z, log-spaced from sub-Da to whole-spectrum."""

    def __init__(self, n_freq=64, min_period=0.02, max_period=2000.0):
        super().__init__()
        periods = torch.logspace(
            math.log10(min_period), math.log10(max_period), n_freq // 2
        )
        self.register_buffer("omega", 2 * math.pi / periods)
        self.out_dim = n_freq

    def forward(self, mz):
        x = mz.unsqueeze(-1) * self.omega
        return torch.cat([x.sin(), x.cos()], dim=-1)


class SpectrumEncoder(nn.Module):
    def __init__(self, out_dim, d_model=256, n_layers=4, n_heads=8, n_freq=64):
        super().__init__()
        self.mass = FourierMass(n_freq)
        # per peak: m/z, neutral loss, intensity
        self.peak_in = nn.Sequential(
            nn.Linear(2 * n_freq + 1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        # No positional encoding: a spectrum is a set, and peak order is arbitrary.
        self.peaks = nn.TransformerEncoder(layer, n_layers)
        self.pool = nn.Linear(d_model, 1)

        self.formula = nn.Sequential(
            nn.Linear(len(ELEMENTS) + 1 + 2, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.adduct = nn.Embedding(len(ADDUCTS) + 1, d_model)
        self.instrument = nn.Embedding(len(INSTRUMENTS) + 1, d_model)
        self.merge = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, out_dim)
        )

    def forward(self, batch):
        mz, inten, mask = batch["mz"], batch["intensity"], batch["peak_mask"]
        prec = batch["precursor_mz"].unsqueeze(1)
        feat = torch.cat(
            [self.mass(mz), self.mass((prec - mz).clamp_min(0.0)), inten.unsqueeze(-1)],
            dim=-1,
        )
        h = self.peaks(self.peak_in(feat), src_key_padding_mask=~mask)
        w = self.pool(h).masked_fill(~mask.unsqueeze(-1), -1e4).softmax(1)
        peaks = (w * h).sum(1)

        f = self.formula(torch.cat(
            [batch["formula"], batch["precursor_mz"].unsqueeze(1) / 1000.0,
             batch["collision_energy"].unsqueeze(1) / 100.0], dim=1))
        return self.merge(torch.cat(
            [peaks, f, self.adduct(batch["adduct"]),
             self.instrument(batch["instrument"])], dim=1))
