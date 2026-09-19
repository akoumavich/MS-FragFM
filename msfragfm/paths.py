"""Filesystem layout.  Override the root with MSFRAGFM_DATA."""

import os
from pathlib import Path

DATA = Path(os.environ.get("MSFRAGFM_DATA", Path.home() / "data" / "msfragfm"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = Path(__file__).resolve().parents[1] / "results"

MSG_TSV = RAW / "MassSpecGym1.5.tsv"
MSG_PRETRAIN_MCES2 = RAW / "MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv"

for _p in (RAW, PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
