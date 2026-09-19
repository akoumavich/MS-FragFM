"""Build the fragment bag (pool) FragFM samples candidates from.

Replaces FragFM's `process/process_fragment_from_lmdb.py`, which hard-codes a
dataset whitelist for the strict/relaxed branch and reads exactly one source.
We need neither restriction: E0-c showed the pool should be harvested from a
large external corpus rather than from MassSpecGym alone (54.4% -> 82.3% test
coverage), so the builder has to pool several sources.

**Leakage boundary.** `random_select_frags_by_occurance` samples with
`p = occurrence / sum` and `replace=False`, so a fragment with zero occurrence in
the selected split is unreachable -- a corpus-harvested pool with
MassSpecGym-only counts would be inert. We therefore write `train_occurance` as
"everything the generator may draw from at training time": the MassSpecGym train
fold plus the external corpus. The corpus is MassSpecGym's own MCES-2-disjoint
release, built to be distant from the test fold, which is what makes including it
safe. MassSpecGym val/test counts are recorded for diagnostics and never folded
into the train bag.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import lmdb
import pandas as pd
from rdkit import RDLogger

from msfragfm.paths import MSG_PRETRAIN_MCES2

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"
_CFG = {}


def _init(decomp):
    _CFG["decomp"] = decomp
    RDLogger.DisableLog("rdApp.*")


def _frags(smi):
    from fragfm.process import process_sample

    try:
        return process_sample(
            {"smi": smi, "data_type": "guacamol", "decomp_method": _CFG["decomp"]}
        )["frag_smi_list"]
    except Exception:  # noqa: BLE001
        return ()


def _proc_frag(smi):
    from fragfm.process import process_frag

    try:
        return smi, process_frag(smi, is_relax=True)
    except Exception:  # noqa: BLE001
        return smi, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg-lmdb", default=None, help="molecule LMDB from build_msg_lmdb.py")
    ap.add_argument("--corpus-n", type=int, default=200_000, help="0 disables the corpus")
    ap.add_argument("--decomp", default="rbrics")
    ap.add_argument("--out", default="msg_rbrics_all")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    processed = FRAGFM / "data" / "processed"
    msg_lmdb = Path(args.msg_lmdb) if args.msg_lmdb else processed / f"{args.out}.lmdb"

    train, valid, test, corpus = Counter(), Counter(), Counter(), Counter()

    env = lmdb.open(str(msg_lmdb), readonly=True, lock=False, map_size=int(1e12))
    with env.begin() as txn:
        for key, val in txn.cursor():
            frags = pickle.loads(val)["frag_smi_list"]
            bucket = (
                train if key.startswith(b"train") else
                valid if key.startswith(b"valid") else test
            )
            bucket.update(frags)
    env.close()
    print(f"MassSpecGym: train {len(train):,} | valid {len(valid):,} | test {len(test):,} fragments")

    if args.corpus_n:
        df = pd.read_csv(MSG_PRETRAIN_MCES2, sep="\t")
        col = "smiles" if "smiles" in df.columns else df.columns[0]
        smis = df.sample(min(args.corpus_n, len(df)), random_state=args.seed)[col].astype(str).tolist()
        t0 = time.perf_counter()
        with mp.Pool(args.procs, initializer=_init, initargs=(args.decomp,)) as pool:
            for frags in pool.imap_unordered(_frags, smis, chunksize=64):
                corpus.update(frags)
        print(f"corpus: {len(corpus):,} fragments from {len(smis):,} molecules "
              f"in {time.perf_counter() - t0:.0f}s")

    # What the generator may draw from at training time.
    drawable = train + corpus
    total = drawable + valid + test
    print(f"pool: {len(drawable):,} drawable, {len(total):,} total")

    ranked = [s for s, _ in total.most_common()]
    with mp.Pool(args.procs, initializer=_init, initargs=(args.decomp,)) as pool:
        results = dict(pool.map(_proc_frag, ranked, chunksize=64))
    dropped = [s for s in ranked if results[s] is None]
    ranked = [s for s in ranked if results[s] is not None]
    print(f"{len(ranked):,} fragments processed, {len(dropped):,} dropped")

    frag_data, smi_to_idx = [], {}
    for i, smi in enumerate(ranked):
        rec = {
            "key": f"frag_{i}",
            "smi": smi,
            "occurance": total[smi],
            "train_occurance": drawable[smi],
            "valid_occurance": valid[smi],
            "test_occurance": test[smi],
        }
        rec.update(results[smi])
        frag_data.append(rec)
        smi_to_idx[smi] = i

    (processed / f"{args.out}_fragment_to_idx.pkl").write_bytes(pickle.dumps(smi_to_idx))
    env = lmdb.open(str(processed / f"{args.out}_fragment.lmdb"), map_size=int(1e12))
    with env.begin(write=True) as txn:
        for rec in frag_data:
            txn.put(rec["key"].encode(), pickle.dumps(rec))
    env.close()

    n_drawable = sum(1 for s in ranked if drawable[s] > 0)
    print(f"\nwrote {processed / (args.out + '_fragment.lmdb')}")
    print(f"{n_drawable:,} fragments have nonzero train occurrence (bag draws need >= n_base_frag)")


if __name__ == "__main__":
    main()
