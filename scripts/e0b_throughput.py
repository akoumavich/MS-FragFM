"""E0-b: FragFM sampling throughput, split by phase.

The proposal budgets ~25x from cutting 500 Euler steps to 20.  That lever only
touches the flow integration; the coarse-to-fine decode (autoencoder plus Blossom
assembly) is a fixed per-sample cost, much of it CPU, that step reduction does
not touch.  A single end-to-end seconds-per-sample number would hide the thing
that decides whether the budget closes: if decode dominates at 20 steps, cutting
steps buys nothing.

One generator is built and reused across the sweep.  LMDB refuses a second open
of the same environment in one process, and rebuilding also re-embeds the whole
fragment pool, which at 54k fragments is minutes per configuration.

Usage:  python scripts/e0b_throughput.py --steps 500 200 100 50 20 10 --bs 256
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from msfragfm.paths import RESULTS

FRAGFM = Path(__file__).resolve().parents[2] / "FragFM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[500, 200, 100, 50, 20, 10])
    ap.add_argument("--bs", type=int, nargs="+", default=[256])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cfg", default="cfgs/generate/npgen.yaml")
    ap.add_argument("--tag", default="npgen")
    args = ap.parse_args()

    sys.path.insert(0, str(FRAGFM))
    os.chdir(FRAGFM)
    from fragfm.mol_generator import FragFMGenerator
    from fragfm.utils.file import read_yaml_as_easydict

    cfg = read_yaml_as_easydict(args.cfg)
    cfg.bs = max(args.bs)
    cfg.n_sample = cfg.bs
    cfg.force_save_dirn = cfg.save_dirn = str(RESULTS / "_e0b_scratch")
    os.makedirs(cfg.save_dirn, exist_ok=True)

    sampler = FragFMGenerator(cfg)
    sampler.set_seed(0)

    # Realistic fragment counts, taken once from the dataset; passing them in
    # explicitly lets us vary batch size without rebuilding the dataloader.
    for batch in sampler.test_loader:
        _, graph = batch
        base_ns = [int((graph.batch == i).sum()) for i in range(graph.batch.max() + 1)]
        break

    rows = []
    for bs in args.bs:
        ns = (base_ns * (bs // len(base_ns) + 1))[:bs]
        for steps in args.steps:
            sampler.cfg.n_euler_step = steps
            flow_s = decode_s = 0.0
            n = 0
            for rep in range(args.repeats + 1):  # first pass is warmup
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                x = sampler.sample_molecule_graph_dynamic(n_frags=ns)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                before = len(sampler.gen_smis)
                sampler.store_smis_from_coarse_graph(*x)
                t2 = time.perf_counter()
                if rep == 0:
                    continue
                flow_s += t1 - t0
                decode_s += t2 - t1
                n += len(sampler.gen_smis) - before

            valid = len(sampler.valid_smis) / max(len(sampler.gen_smis), 1)
            uniq = len(sampler.unique_smis) / max(len(sampler.valid_smis), 1)
            row = {
                "bs": bs, "steps": steps, "n_sampled": n,
                "flow_s_per_sample": flow_s / n,
                "decode_s_per_sample": decode_s / n,
                "total_s_per_sample": (flow_s + decode_s) / n,
                "decode_share": decode_s / (flow_s + decode_s),
                "cumulative_validity": valid, "cumulative_uniqueness": uniq,
                "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            rows.append(row)
            print(
                f"bs={bs:4d} steps={steps:4d}  flow={row['flow_s_per_sample'] * 1e3:8.2f} ms  "
                f"decode={row['decode_s_per_sample'] * 1e3:8.2f} ms  "
                f"total={row['total_s_per_sample']:6.3f} s  "
                f"decode_share={row['decode_share']:.2f}  "
                f"valid={valid:.3f} uniq={uniq:.3f}",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()

    path = RESULTS / f"e0b_throughput_{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {path}")
    print("validity/uniqueness are cumulative over the whole sweep, not per row")


if __name__ == "__main__":
    main()
