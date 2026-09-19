"""Verify that the FragFM generator stack and the ms-pred oracle stack coexist.

Run once after scripts/setup_env.sh.  Everything downstream assumes this passes.
"""

import sys
import time
import traceback

RESULTS = []


def check(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            RESULTS.append((True, name, detail, time.perf_counter() - t0))
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((False, name, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0))
            traceback.print_exc()
        return fn

    return deco


@check("core versions")
def _():
    import numpy, rdkit, torch

    return (
        f"py={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda} numpy={numpy.__version__} rdkit={rdkit.__version__}"
    )


@check("gpu + bf16 matmul")
def _():
    import torch

    assert torch.cuda.is_available(), "no CUDA device"
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t0) / 1e12
    return f"{name} {mem:.0f}GiB bf16={tflops:.0f} TFLOP/s"


@check("torch.compile")
def _():
    import torch

    f = torch.compile(lambda x: (x * 2).relu().sum())
    return f"ok, out={float(f(torch.randn(64, 64, device='cuda'))):.1f}"


@check("pyg + torch_scatter + dgl")
def _():
    import dgl, torch_geometric, torch_scatter, torch_sparse  # noqa: F401

    return f"pyg={torch_geometric.__version__} dgl={dgl.__version__}"


@check("fragfm imports under numpy2/rdkit2025")
def _():
    import fragfm.process as P  # noqa: F401
    from fragfm.model import flow  # noqa: F401

    return "fragfm.process + fragfm.model.flow import clean"


@check("fragfm BRICS decomposition on natural products")
def _():
    from fragfm.process import process_sample

    # Representative MassSpecGym-style chemistry: flavonoid, alkaloid, macrolide-ish,
    # peptide-like, and a halogenated drug.
    smis = [
        "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",  # quercetin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC1=C(C(=O)Nc2ccccc2)S(=O)(=O)c2ccccc2N1C",  # sultam
        "NC(Cc1ccc(O)cc1)C(=O)NC(CC(=O)O)C(=O)O",  # dipeptide
        "Clc1ccc(C2(Cn3cncn3)OC(COc3ccc(N4CCN(c5ccc(N6CCCC6=O)cc5)CC4)cc3)CO2)c(Cl)c1",  # posaconazole-like
    ]
    out = []
    for smi in smis:
        s = process_sample({"smi": smi, "data_type": "npgen", "decomp_method": "brics"})
        out.append(f"{s['n_frag']}f/{len(s['h'])}a")
    return "n_frag/n_atom(with H): " + " ".join(out)


@check("ms-pred oracle imports")
def _():
    # glacier/ and iceberg/ ship without __init__.py, so import the modules directly.
    from ms_pred.glacier import joint_model as glacier_jm  # noqa: F401
    from ms_pred.iceberg import joint_model as iceberg_jm  # noqa: F401

    return "glacier.joint_model + iceberg.joint_model import clean"


@check("msbuddy / formula utils")
def _():
    from ms_pred.common import chem_utils

    return f"chem_utils ok ({len(dir(chem_utils))} symbols)"


if __name__ == "__main__":
    print("\n" + "=" * 78)
    print(f"{'':<2} {'CHECK':<46} {'TIME':>7}  DETAIL")
    print("-" * 78)
    for ok, name, detail, dt in RESULTS:
        print(f"{'PASS' if ok else 'FAIL':<5}{name:<44} {dt:6.2f}s  {detail}")
    print("=" * 78)
    n_fail = sum(not ok for ok, *_ in RESULTS)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)
