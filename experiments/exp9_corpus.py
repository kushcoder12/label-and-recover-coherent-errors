"""Experiment 9 / Figure 9 — Corpus generality (GAPS_CODE.md GAP 4).

The labeled-RC coherent-error recovery of GAP 3 is run across a curated corpus of
standard circuit families (loaded from the bundled QASMBench set via the native
core.qasm_loader, plus one native random-Clifford circuit). Each circuit gets the
SAME fixed injected coherent-error profile (a few 1q over-rotations + 2q ZZ crosstalk),
run under the SAME fixed device NoiseProfile; we report the per-circuit recovery bias
and the shot-noise floor. Message: recovery is unbiased and uniform across families,
far below the shot-noise floor -- not a Trotter-only result.

Writes figures/fig9_corpus.{pdf,png} and figures/corpus_manifest.json.
Each circuit passes the noiseless-limit smoke test before its noisy run.
Model functions are import-safe; tests import them.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.qasm_loader import load_circuit
from core.error_map import CoherentError, recover_error_vector
from core.single_qubit_gates import Hadamard, P_gate
from core.two_qubit_gates import CX_gate
from core.noise import NoiseProfile, _embed_two_qubit_op
from core.qasm_loader import _embed_1q

_QDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm",
                     "qasmbench-small")

# Curated corpus: (display name, family, qasm subdir) -- n <= 6 for exact noisy DM.
CORPUS = [
    ("QFT", "qft_n4"),
    ("Adder", "adder_n4"),
    ("QAOA", "qaoa_n6"),
    ("VQE-UCCSD", "vqe_uccsd_n6"),
    ("GHZ/cat", "cat_state_n4"),
    ("Trotter", "basis_trotter_n4"),
    ("Simon", "simon_n6"),
    ("Grover", "grover_n2"),
    ("Teleport", "teleportation_n3"),
    ("HW-eff", "variational_n4"),
    ("Deutsch", "deutsch_n2"),
]


def _native_random_clifford(n=4, depth=20, seed=7):
    """A native random-Clifford circuit (H, S, CX) as a list of dense ops + gatelist."""
    rng = np.random.default_rng(seed)
    ops, gl = [], []
    H, S = Hadamard(), P_gate(np.pi / 2)
    for _ in range(depth):
        if rng.random() < 0.5:
            q = int(rng.integers(1, n + 1))
            g = "h" if rng.random() < 0.5 else "s"
            ops.append(_embed_1q(H if g == "h" else S, q, n))
            gl.append((g, [], (q,)))
        else:
            a = int(rng.integers(1, n + 1))
            b = int(rng.integers(1, n + 1))
            while b == a:
                b = int(rng.integers(1, n + 1))
            ops.append(_embed_two_qubit_op(CX_gate(), a, b, n))
            gl.append(("cx", [], (a, b)))
    return n, ops, gl


def device_profile():
    return NoiseProfile(dt=1.0, T1=80.0, T2=60.0, r_1q=0.004, r_2q=0.02,
                        e01=0.0, e10=0.0)


def _inject(n, n_ops):
    """Fixed known coherent-error profile at mid-circuit (pos), adapted to n."""
    pos = max(1, n_ops // 2)
    errs = [CoherentError(pos=pos, qubits=(1,), gen="Z", theta=0.12),
            CoherentError(pos=pos, qubits=(2,), gen="X", theta=0.07)]
    if n >= 3:
        errs.append(CoherentError(pos=pos, qubits=(1, 2), gen="ZZ", theta=0.09))
    if n >= 4:
        errs.append(CoherentError(pos=pos, qubits=(2, 3), gen="ZZ", theta=0.05))
        errs.append(CoherentError(pos=pos, qubits=(3,), gen="Z", theta=0.10))
    return errs


def load_corpus():
    """Return list of dicts: name, n, depth, twoq, ideal_ops."""
    items = []
    for disp, sub in CORPUS:
        path = os.path.join(_QDIR, sub, sub + ".qasm")
        n, ops, gl = load_circuit(path)
        twoq = sum(1 for name, _, qs in gl if len(qs) >= 2)
        items.append(dict(name=disp, src=sub, n=n, depth=len(ops), twoq=twoq, ops=ops))
    # native random Clifford
    n, ops, gl = _native_random_clifford()
    items.append(dict(name="RandClifford", src="native", n=n, depth=len(ops),
                      twoq=sum(1 for g in gl if len(g[2]) >= 2), ops=ops))
    return items


def run(shots=8000):
    prof = device_profile()
    clean = NoiseProfile(dt=1.0, T1=np.inf, T2=np.inf)
    results = []
    for it in load_corpus():
        errs = _inject(it["n"], it["depth"])
        # noiseless smoke test
        res0 = recover_error_vector(it["n"], it["ops"], errs, clean)
        smoke = max((abs(r.theta_hat - r.theta_true) for r in res0 if not r.silent),
                    default=0.0)
        # noisy recovery
        res = recover_error_vector(it["n"], it["ops"], errs, prof, shots=shots)
        active = [r for r in res if not r.silent]
        biases = np.array([r.theta_hat - r.theta_true for r in active])
        sigmas = np.array([r.sigma for r in active])
        results.append(dict(it, smoke=smoke, n_active=len(active),
                            max_bias=float(np.max(np.abs(biases))) if len(biases) else 0.0,
                            rmse=float(np.sqrt(np.mean(biases**2))) if len(biases) else 0.0,
                            floor=float(np.median(sigmas)) if len(sigmas) else np.nan))
    return results


# --------------------------------------------------------------------------
# Figure 9
# --------------------------------------------------------------------------
def make_figure(results=None):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    if results is None:
        results = run()
    names = [r["name"] for r in results]
    max_bias = np.array([r["max_bias"] for r in results])
    floors = np.array([r["floor"] for r in results])
    x = np.arange(len(results))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    # shot-noise floor band (median +/- spread across circuits)
    fmed = np.nanmedian(floors)
    ax.axhspan(np.nanmin(floors), np.nanmax(floors), color=C["grey"], alpha=0.18,
               label="shot-noise floor (range across corpus)")
    ax.axhline(fmed, color=C["grey"], lw=1.2, ls="--", label=f"median floor ({fmed:.1e})")
    ax.bar(x, np.maximum(max_bias, 1e-8), color=C["blue"], edgecolor="k", linewidth=0.5,
           width=0.6, label=r"recovery $\max_a|\hat\theta_a - \theta^*_a|$ (systematic)")
    ax.set_yscale("log")
    ax.set_ylabel(r"error in recovered coherent angle (rad)")
    ax.set_xlabel("circuit family")
    ax.set_title(r"Uniform, unbiased coherent-error recovery across circuit families")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(1e-9, max(fmed * 5, np.nanmax(floors) * 2))
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig9_corpus")
    plt.close(fig)


def write_manifest(results, path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "figures", "corpus_manifest.json")
    manifest = [dict(name=r["name"], source=r["src"], n=r["n"], depth=r["depth"],
                     two_qubit_gates=r["twoq"], n_active_errors=r["n_active"],
                     max_bias=r["max_bias"], shot_floor=r["floor"]) for r in results]
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote {path}")


def main():
    print("Corpus recovery (device profile):")
    results = run()
    print(f"  {'family':14s} {'n':>2s} {'depth':>6s} {'2q':>4s} {'active':>6s} "
          f"{'smoke':>9s} {'max|bias|':>10s} {'floor':>9s}")
    worst_smoke = 0.0
    for r in results:
        worst_smoke = max(worst_smoke, r["smoke"])
        print(f"  {r['name']:14s} {r['n']:>2d} {r['depth']:>6d} {r['twoq']:>4d} "
              f"{r['n_active']:>6d} {r['smoke']:>9.1e} {r['max_bias']:>10.1e} "
              f"{r['floor']:>9.1e}")
    # verification: every circuit passes the noiseless smoke test, and noisy bias << floor
    if worst_smoke > 1e-5:
        raise SystemExit(f"a circuit failed the noiseless smoke test: {worst_smoke:.2e}")
    for r in results:
        if r["max_bias"] > 0.2 * r["floor"] + 1e-4:
            raise SystemExit(f"{r['name']}: bias {r['max_bias']:.2e} not below shot floor")
    print(f"All circuits: noiseless-exact (<= {worst_smoke:.1e}) and noisy bias below floor.")
    print("Generating Figure 9 ...")
    make_figure(results)
    write_manifest(results)
    print("Figure 9 done.")


if __name__ == "__main__":
    main()
