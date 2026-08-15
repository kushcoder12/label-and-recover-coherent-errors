"""Experiment 8 / Figure 8 — Recovered coherent-error MAP.

A multi-gate 5-qubit circuit carries a VECTOR of K coherent errors of BOTH types --
single-qubit over-rotations and two-qubit ZZ crosstalk -- each twirled with its own
per-shot sign label. Keeping the labels, we recover every coherent angle individually
and unbiased under the full device noise profile (core.error_map), reading each
visibility self-consistently from the marginal (never from the truth).

Figure 8:
  (a) recovered vs true angle (1q and 2q as two series), y=x line, shot-noise error bars;
  (b) a coherent-error map of the chip: per-qubit 1q magnitude (nodes) and per-edge 2q
      magnitude (links) on the device layout.

Verification: noiseless -> exact recovery (N5 smoke); noisy -> |bias| < a few x shot-noise.
Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Ry, Rz, Rx
from core.two_qubit_gates import CX_gate
from core.error_map import CoherentError, recover_error_vector, _embed_1q
from core.noise import NoiseProfile

N = 5
_CX = CX_gate()


def _cx(control, target, n):
    """Embed CX(control,target) (1-indexed) as a dense 2^n unitary via basis action."""
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)
    cbit = n - control
    tbit = n - target
    for r in range(dim):
        c = (r >> cbit) & 1
        out = r ^ (1 << tbit) if c == 1 else r
        U[out, r] = 1.0
    return U


def build_circuit(seed=1):
    """A hardware-efficient-style 5-qubit circuit (dense ideal ops) + injected errors."""
    rng = np.random.default_rng(seed)
    ops = []
    # layer of Ry rotations
    for q in range(1, N + 1):
        ops.append(_embed_1q(Ry(0.5 + 0.3 * q), q, N))
    # entangling chain of CX
    for q in range(1, N):
        ops.append(_cx(q, q + 1, N))
    # layer of Rz rotations
    for q in range(1, N + 1):
        ops.append(_embed_1q(Rz(0.4 + 0.2 * q), q, N))
    # second entangling chain
    for q in range(1, N):
        ops.append(_cx(q, q + 1, N))
    L = len(ops)

    # ground-truth coherent errors (avoid silent kernel; angles 0.02-0.3 as specified)
    errors = [
        CoherentError(pos=N, qubits=(1,), gen="Z", theta=0.24),          # after Ry layer
        CoherentError(pos=N, qubits=(3,), gen="Z", theta=0.06),
        CoherentError(pos=N, qubits=(5,), gen="X", theta=0.15),
        CoherentError(pos=2 * N - 1, qubits=(1, 2), gen="ZZ", theta=0.08),  # after 1st CX chain
        CoherentError(pos=2 * N - 1, qubits=(2, 3), gen="ZZ", theta=0.03),
        CoherentError(pos=2 * N - 1, qubits=(3, 4), gen="ZZ", theta=0.05),
        CoherentError(pos=L, qubits=(4,), gen="Z", theta=0.19),          # at the end
        CoherentError(pos=L, qubits=(4, 5), gen="ZZ", theta=0.09),
    ]
    return ops, errors


def run(profile, shots=4000, seed=1):
    ops, errors = build_circuit(seed)
    return recover_error_vector(N, ops, errors, profile, shots=shots), errors


def device_profile():
    """Device-representative single-qubit-tier profile (per twirled cycle)."""
    return NoiseProfile(dt=1.0, T1=60.0, T2=40.0, r_1q=0.004, r_2q=0.0,
                        e01=0.0, e10=0.0)


# --------------------------------------------------------------------------
# Figure 8
# --------------------------------------------------------------------------
def make_figure():
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    res, _ = run(device_profile())
    res = [r for r in res if not r.silent]
    one = [r for r in res if r.kind == "1q"]
    two = [r for r in res if r.kind == "2q"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.7))

    # ---- (a) recovered vs true ----
    lim = 0.30
    axA.plot([0, lim], [0, lim], "-", color="0.6", lw=1.0, zorder=0, label="ideal $y=x$")
    for grp, col, mk, lab in [(one, C["blue"], "o", "1q over-rotation"),
                              (two, C["vermil"], "s", "2q $ZZ$ crosstalk")]:
        xt = [r.theta_true for r in grp]
        yh = [r.theta_hat for r in grp]
        ye = [max(r.sigma, 1e-4) for r in grp]
        axA.errorbar(xt, yh, yerr=ye, fmt=mk, color=col, ms=6, mec="k", mew=0.5,
                     capsize=3, lw=1.0, label=lab)
    axA.set_xlabel(r"true coherent angle  $\theta^*_a$  (rad)")
    axA.set_ylabel(r"recovered  $\hat\theta_a$  (rad)")
    axA.set_title(r"(a) Full-vector recovery (both error types)")
    axA.set_xlim(0, lim)
    axA.set_ylim(0, lim)
    axA.legend(loc="upper left", fontsize=9)

    # ---- (b) chip coherent-error map ----
    xs = np.arange(1, N + 1)
    ys = np.zeros(N)
    node_mag = np.zeros(N)
    for r in one:
        node_mag[r.qubits[0] - 1] += abs(r.theta_hat)
    edge_mag = {}
    for r in two:
        edge_mag[tuple(r.qubits)] = abs(r.theta_hat)

    # edges of the linear chain
    segs, widths, evals = [], [], []
    for q in range(1, N):
        segs.append([(q, 0), (q + 1, 0)])
        m = edge_mag.get((q, q + 1), 0.0)
        evals.append(m)
    lc = LineCollection(segs, cmap="magma", array=np.array(evals),
                        linewidths=8, zorder=1)
    lc.set_clim(0, max(max(evals) if evals else 1e-6, 1e-6))
    axB.add_collection(lc)
    sc = axB.scatter(xs, ys, c=node_mag, cmap="viridis", s=650, edgecolors="k",
                     linewidths=1.0, zorder=2, vmin=0)
    for q in range(N):
        axB.annotate(f"q{q+1}", (xs[q], 0), ha="center", va="center", fontsize=9,
                     color="white", zorder=3)
    cb1 = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
    cb1.set_label(r"1q $|\hat\theta|$ (node)", fontsize=9)
    cb2 = fig.colorbar(lc, ax=axB, fraction=0.046, pad=0.12)
    cb2.set_label(r"2q $ZZ\ |\hat\theta|$ (edge)", fontsize=9)
    axB.set_xlim(0.5, N + 0.5)
    axB.set_ylim(-1, 1)
    axB.set_yticks([])
    axB.set_xticks(xs)
    axB.set_xlabel("qubit (linear chain)")
    axB.set_title(r"(b) Recovered coherent-error map of the chip")

    save(fig, "fig8_error_map")
    plt.close(fig)


def report():
    print("Recovered coherent-error map (device profile):")
    res, _ = run(device_profile())
    print(f"  {'error':12s} {'true':>7s} {'hat':>9s} {'bias':>9s} {'sigma':>8s} {'v':>6s}")
    biases, sigmas = [], []
    for r in res:
        if r.silent:
            print(f"  {r.label:12s}  SILENT (kernel, g={r.g:+.2f})")
            continue
        b = r.theta_hat - r.theta_true
        biases.append(b); sigmas.append(r.sigma)
        print(f"  {r.label:12s} {r.theta_true:>7.3f} {r.theta_hat:>9.4f} {b:>+9.5f} "
              f"{r.sigma:>8.4f} {r.visibility:>6.3f}")
    biases = np.array(biases); sigmas = np.array(sigmas)
    print(f"  max|bias| = {np.max(np.abs(biases)):.2e}   RMSE = {np.sqrt(np.mean(biases**2)):.2e}"
          f"   median shot-sigma = {np.median(sigmas):.2e}")
    return biases, sigmas


def main():
    # N5 smoke test: noiseless -> exact recovery
    res0, _ = run(NoiseProfile(dt=1.0, T1=np.inf, T2=np.inf), shots=None)
    err0 = max(abs(r.theta_hat - r.theta_true) for r in res0 if not r.silent)
    print(f"N5 smoke (noiseless recovery): max|hat - true| = {err0:.2e}")
    if err0 > 1e-5:
        raise SystemExit(f"noiseless recovery FAILED: {err0:.2e}")

    biases, sigmas = report()
    # unbiasedness: |bias| within a few x shot-noise for all K
    if np.max(np.abs(biases)) > 3 * np.max(sigmas) + 1e-4:
        raise SystemExit("recovery bias exceeds shot-noise tolerance")
    print("Unbiased full-vector recovery confirmed.\nGenerating Figure 8 ...")
    make_figure()
    print("Figure 8 done.")


if __name__ == "__main__":
    main()
