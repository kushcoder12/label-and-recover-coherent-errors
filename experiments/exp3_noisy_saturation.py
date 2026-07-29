"""Experiment 3 / Figure 3 — Noisy saturation of the recoverable coherent information.

Single-qubit L-cycle model (as in exp2) but with pure dephasing (rate gamma_z,
T2-type) applied after every cycle. We compute the leading-order (theta->0)
labeled Fisher information E_s[F_Q] exactly, without pattern enumeration, using
the fact that at full twirl (mu=0) the cross-terms vanish:

    E_s[F_Q] = sum_ell F_ell,

where F_ell is the SLD (mixed-state) Fisher information of the cycle-ell error
perturbation  -i[Z/2, rho_ell]  propagated through the remaining noisy channel to
the output, evaluated on the noisy ideal final state rho_0 = sum_j lam_j |j><j|:

    F_ell = 2 sum_{jk: lam_j+lam_k>0} |<j| D_ell |k>|^2 / (lam_j + lam_k),
    D_ell = R_ell[ -i[Z/2, rho_ell] ].

At gamma_z=0 this reduces exactly to Oracle O2 (sum_ell sin^2(ell*alpha)); with
dephasing the curve grows linearly then saturates and peaks at a depth L*(T2),
beyond which decoherence erases coherent information faster than it accumulates.

No external oracle (new physics); the gamma_z=0 limit is the sanity anchor.
Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Rx
from core.channels import dephasing_kraus, apply_kraus_dm
from core.pauli import pauli

_Z = pauli("Z")


def noisy_labeled_EFQ(L, alpha, gamma_z, dt=1.0):
    """Leading-order labeled E_s[F_Q] with per-cycle dephasing (exact, O(L^2))."""
    K = dephasing_kraus(gamma_z, dt) if gamma_z > 0 else None
    U = Rx(alpha)

    # Forward pass: noisy ideal evolution (theta=0), recording each pre-error DM.
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    pre = []
    for _ in range(L):
        rho = U @ rho @ U.conj().T
        pre.append(rho.copy())              # state just before the (identity) error
        if K is not None:
            rho = apply_kraus_dm(rho, K)
    rho0 = rho

    lam, V = np.linalg.eigh(rho0)
    lam = np.clip(lam.real, 0.0, None)

    def sld_fisher(D):
        Dm = V.conj().T @ D @ V
        F = 0.0
        for j in range(2):
            for k in range(2):
                s = lam[j] + lam[k]
                if s > 1e-12:
                    F += 2.0 * abs(Dm[j, k]) ** 2 / s
        return F

    total = 0.0
    for ell in range(L):
        # error perturbation at cycle ell: -i[Z/2, rho_ell]
        P = -1j * (0.5 * _Z @ pre[ell] - pre[ell] @ (0.5 * _Z))
        if K is not None:                   # cycle-ell noise, then remaining cycles
            P = apply_kraus_dm(P, K)
        for _k in range(ell + 1, L):
            P = U @ P @ U.conj().T
            if K is not None:
                P = apply_kraus_dm(P, K)
        total += sld_fisher(P)
    return float(total)


def noiseless_labeled(L, alpha):
    """gamma_z = 0 reference = Oracle O2 labeled = sum_ell sin^2(ell*alpha)."""
    return float(sum(np.sin(ell * alpha) ** 2 for ell in range(1, L + 1)))


def T2_cycles(gamma_z, dt=1.0):
    """Coherence time in cycles: coherence ~ exp(-2 gamma_z dt * L) -> T2 = 1/(2 gamma_z dt)."""
    return np.inf if gamma_z == 0 else 1.0 / (2.0 * gamma_z * dt)


def saturation_depth(alpha, gamma_z, L_max=60, dt=1.0):
    """L*(T2) = argmax_L E_s[F_Q](L): the depth of maximum recoverable information."""
    vals = [noisy_labeled_EFQ(L, alpha, gamma_z, dt) for L in range(1, L_max + 1)]
    return int(np.argmax(vals)) + 1, float(np.max(vals))


# --------------------------------------------------------------------------
# Figure 3
# --------------------------------------------------------------------------
def make_figure(alpha=1.0, gammas=(0.02, 0.05, 0.10, 0.20), L_max=24):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    Ls = np.arange(1, L_max + 1)
    colors = [C["green"], C["skyblue"], C["orange"], C["vermil"]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    # Panel A: labeled E[F_Q] vs L for several gamma_z + noiseless reference
    axA.plot(Ls, [noiseless_labeled(L, alpha) for L in Ls], "-", color=C["black"],
             lw=2.0, label=r"$\gamma_z=0$ (no dephasing): $\sim L/2$")
    for g, col in zip(gammas, colors):
        vals = np.array([noisy_labeled_EFQ(L, alpha, g) for L in Ls])
        Lstar = int(np.argmax(vals)) + 1
        axA.plot(Ls, vals, "-o", color=col, ms=3.5,
                 label=rf"$\gamma_z={g}$  ($T_2\approx{T2_cycles(g):.0f}$ cyc)")
        axA.plot(Lstar, vals[Lstar - 1], "*", color=col, ms=15, mec="k", mew=0.5,
                 zorder=6)
    axA.set_xlabel(r"circuit depth  $L$  (Trotter cycles)")
    axA.set_ylabel(r"labeled  $\mathbb{E}_s[F_Q]$  about $\theta$  (per shot)")
    axA.set_title(r"(a) Dephasing saturates the recoverable information")
    axA.set_xlim(0, L_max + 0.5)
    axA.set_ylim(0, None)
    axA.legend(loc="upper right", fontsize=9)
    axA.annotate(r"$\star = L^*(T_2)$ (saturation depth)", xy=(0.40, 0.46),
                 xycoords="axes fraction", fontsize=10)

    # Panel B: saturation depth L*(T2) vs T2 (the DoD saturation-depth curve)
    g_sweep = np.linspace(0.02, 0.4, 20)
    T2s, Lstars = [], []
    for g in g_sweep:
        Ls_, _ = saturation_depth(alpha, g, L_max=80)
        T2s.append(T2_cycles(g))
        Lstars.append(Ls_)
    axB.plot(T2s, Lstars, "-o", color=C["blue"], ms=4)
    axB.set_xlabel(r"coherence time  $T_2 = 1/(2\gamma_z)$  (cycles)")
    axB.set_ylabel(r"saturation depth  $L^*$")
    axB.set_title(r"(b) Useful depth grows with $T_2$")
    axB.set_xlim(0, max(T2s) * 1.05)
    axB.set_ylim(0, None)

    save(fig, "fig3_noisy_saturation")
    plt.close(fig)


def verify_gamma0_recovers_O2(verbose=True):
    max_err = max(abs(noisy_labeled_EFQ(L, 1.0, 0.0) - noiseless_labeled(L, 1.0))
                  for L in range(1, 13))
    if verbose:
        print(f"gamma_z=0 recovers Oracle O2:  max|code - O2| = {max_err:.2e}")
    return max_err


def main():
    err = verify_gamma0_recovers_O2()
    if err > 1e-9:
        raise SystemExit(f"gamma_z=0 sanity FAILED: {err:.2e} > 1e-9")
    print("Saturation depths L*(T2):")
    for g in (0.02, 0.05, 0.10, 0.20):
        Lstar, peak = saturation_depth(1.0, g)
        print(f"  gamma_z={g}: T2~{T2_cycles(g):.1f} cyc  L*={Lstar}  peak E[F_Q]={peak:.3f}")
    print("Generating Figure 3 ...")
    make_figure()
    print("Figure 3 done.")


if __name__ == "__main__":
    main()
