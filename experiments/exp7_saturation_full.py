"""Experiment 7 / Figure 7 — Depth saturation of the recoverable coherent information
under the full device noise profile.

This repeats the exp3 saturation study but replaces the single dephasing channel with
the complete per-cycle incoherent layer of a NoiseProfile (amplitude damping T1 +
dephasing T2 + depolarizing r_1q). The leading-order (theta->0) labeled Fisher is
computed exactly, without pattern enumeration (full twirl mu=0 -> cross-terms vanish):

    E_s[F_Q](L) = sum_ell F_ell,
    F_ell = SLD-Fisher of the cycle-ell perturbation -i[Z/2, rho_ell] propagated through
            the remaining ideal drives + incoherent layers to the output, on the noisy
            ideal final state rho_0.

At the noiseless limit this reduces to Oracle O2 (sum sin^2(ell*alpha)); with the full
profile the curve grows then saturates/peaks at a depth L*(T2) that grows with T2.

Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Rx
from core.noise import NoiseProfile, apply_incoherent_layer
from core.pauli import pauli

_Z = pauli("Z")


def _superop(func_on_rho):
    """4x4 superoperator S with vec(func(rho)) = S @ rho.reshape(-1) (row-major vec).

    Built by probing the linear map on the 2x2 matrix basis -- foolproof, no vec
    convention to get wrong. Makes each channel application a single 4x4 matmul.
    """
    S = np.zeros((4, 4), dtype=complex)
    for idx in range(4):
        E = np.zeros((2, 2), dtype=complex)
        E[idx // 2, idx % 2] = 1.0
        S[:, idx] = func_on_rho(E).reshape(-1)
    return S


def noisy_labeled_EFQ_profile(L, alpha, profile, dt=1.0):
    """Leading-order labeled E_s[F_Q] with a full per-cycle NoiseProfile (exact, O(L^2)).

    Optimized with precomputed superoperators for the ideal drive (Rx conjugation) and
    the incoherent layer, so the O(L^2) propagation is small matrix-vector products.
    """
    U = Rx(alpha)
    noiseless = profile.is_noiseless()
    Rsup = _superop(lambda r: U @ r @ U.conj().T)
    Lsup = (np.eye(4) if noiseless
            else _superop(lambda r: apply_incoherent_layer(r, profile, 1, mode="dm")))
    step = Lsup @ Rsup           # one full cycle: drive then incoherent layer

    # Forward pass: noisy ideal evolution (theta=0), recording each pre-error DM (post-drive).
    v = np.array([1, 0, 0, 0], dtype=complex)     # vec(|0><0|)
    pre = []
    for _ in range(L):
        v = Rsup @ v                              # after the ideal drive (pre-error)
        pre.append(v.copy())
        v = Lsup @ v                              # after the incoherent layer
    rho0 = v.reshape(2, 2)

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
        rho_ell = pre[ell].reshape(2, 2)
        P = -1j * (0.5 * _Z @ rho_ell - rho_ell @ (0.5 * _Z))
        pv = Lsup @ P.reshape(-1)                 # cycle-ell incoherent layer
        for _k in range(ell + 1, L):
            pv = step @ pv                        # remaining cycles (drive + layer)
        total += sld_fisher(pv.reshape(2, 2))
    return float(total)


def noiseless_labeled(L, alpha):
    return float(sum(np.sin(ell * alpha) ** 2 for ell in range(1, L + 1)))


def full_profile(x_T2, dt=1.0, ratio_T1=2.0, r_1q=0.0):
    """Device-style profile parametrized by dephasing strength x_T2 = dt/T2.

    T1 = ratio_T1 * T2 (so T2 <= 2 T1 holds for ratio_T1 >= 0.5); optional depolarizing.
    """
    if x_T2 == 0:
        return NoiseProfile(dt=dt, T1=np.inf, T2=np.inf, r_1q=r_1q)
    T2 = dt / x_T2
    return NoiseProfile(dt=dt, T1=ratio_T1 * T2, T2=T2, r_1q=r_1q)


def saturation_depth(alpha, profile, L_max=60):
    vals = [noisy_labeled_EFQ_profile(L, alpha, profile) for L in range(1, L_max + 1)]
    return int(np.argmax(vals)) + 1, float(np.max(vals))


# --------------------------------------------------------------------------
# Figure 7
# --------------------------------------------------------------------------
def make_figure(alpha=1.0, x_T2_list=(0.02, 0.05, 0.10, 0.20), r_1q=0.002, L_max=24):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    Ls = np.arange(1, L_max + 1)
    colors = [C["green"], C["skyblue"], C["orange"], C["vermil"]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.5))

    axA.plot(Ls, [noiseless_labeled(L, alpha) for L in Ls], "-", color=C["black"],
             lw=2.0, label=r"noiseless: $\sim L/2$")
    for x, col in zip(x_T2_list, colors):
        prof = full_profile(x, r_1q=r_1q)
        vals = np.array([noisy_labeled_EFQ_profile(L, alpha, prof) for L in Ls])
        Lstar = int(np.argmax(vals)) + 1
        axA.plot(Ls, vals, "-o", color=col, ms=3.5,
                 label=rf"$\delta t/T_2={x}$ ($T_2\approx{1/x:.0f}$, $T_1=2T_2$, $r_{{1q}}={r_1q}$)")
        axA.plot(Lstar, vals[Lstar - 1], "*", color=col, ms=15, mec="k", mew=0.5, zorder=6)
    axA.set_xlabel(r"circuit depth  $L$  (Trotter cycles)")
    axA.set_ylabel(r"labeled  $\mathbb{E}_s[F_Q]$  (per shot)")
    axA.set_title(r"(a) Saturation under the full profile ($T_1{+}T_2{+}$depol)")
    axA.set_xlim(0, L_max + 0.5)
    axA.set_ylim(0, None)
    axA.legend(loc="upper right", fontsize=8)
    axA.annotate(r"$\star = L^*$", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=10)

    # Panel (b): L*(T2) vs T2 for the full profile
    x_sweep = np.linspace(0.02, 0.4, 20)
    T2s = 1.0 / x_sweep
    Lstars = [saturation_depth(alpha, full_profile(x, r_1q=r_1q), L_max=80)[0]
              for x in x_sweep]
    axB.plot(T2s, Lstars, "-o", color=C["blue"], ms=4)
    axB.set_xlabel(r"coherence time  $T_2$  (cycles)")
    axB.set_ylabel(r"saturation depth  $L^*$")
    axB.set_title(r"(b) Useful depth grows with $T_2$ (full profile)")
    axB.set_xlim(0, max(T2s) * 1.05)
    axB.set_ylim(0, None)

    save(fig, "fig7_saturation_full")
    plt.close(fig)


def verify_noiseless_recovers_O2(verbose=True):
    prof = NoiseProfile(dt=1.0)      # noiseless
    err = max(abs(noisy_labeled_EFQ_profile(L, 1.0, prof) - noiseless_labeled(L, 1.0))
              for L in range(1, 13))
    if verbose:
        print(f"noiseless profile recovers Oracle O2: max|code - O2| = {err:.2e}")
    return err


def main():
    err = verify_noiseless_recovers_O2()
    if err > 1e-9:
        raise SystemExit(f"noiseless sanity FAILED: {err:.2e}")
    print("Saturation depths L*(T2) under full profile (alpha=1, r_1q=0.002):")
    for x in (0.02, 0.05, 0.10, 0.20):
        Lstar, peak = saturation_depth(1.0, full_profile(x, r_1q=0.002))
        print(f"  dt/T2={x}: T2~{1/x:.1f}  L*={Lstar}  peak E[F_Q]={peak:.3f}")
    print("Generating Figure 7 ...")
    make_figure()
    print("Figure 7 done.")


if __name__ == "__main__":
    main()
