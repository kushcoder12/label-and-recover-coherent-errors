"""Experiment 4 / Figure 4 — Two-qubit TFIM: conservation + silent kernel.

TFIM Trotter model (CLAUDE.md §2.5): H = Z1 Z2 + g (X1 + X2), cycle
  U0 = exp(-i tau Z1 Z2) exp(-i tau g X1) exp(-i tau g X2) = Rzz(2 tau) Rx1(2 tau g) Rx2(2 tau g),
with a twirled coherent over-rotation exp(-i s_ell delta Z1 Z2 / 2) each cycle.

(a) The Fisher chain rule / conservation law F_joint = F_marg + E_b[F_{s|b}] holds
    to machine precision (~1e-15) at every depth L, now with entangled states and
    Pauli-string generators.
(b) Propagated generators G-tilde_ell = U0^(L-ell) ZZ (U0^(L-ell))^dagger are Pauli
    strings squaring to I; a cycle is SILENT (in the kernel) when <G-tilde_ell>_chi = +/-1,
    contributing 1 - <G-tilde_ell>^2 = 0 (Oracle O5: XX, ZZ, YY are Bell stabilizers).

Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import trotter_tfim
from core.pauli import pauli_string, expectation
from core.fisher import qfi_variance, decompose_joint_fisher
from core.entropy_check import entropy
from core.twirl import all_sign_patterns, pattern_probability


# --------------------------------------------------------------------------
# Core TFIM quantities
# --------------------------------------------------------------------------
def tfim_generators(L, tau, g):
    _, chi, gens = trotter_tfim(L, tau, g, 0.0, [1] * L)
    return gens, chi


def percycle_contrib(L, tau, g):
    """Per-cycle Fisher contribution c_ell = 1 - <G-tilde_ell>_chi^2 (S5 generalized)."""
    gens, chi = tfim_generators(L, tau, g)
    return np.array([1.0 - expectation(chi, G) ** 2 for G in gens])


def labeled_EFQ_tfim(L, tau, g):
    """Labeled E_s[F_Q] = sum_ell c_ell (full twirl)."""
    return float(np.sum(percycle_contrib(L, tau, g)))


def labeled_EFQ_enumerate(L, tau, g, p=0.5):
    """Cross-check by exact 2^L enumeration of E_s[4 Var_chi(1/2 sum s_ell G_ell)]."""
    gens, chi = tfim_generators(L, tau, g)
    tot = 0.0
    for pat in all_sign_patterns(L):
        Gcal = 0.5 * sum(pat[i] * gens[i] for i in range(L))
        tot += pattern_probability(pat, p) * qfi_variance(chi, Gcal)
    return float(tot)


def conservation_split(L, tau, g, delta, p=0.5):
    """(F_joint, F_marg, E_b, residual) for the twirled TFIM, computational-basis readout.

    Joint P[b, pattern] over 4 computational outcomes b and 2^L sign patterns.
    The chain-rule residual |F_joint - (F_marg + E_b)| is a machine-precision check
    of the conservation bookkeeping in the entangled 2-qubit setting.
    """
    pats = all_sign_patterns(L)
    weights = np.array([pattern_probability(pat, p) for pat in pats])

    def joint(d):
        P = np.zeros((4, len(pats)))
        for si, pat in enumerate(pats):
            psi, _, _ = trotter_tfim(L, tau, g, d, pat)
            P[:, si] = weights[si] * np.abs(psi) ** 2
        return P

    Fj, Fm, Eb = decompose_joint_fisher(joint, delta)
    return Fj, Fm, Eb, abs(Fj - (Fm + Eb))


def bell_kernel_demo():
    """Oracle O5: contributions c = 1-<G>^2 of all nine two-qubit Pauli-pair error
    directions {X,Y,Z}x{X,Y,Z} on the Bell state |Phi+>. The three stabilizers
    XX, YY, ZZ are EXACTLY silent (c=0, the kernel); the other six are fully active."""
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1 / np.sqrt(2)
    out = {}
    for a in "XYZ":
        for b in "XYZ":
            ev = expectation(phi, pauli_string(a + b))
            out[a + b] = (ev, 1.0 - ev**2)
    return out


def decoupling_sanity(tau):
    """g->0 decoupling: chi is a product state (entanglement entropy 0) and every
    ZZ error is silent (c_ell = 0). Returns (max_entanglement, max_contrib)."""
    L = 6
    gens, chi = tfim_generators(L, tau, 0.0)
    S = entropy(chi, 2, 1)                        # bipartite entanglement of |chi>
    c = percycle_contrib(L, tau, 0.0)
    return float(abs(S)), float(np.max(c))


# --------------------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------------------
def make_figure(tau=0.3, g=0.8, L_max=7, delta=0.15):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.8, 4.5))

    # ---- Panel (a): conservation vs L (log) ----
    Ls = np.arange(1, L_max + 1)
    Fj, Fm, Eb, res = [], [], [], []
    for L in Ls:
        a, b, c, r = conservation_split(L, tau, g, delta)
        Fj.append(a); Fm.append(b); Eb.append(c); res.append(max(r, 1e-17))
    axA.semilogy(Ls, Fj, "-o", color=C["black"], ms=4,
                 label=r"$F_{\mathrm{joint}}$ (conserved total)")
    axA.semilogy(Ls, Eb, "-s", color=C["vermil"], ms=4,
                 label=r"$\mathbb{E}_b[F_{s|b}]$ (recovered from labels)")
    axA.semilogy(Ls, np.maximum(Fm, 1e-6), "-^", color=C["blue"], ms=4,
                 label=r"$F_{\mathrm{marg}}$ (standard RC)")
    axA.semilogy(Ls, res, "-D", color=C["green"], ms=4,
                 label=r"residual $|F_{\mathrm{joint}}-(F_{\mathrm{marg}}+\mathbb{E}_b)|$")
    axA.set_xlabel(r"circuit depth  $L$  (Trotter cycles)")
    axA.set_ylabel(r"Fisher information about $\delta$")
    axA.set_title(r"(a) Conservation holds to machine precision (2 qubits)")
    axA.set_ylim(1e-17, None)
    axA.legend(loc="center right", fontsize=8.5)

    # ---- Panel (b): silent kernel — 9 Pauli-pair error directions on |Phi+> (O5) ----
    demo = bell_kernel_demo()
    names = list(demo.keys())
    contribs = np.array([demo[k][1] for k in names])
    silent = contribs < 1e-9
    x = np.arange(len(names))
    colors = [C["grey"] if s else C["vermil"] for s in silent]
    bars = axB.bar(x, contribs, color=colors, edgecolor="k", linewidth=0.5)
    for xi, s in zip(x, silent):
        if s:
            axB.annotate("kernel\n(silent)", xy=(xi, 0.0), xytext=(xi, 0.30),
                         ha="center", fontsize=8.5, color=C["black"],
                         arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
    axB.set_xlabel(r"error direction  $G = P_1 P_2$")
    axB.set_ylabel(r"contribution  $c = 1-\langle G\rangle_{\Phi^+}^2$")
    axB.set_title(r"(b) Silent kernel on Bell state $|\Phi^+\rangle$  (Oracle O5)")
    axB.set_xticks(x)
    axB.set_xticklabels(names, fontsize=10)
    axB.set_ylim(0, 1.25)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(facecolor=C["vermil"], edgecolor="k", label="active"),
                        Patch(facecolor=C["grey"], edgecolor="k",
                              label="silent (stabilizer): XX, YY, ZZ")],
               loc="upper center", fontsize=9, ncol=1)

    save(fig, "fig4_two_qubit_tfim")
    plt.close(fig)


def main():
    tau, g = 0.3, 0.8
    print("Conservation residual & info split (TFIM, tau=0.3, g=0.8):")
    max_res = 0.0
    for L in range(1, 8):
        Fj, Fm, Eb, r = conservation_split(L, tau, g, 0.15)
        max_res = max(max_res, r)
        print(f"  L={L}: F_joint={Fj:.6f}  F_marg={Fm:.6f}  E_b={Eb:.6f}  residual={r:.2e}")
    print(f"  max chain-rule residual over L = {max_res:.2e}")

    # labeled vs enumeration cross-check
    ee = max(abs(labeled_EFQ_tfim(L, tau, g) - labeled_EFQ_enumerate(L, tau, g))
             for L in range(1, 7))
    print(f"labeled Sum(1-<G>^2) vs enumeration: max diff = {ee:.2e}")

    print("Oracle O5 (Bell stabilizer kernel):")
    for name, (ev, c) in bell_kernel_demo().items():
        tag = "SILENT" if c < 1e-9 else "active"
        print(f"  <{name}>_Phi+ = {ev:+.3f}  c = {c:.3f}  {tag}")

    Sdec, cdec = decoupling_sanity(tau)
    print(f"g->0 decoupling: chi entanglement = {Sdec:.2e}, max c_ell = {cdec:.2e} "
          f"(product state, ZZ error silent)")

    for name, err, tol in [("residual", max_res, 1e-10),
                           ("enum", ee, 1e-9),
                           ("decouple-entangle", Sdec, 1e-9),
                           ("decouple-silent", cdec, 1e-9)]:
        if err > tol:
            raise SystemExit(f"{name} FAILED: {err:.2e} > {tol}")
    print("Conservation, kernel, and decoupling checks PASSED.\nGenerating Figure 4 ...")
    make_figure()
    print("Figure 4 done.")


if __name__ == "__main__":
    main()
