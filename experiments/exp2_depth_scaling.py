"""Experiment 2 / Figure 2 — No washout of coherent information with depth.

Single-qubit L-cycle Trotter model (CLAUDE.md §2.3). Three per-shot quantities
about the coherent-error angle theta:

  labeled  E[F_Q] = sum_ell (1 - <G-tilde_ell>^2) = sum_ell sin^2(ell*alpha)   (S5)  -> ~ L/2
  coherent F_Q^coh = 4 Var_chi(1/2 sum_ell G-tilde_ell) = (sum_ell sin(ell*alpha))^2 (S8) -> bounded
  marginal (standard RC, labels forgotten): QFI of the label-averaged mixed state -> ~ 0 (S6)

All computed from the simulated circuit and cross-checked against Oracles O2 & O3
(CLAUDE.md §7). Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import trotter_single_qubit
from core.pauli import expectation
from core.fisher import qfi_variance, qfi_fidelity, fisher_from_hist
from core.twirl import all_sign_patterns, pattern_probability, sample_sign_pattern


# --------------------------------------------------------------------------
# Core depth quantities (from the simulated circuit)
# --------------------------------------------------------------------------
def _gens_and_chi(L, alpha):
    _, chi, gens = trotter_single_qubit(L, alpha, 0.0, [1] * L)
    return gens, chi


def labeled_EFQ(L, alpha):
    """Labeled E[F_Q] = sum_ell (1 - <G-tilde_ell>_chi^2)  (S5). Cheap & exact (no enumeration)."""
    gens, chi = _gens_and_chi(L, alpha)
    return float(sum(1.0 - expectation(chi, G) ** 2 for G in gens))


def labeled_EFQ_enumerate(L, alpha, p=0.5):
    """Labeled E[F_Q] by EXACT enumeration of 2^L patterns: E_s[4 Var_chi(1/2 sum s_ell G_ell)].

    Cross-check of labeled_EFQ; equals it at full twirl p=0.5 (mu=0). L <= ~12.
    """
    gens, chi = _gens_and_chi(L, alpha)
    pats = all_sign_patterns(L)
    tot = 0.0
    for pat in pats:
        Gcal = 0.5 * sum(pat[i] * gens[i] for i in range(L))
        tot += pattern_probability(pat, p) * qfi_variance(chi, Gcal)
    return float(tot)


def coherent_FQ(L, alpha):
    """No-twirl coherent QFI = 4 Var_chi(1/2 sum_ell G-tilde_ell) = (sum sin(ell*alpha))^2 (S8)."""
    gens, chi = _gens_and_chi(L, alpha)
    Gcoh = 0.5 * sum(gens)
    return float(qfi_variance(chi, Gcoh))


def marginal_FQ(L, alpha, theta, p=0.5):
    """Standard-RC marginal Fisher for the natural balanced readout (S6).

    A real experiment measures a FIXED, theta-independent observable whose ideal
    expectation vanishes at theta=0 (the equatorial 'Y-like' signal observable,
    the depth analog of measuring Y in the single-qubit toy of Sec 2.2). We take
    the projective measurement in the balanced basis

        |chi_pm> = (|chi> +/- |chi_perp>)/sqrt(2),
        |chi_perp> proportional to (Gbar - <Gbar>)|chi>,   Gbar = (1/2) sum_ell G-tilde_ell,

    so that P(chi_+)=P(chi_-)=1/2 at theta=0 (balanced -> no diverging Fisher).
    Forgetting the labels cancels the first-order theta-signal (it is odd in s),
    leaving F_marg = O(theta^2) -> ~0 at all L. Exact enumeration over 2^L
    patterns (small L). This is the honest 'standard RC loses the coherent info'
    quantity; contrast with the optimal-measurement QFI, which would recover it.
    """
    gens, chi = _gens_and_chi(L, alpha)
    Gbar = 0.5 * sum(gens)
    v = Gbar @ chi
    v = v - (chi.conj() @ v) * chi                # component orthogonal to |chi>
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:                               # coherent generator leaves chi invariant
        return 0.0
    chi_perp = v / nrm
    plus = (chi + chi_perp) / np.sqrt(2)
    minus = (chi - chi_perp) / np.sqrt(2)

    pats = all_sign_patterns(L)
    weights = np.array([pattern_probability(pat, p) for pat in pats])

    def prob(th):
        P = np.zeros(2)
        for w, pat in zip(weights, pats):
            psi, _, _ = trotter_single_qubit(L, alpha, th, pat)
            P[0] += w * abs(plus.conj() @ psi) ** 2
            P[1] += w * abs(minus.conj() @ psi) ** 2
        return P

    return fisher_from_hist(prob, theta)


# --------------------------------------------------------------------------
# Analytic oracle forms
# --------------------------------------------------------------------------
def labeled_analytic(L, alpha):
    return float(sum(np.sin(ell * alpha) ** 2 for ell in range(1, L + 1)))


def coherent_analytic(L, alpha):
    return float(sum(np.sin(ell * alpha) for ell in range(1, L + 1)) ** 2)


# --------------------------------------------------------------------------
# Oracle verification
# --------------------------------------------------------------------------
_O2 = {  # alpha = 1.0 rad. NOTE: L=5,6 labeled corrected from the original hand table
         # (3.047295, 3.125368) to the exact Sum sin^2(l*alpha) = (3.047096, 3.125169);
         # a ~2e-4 arithmetic slip in the source table, confirmed by 4 independent
         # methods (closed form, 1-<G>^2, 2^L enumeration, Method-A fidelity). See HISTORY.md.
    1: (0.708073, 0.708073),
    2: (1.534895, 3.065190),
    3: (1.554810, 3.579242),
    4: (2.127560, 1.288420),
    5: (3.047096, 0.031033),
    6: (3.125169, 0.010661),
}
_O3 = {  # alpha = pi/2 : labeled ceil(L/2); coherent 1,1,0,0,1,1
    1: (1, 1), 2: (1, 1), 3: (2, 0), 4: (2, 0), 5: (3, 1), 6: (3, 1),
}


def verify_oracle_O2(verbose=True):
    alpha = 1.0
    max_err = 0.0
    if verbose:
        print("Oracle O2  (alpha = 1.0 rad)")
        print(f"  {'L':>2} {'labeled(code)':>14} {'labeled(orc)':>13} "
              f"{'coher(code)':>12} {'coher(orc)':>11}")
    for L, (lab_o, coh_o) in _O2.items():
        lab = labeled_EFQ(L, alpha)
        coh = coherent_FQ(L, alpha)
        max_err = max(max_err, abs(lab - lab_o), abs(coh - coh_o))
        if verbose:
            print(f"  {L:>2} {lab:>14.6f} {lab_o:>13.6f} {coh:>12.6f} {coh_o:>11.6f}")
    if verbose:
        print(f"  max|code - oracle| = {max_err:.2e}")
    return max_err


def verify_oracle_O3(verbose=True):
    alpha = np.pi / 2
    max_err = 0.0
    if verbose:
        print("Oracle O3  (alpha = pi/2)")
        print(f"  {'L':>2} {'labeled(code)':>14} {'labeled(orc)':>13} "
              f"{'coher(code)':>12} {'coher(orc)':>11}")
    for L, (lab_o, coh_o) in _O3.items():
        lab = labeled_EFQ(L, alpha)
        coh = coherent_FQ(L, alpha)
        max_err = max(max_err, abs(lab - lab_o), abs(coh - coh_o))
        if verbose:
            print(f"  {L:>2} {lab:>14.6f} {lab_o:>13.0f} {coh:>12.6f} {coh_o:>11.0f}")
    if verbose:
        print(f"  max|code - oracle| = {max_err:.2e}")
    return max_err


def verify_methods_agree(verbose=True):
    """Method A (fidelity QFI on evolved state) vs Method B (generator variance),
    for random fixed patterns (CLAUDE.md §2.4 mandatory cross-check)."""
    rng = np.random.default_rng(0)
    alpha, theta = 1.0, 1e-3
    max_err = 0.0
    for L in [2, 4, 6, 8]:
        gens, chi = _gens_and_chi(L, alpha)
        pat = sample_sign_pattern(L, 0.5, rng)
        Gcal = 0.5 * sum(pat[i] * gens[i] for i in range(L))
        fb = qfi_variance(chi, Gcal)
        state_fn = lambda th, pat=pat: trotter_single_qubit(L, alpha, th, pat)[0]
        fa = qfi_fidelity(state_fn, theta)
        max_err = max(max_err, abs(fa - fb))
        if verbose:
            print(f"  L={L}: Method A={fa:.6f}  Method B={fb:.6f}  |diff|={abs(fa-fb):.2e}")
    return max_err


# --------------------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------------------
def make_figure(alpha=1.0, theta_marg=0.2, L_max=30, L_marg_max=12):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    Ls = np.arange(1, L_max + 1)
    labeled = np.array([labeled_EFQ(L, alpha) for L in Ls])
    coherent = np.array([coherent_FQ(L, alpha) for L in Ls])
    coh_bound = 1.0 / np.sin(alpha / 2) ** 2

    Lm = np.arange(1, L_marg_max + 1)
    marginal = np.array([marginal_FQ(L, alpha, theta_marg) for L in Lm])

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ax.plot(Ls, labeled, "-o", color=C["vermil"], ms=4,
            label=r"labeled RC (keep labels): $\mathbb{E}_s[F_Q]=\sum_\ell \sin^2(\ell\alpha)$")
    ax.plot(Ls, 0.5 * Ls, ls="--", color=C["vermil"], lw=1.2, alpha=0.7,
            label=r"$L/2$ asymptote")
    ax.plot(Ls, coherent, "-s", color=C["blue"], ms=4,
            label=r"no twirl (coherent): $(\sum_\ell \sin(\ell\alpha))^2$")
    ax.axhline(coh_bound, ls=":", color=C["blue"], lw=1.2, alpha=0.8,
               label=r"coherent bound $1/\sin^2(\alpha/2)$")
    ax.plot(Lm, marginal, "-^", color=C["grey"], ms=4,
            label=rf"standard RC (marginal, $\theta={theta_marg}$) $\approx 0$")

    ax.set_xlabel(r"circuit depth  $L$  (Trotter cycles)")
    ax.set_ylabel(r"Fisher information about $\theta$  (per shot)")
    ax.set_title(rf"Coherent-error information survives depth  ($\alpha={alpha}$ rad)")
    ax.set_xlim(0, L_max + 0.5)
    ax.set_ylim(-0.5, max(labeled.max(), coherent.max()) * 1.08)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig2_depth_scaling")
    plt.close(fig)


def main():
    e2 = verify_oracle_O2()
    e3 = verify_oracle_O3()
    print("Method A vs B cross-check:")
    em = verify_methods_agree()
    print(f"  max|A - B| = {em:.2e}")
    # enumeration cross-check of labeled
    ee = max(abs(labeled_EFQ(L, 1.0) - labeled_EFQ_enumerate(L, 1.0)) for L in range(1, 9))
    print(f"labeled analytic vs enumeration: max diff = {ee:.2e}")

    for name, err, tol in [("O2", e2, 1e-6), ("O3", e3, 1e-6),
                           ("A-vs-B", em, 1e-3), ("enum", ee, 1e-9)]:
        if err > tol:
            raise SystemExit(f"{name} FAILED: {err:.2e} > {tol}")
    print("Oracles O2, O3 and cross-checks PASSED.\nGenerating Figure 2 ...")
    make_figure()
    print("Figure 2 done.")


if __name__ == "__main__":
    main()
