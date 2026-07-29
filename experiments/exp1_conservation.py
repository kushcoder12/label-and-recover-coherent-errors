"""Experiment 1 / Figure 1 — Fisher-information conservation & complementarity.

Single-qubit E/R model (CLAUDE.md §2.2): coherent Z-error Rz(s*theta) on |+>,
sign label s in {+1,-1} with P(+1)=p (mean sign mu=2p-1), measured in Y.

Everything is computed NUMERICALLY from the simulated circuit statistics and
cross-checked against the analytic model and Oracle O1 (CLAUDE.md §7).

Model functions are import-safe (no side effects); tests import them.
Run `python experiments/exp1_conservation.py` to (re)generate Figure 1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pauli import pauli, expectation
from core.twirl import apply_signed_z_error
from core.fisher import fisher_from_hist, decompose_joint_fisher

_Y = pauli("Y")
_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


# --------------------------------------------------------------------------
# Physics model (numeric — simulated circuit)
# --------------------------------------------------------------------------
def mean_Y(theta, s):
    """<Y> after Rz(s*theta)|+>. Equals s*sin(theta) analytically (E2)."""
    st = apply_signed_z_error(_PLUS.copy(), theta, s, 1, 1)
    return expectation(st, _Y)


def _binary_probs(mean):
    """Outcome distribution [P(b=+1), P(b=-1)] for a Y-measurement with mean <Y>."""
    return np.array([(1 + mean) / 2, (1 - mean) / 2])


def joint_prob(theta, p):
    """Joint P[b, s] over outcome b in {+1,-1} (rows) and label s in {+1,-1} (cols).

    P(b,s) = pi(s) * (1 + b <Y>_s)/2  (E2), with pi(+)=p, pi(-)=1-p.
    """
    out = np.zeros((2, 2))
    for si, s in enumerate((+1, -1)):
        pi_s = p if s == 1 else (1 - p)
        m = mean_Y(theta, s)
        out[0, si] = pi_s * (1 + m) / 2      # b = +1
        out[1, si] = pi_s * (1 - m) / 2      # b = -1
    return out


def numeric_decomposition(theta, p):
    """Fully numeric Fisher decomposition {F0, F_marg, E_b, F_joint} at (theta, p)."""
    F_joint, F_marg, E_b = decompose_joint_fisher(lambda th: joint_prob(th, p), theta)

    # F0 = conditional (labeled) Fisher: average over s of the per-label Fisher.
    F0 = 0.0
    for s in (+1, -1):
        pi_s = p if s == 1 else (1 - p)
        prob_fn = lambda th, s=s: _binary_probs(mean_Y(th, s))
        F0 += pi_s * fisher_from_hist(prob_fn, theta)
    return dict(F0=F0, F_marg=F_marg, E_b=E_b, F_joint=F_joint)


# --------------------------------------------------------------------------
# Analytic model (Oracle O1)
# --------------------------------------------------------------------------
def analytic_decomposition(theta, mu):
    """Analytic {F0, F_marg, E_b} (E5, E7, R1). F0 = 1 exactly."""
    F0 = 1.0
    F_marg = mu**2 * np.cos(theta) ** 2 / (1 - mu**2 * np.sin(theta) ** 2)
    return dict(F0=F0, F_marg=F_marg, E_b=F0 - F_marg)


def leading_order(T):
    """R2 leading-order complementarity: F_marg=(1-T)^2, E_b=2T-T^2 (CLAUDE.md §2.2)."""
    return (1 - T) ** 2, 2 * T - T**2


# --------------------------------------------------------------------------
# Oracle O1 verification
# --------------------------------------------------------------------------
def verify_oracle_O1(verbose=True):
    """Reproduce Oracle O1 (theta=0.2) numerically & analytically. Returns max error."""
    theta = 0.2
    rows = [(1.0, 1.000000, 0.000000),
            (0.5, 0.242526, 0.757474),
            (0.0, 0.000000, 1.000000)]
    max_err = 0.0
    if verbose:
        print(f"Oracle O1  (theta = {theta})")
        print(f"  {'mu':>5} {'F_marg(num)':>12} {'F_marg(oracle)':>14} "
              f"{'E_b(num)':>10} {'E_b(oracle)':>12} {'sum':>8}")
    for mu, fm_oracle, eb_oracle in rows:
        p = (1 + mu) / 2
        num = numeric_decomposition(theta, p)
        err = max(abs(num["F_marg"] - fm_oracle), abs(num["E_b"] - eb_oracle),
                  abs(num["F0"] - 1.0), abs(num["F_joint"] - 1.0))
        max_err = max(max_err, err)
        if verbose:
            print(f"  {mu:>5.1f} {num['F_marg']:>12.6f} {fm_oracle:>14.6f} "
                  f"{num['E_b']:>10.6f} {eb_oracle:>12.6f} "
                  f"{num['F_marg'] + num['E_b']:>8.4f}")
    if verbose:
        print(f"  max|code - oracle| = {max_err:.2e}")
    return max_err


# --------------------------------------------------------------------------
# Figure 1
# --------------------------------------------------------------------------
def make_figure(theta=0.2):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    T = np.linspace(0.0, 1.0, 201)
    mu = 1.0 - T                                   # mu >= 0 branch; T = 1 - |mu|
    d = [analytic_decomposition(theta, m) for m in mu]
    F_marg_exact = np.array([x["F_marg"] for x in d])
    E_b_exact = np.array([x["E_b"] for x in d])
    F_marg_lo, E_b_lo = leading_order(T)

    # numeric validation markers at a few T
    T_mark = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    num_marg, num_eb = [], []
    for Tm in T_mark:
        d = numeric_decomposition(theta, (1 + (1 - Tm)) / 2)
        num_marg.append(d["F_marg"])
        num_eb.append(d["E_b"])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(T, F_marg_exact, color=C["blue"], lw=2.2,
            label=r"$F_{\mathrm{marg}}$  (RC keeps this)")
    ax.plot(T, E_b_exact, color=C["vermil"], lw=2.2,
            label=r"$\mathbb{E}_b[F_{s|b}]$  (RC discards — recoverable)")
    ax.plot(T, F_marg_exact + E_b_exact, color=C["black"], lw=2.0, ls="--",
            label=r"sum $= F_0 = 1$  (conserved)")
    # leading-order R2 references
    ax.plot(T, F_marg_lo, color=C["blue"], lw=1.0, ls=":", alpha=0.7,
            label=r"$(1-T)^2$ (leading order)")
    ax.plot(T, E_b_lo, color=C["vermil"], lw=1.0, ls=":", alpha=0.7,
            label=r"$2T-T^2$ (leading order)")
    # numeric markers
    ax.plot(T_mark, num_marg, "o", color=C["blue"], mec="k", mew=0.5, zorder=5)
    ax.plot(T_mark, num_eb, "s", color=C["vermil"], mec="k", mew=0.5, zorder=5)

    ax.annotate("RC keeps this", xy=(0.18, analytic_decomposition(theta, 0.82)["F_marg"]),
                xytext=(0.05, 0.35), fontsize=10, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.0))
    ax.annotate("RC discards this\n(recoverable from labels)",
                xy=(0.8, analytic_decomposition(theta, 0.2)["E_b"]),
                xytext=(0.42, 0.55), fontsize=10, color=C["vermil"],
                arrowprops=dict(arrowstyle="->", color=C["vermil"], lw=1.0))

    ax.set_xlabel(r"tailoring  $T = 1 - |\mu|$")
    ax.set_ylabel(r"Fisher information about $\theta$  (units of $F_0$)")
    ax.set_title(r"Conservation of coherent-error information under RC "
                 rf"($\theta={theta}$)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    save(fig, "fig1_conservation")
    plt.close(fig)


def main():
    max_err = verify_oracle_O1()
    tol = 1e-6
    if max_err > tol:
        raise SystemExit(f"Oracle O1 FAILED: max error {max_err:.2e} > {tol}")
    print("Oracle O1 PASSED.\nGenerating Figure 1 ...")
    make_figure()
    print("Figure 1 done.")


if __name__ == "__main__":
    main()
