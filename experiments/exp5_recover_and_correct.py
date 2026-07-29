"""Experiment 5 / Figure 5 — Recover-and-correct payoff (the headline result).

A single qubit runs L cycles of an ideal drive Rx(alpha); each cycle carries a
coherent Z over-rotation of angle theta* (Rz per cycle). We compare the average
final-state fidelity to the ideal state |chi> = Rx(L alpha)|0> under three
protocols, as a function of the coherent angle theta*:

  1. no-RC          : errors coherent (same sign) -> accumulate; fidelity collapses.
  2. plain RC       : per-cycle sign twirl -> coherent error tailored into stochastic;
                      helps in the large-error regime but is uncorrected.
  3. labeled RC + correction :
                      keep the twirl labels, estimate theta* from M labeled
                      calibration shots (labeled_mle, quantum-limited F0=1), then
                      apply the per-shot correction Rz(-s_ell theta_hat) at each error
                      location. Residual ~ theta*-theta_hat -> fidelity ~ 1, limited
                      only by the estimation variance 1/M.

The shaded band is the fidelity recovered for free by keeping the labels.
Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Rx, Rz
from core.operator_on_state import single_qubit_op
from core.twirl import all_sign_patterns, pattern_probability
from core.estimator import sample_labeled_shots, labeled_mle


# --------------------------------------------------------------------------
# Circuit + fidelity primitives
# --------------------------------------------------------------------------
def _chi(alpha, L):
    s = np.array([1, 0], dtype=complex)
    for _ in range(L):
        s = single_qubit_op(s, Rx(alpha), 1, 1)
    return s


def _depth_state(alpha, angles):
    """State after cycles [Rx(alpha), Rz(angle_ell)] for angle_ell in `angles`."""
    s = np.array([1, 0], dtype=complex)
    for a in angles:
        s = single_qubit_op(s, Rx(alpha), 1, 1)
        s = single_qubit_op(s, Rz(a), 1, 1)
    return s


def fidelity_noRC(alpha, L, theta):
    """Coherent (untwirled) errors of angle theta each cycle."""
    c = _chi(alpha, L)
    st = _depth_state(alpha, [theta] * L)
    return float(abs(np.vdot(c, st)) ** 2)


def fidelity_twirled(alpha, L, eps, p=0.5):
    """Average fidelity when each cycle's error is Rz(s_ell * eps), sign-twirled.

    Exact enumeration of 2^L patterns. eps = theta* for plain RC; eps = theta*-theta_hat
    for the corrected residual. Returns E_s[|<chi|psi_s>|^2] = <chi|rho|chi>.
    """
    c = _chi(alpha, L)
    F = 0.0
    for pat in all_sign_patterns(L):
        st = _depth_state(alpha, [pat[i] * eps for i in range(L)])
        F += pattern_probability(pat, p) * abs(np.vdot(c, st)) ** 2
    return float(F)


def _twirled_grid(alpha, L, eps_grid, p=0.5):
    """Precompute fidelity_twirled on a grid of eps (for fast interpolation)."""
    return np.array([fidelity_twirled(alpha, L, e, p) for e in eps_grid])


# --------------------------------------------------------------------------
# Three-protocol curves vs theta*
# --------------------------------------------------------------------------
def payoff_curves(alpha=0.7, L=10, thetas=None, M=2000, trials=200, seed=0):
    """Return dict of arrays: theta, noRC, plainRC, corrected, corrected_std."""
    if thetas is None:
        thetas = np.linspace(0.0, 0.7, 25)
    rng = np.random.default_rng(seed)

    # grid for the corrected-residual fidelity g(eps) = fidelity_twirled(eps)
    eps_grid = np.linspace(-0.8, 0.8, 321)
    g = _twirled_grid(alpha, L, eps_grid)

    def g_interp(e):
        return np.interp(e, eps_grid, g)

    noRC, plainRC, corrected, corrected_std = [], [], [], []
    for th in thetas:
        noRC.append(fidelity_noRC(alpha, L, th))
        plainRC.append(fidelity_twirled(alpha, L, th))
        # corrected: estimate theta_hat from M labeled shots, correct, average over trials
        fc = np.empty(trials)
        for t in range(trials):
            th_hat = labeled_mle(sample_labeled_shots(th, M, rng=rng))
            fc[t] = g_interp(th - th_hat)
        corrected.append(fc.mean())
        corrected_std.append(fc.std())
    return dict(theta=np.asarray(thetas), noRC=np.asarray(noRC),
                plainRC=np.asarray(plainRC), corrected=np.asarray(corrected),
                corrected_std=np.asarray(corrected_std),
                alpha=alpha, L=L, M=M)


# --------------------------------------------------------------------------
# Figure 5
# --------------------------------------------------------------------------
def make_figure(data=None):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    if data is None:
        data = payoff_curves()
    th = data["theta"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # shaded region: fidelity recovered for free (corrected above plain RC)
    ax.fill_between(th, data["plainRC"], data["corrected"],
                    where=data["corrected"] >= data["plainRC"],
                    color=C["green"], alpha=0.15,
                    label="fidelity recovered for free")

    ax.plot(th, data["noRC"], "-o", color=C["blue"], ms=4,
            label="no RC (coherent, uncorrected)")
    ax.plot(th, data["plainRC"], "-s", color=C["orange"], ms=4,
            label="plain RC (tailored, uncorrected)")
    ax.plot(th, data["corrected"], "-^", color=C["green"], ms=4,
            label=rf"labeled RC + correction ($M={data['M']}$)")
    ax.fill_between(th, data["corrected"] - data["corrected_std"],
                    data["corrected"] + data["corrected_std"],
                    color=C["green"], alpha=0.25, lw=0)

    ax.set_xlabel(r"coherent over-rotation per cycle  $\theta^*$  (rad)")
    ax.set_ylabel(r"average final-state fidelity  $\langle\chi|\rho|\chi\rangle$")
    ax.set_title(rf"Recover-and-correct: labels restore fidelity for free "
                 rf"($L={data['L']}$, $\alpha={data['alpha']}$)")
    ax.set_xlim(th.min(), th.max())
    ax.set_ylim(min(data["noRC"].min(), data["plainRC"].min()) - 0.03, 1.01)
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "fig5_recover_and_correct")
    plt.close(fig)


def main():
    print("Computing recover-and-correct payoff curves ...")
    data = payoff_curves()
    print(f"  (alpha={data['alpha']}, L={data['L']}, M={data['M']} calibration shots)")
    print(f"  {'theta*':>7} {'no-RC':>8} {'plainRC':>8} {'corrected':>10}")
    for i in range(0, len(data["theta"]), 4):
        print(f"  {data['theta'][i]:>7.3f} {data['noRC'][i]:>8.4f} "
              f"{data['plainRC'][i]:>8.4f} {data['corrected'][i]:>10.4f}")

    # Payoff claim (honest): labeled+correction wins decisively in the meaningful
    # regime (theta* >= 0.1) and is never meaningfully worse anywhere. For very small
    # theta* the fixed estimation cost ~1/M can tie the (already near-perfect) plain RC.
    diff = data["corrected"] - data["plainRC"]
    meaningful = data["theta"] >= 0.1
    win_meaningful = np.min(diff[meaningful])
    win_max = np.max(diff)
    never_worse = np.min(diff)
    print(f"min (corrected - plainRC) for theta* >= 0.1 = {win_meaningful:+.4f}")
    print(f"max (corrected - plainRC) over range        = {win_max:+.4f}")
    print(f"min (corrected - plainRC) anywhere          = {never_worse:+.4f}")
    if win_meaningful <= 0 or never_worse < -2e-3:
        raise SystemExit("payoff FAILED: labeled+correction does not beat plain RC")
    print("Payoff confirmed (labeled RC + correction > plain RC in the useful regime)."
          "\nGenerating Figure 5 ...")
    make_figure(data)
    print("Figure 5 done.")


if __name__ == "__main__":
    main()
