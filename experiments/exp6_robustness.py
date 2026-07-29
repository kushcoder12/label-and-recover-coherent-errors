"""Experiment 6 / Figure 6 — Robustness of labeled coherent-error recovery under the
full standard noise profile (NOISE_MODULE.md §7).

Single-qubit calibration model (core.estimator): prepare |+>, coherent Z-error
Rz(theta*), one incoherent noise layer (amplitude damping / dephasing / depolarizing),
Y-measurement with asymmetric readout; the RC sign label s is the X-dressing.

Three truths this figure establishes (all computed EXACTLY from the model, no MC):

(a) BIAS. The full-twirl labeled correlation E[b s] is immune to label-independent
    additive offsets (readout asymmetry, non-unital T1 whose offset is orthogonal to
    the Y quadrature). Every channel acts as a multiplicative SHRINK lambda(noise):
    a NOISE-AWARE estimator inverts it exactly (bias = 0), while a NAIVE noiseless-
    model estimator is biased by -arcsin-of-the-shrink.

(b) VARIANCE. The information degrades gracefully: the per-shot labeled Fisher
    F_noisy = lambda^2-ish shrinks with noise but stays finite; the CRB 1/(M F_noisy)
    is saturated by the noise-aware MLE (validated separately in tests/test_noise_estimation).

(c) UNBIASEDNESS vs the MARGINAL route. Under asymmetric readout the full-twirl
    labeled estimate stays unbiased while the marginal / partial-twirl estimate is
    biased by exactly the readout offset (e10 - e01). This is the sorting: coherent
    signal -> label-correlated (recovered); additive background -> marginal (discarded).

Device anchor (NOISE_MODULE.md §8, Heron-class): 1q gate ~30 ns, T1,T2 ~ 100-300 us,
so per-gate dt/T2 ~ 3e-4 (essentially clean); the sweeps below stress-test AROUND
this out to elevated noise. Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.noise import NoiseProfile
from core.estimator import (labeled_correlation, marginal_mean, noisy_labeled_fisher,
                            calib_probs)

_DT = 1.0  # dimensionless layer; sweeps are in units of dt/T and per-layer probabilities


# --------------------------------------------------------------------------
# Profiles for each swept channel
# --------------------------------------------------------------------------
def _profile_T2(x):      # x = dt/T2 (pure dephasing; T1 = inf)
    return NoiseProfile(dt=_DT, T1=np.inf, T2=(np.inf if x == 0 else _DT / x))


def _profile_T1(x):      # x = dt/T1 (pure amplitude damping; T2 = 2 T1)
    if x == 0:
        return NoiseProfile(dt=_DT, T1=np.inf, T2=np.inf)
    T1 = _DT / x
    return NoiseProfile(dt=_DT, T1=T1, T2=2 * T1)


def _profile_depol(r):   # r = single-qubit depolarizing gate error
    return NoiseProfile(dt=_DT, T1=np.inf, T2=np.inf, r_1q=r)


def _profile_readout(e10):  # asymmetric readout, e01 = 0.25 e10
    return NoiseProfile(dt=_DT, T1=np.inf, T2=np.inf, e01=0.25 * e10, e10=e10)


# --------------------------------------------------------------------------
# Exact expected estimates (no Monte-Carlo)
# --------------------------------------------------------------------------
def naive_bias(theta, profile, p=0.5):
    """Bias of the naive (noiseless-model) labeled estimator = arcsin(E[b s]) - theta."""
    C = np.clip(labeled_correlation(theta, profile, p), -1, 1)
    return float(np.arcsin(C) - theta)


def noise_aware_bias(theta, profile, p=0.5, grid=4001):
    """Bias of the noise-aware MLE via its expected log-likelihood maximizer.

    Maximize sum_{s,b} pi(s) P(b|s,theta_true) log P(b|s,theta) over theta -> theta_true
    (Gibbs' inequality), so this is ~0 up to grid resolution. Deterministic (no MC).
    """
    thetas = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, grid)
    # expected outcome probs at theta_true
    pr_true = {s: calib_probs(theta, s, profile) for s in (1, -1)}
    best_t, best_ll = 0.0, -np.inf
    for th in thetas:
        ll = 0.0
        for s, pi in [(1, p), (-1, 1 - p)]:
            pr = calib_probs(th, s, profile)
            pr = np.clip(pr, 1e-12, 1)
            ll += pi * (pr_true[s][0] * np.log(pr[0]) + pr_true[s][1] * np.log(pr[1]))
        if ll > best_ll:
            best_ll, best_t = ll, th
    return float(best_t - theta)


def marginal_bias(theta, profile, p, mu):
    """Bias of the marginal (label-forgotten) estimator theta_hat = arcsin(E[b]/mu)."""
    if abs(mu) < 1e-9:
        return np.nan
    val = np.clip(marginal_mean(theta, profile, p) / mu, -1, 1)
    return float(np.arcsin(val) - theta)


# --------------------------------------------------------------------------
# Figure 6
# --------------------------------------------------------------------------
def make_figure(theta=0.3):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.5))

    sweeps = [
        ("dephasing  $\\delta t/T_2$", np.linspace(0, 0.5, 26), _profile_T2, C["blue"]),
        ("relaxation  $\\delta t/T_1$", np.linspace(0, 0.5, 26), _profile_T1, C["vermil"]),
        ("depolarizing  $r_{1q}$", np.linspace(0, 0.1, 26), _profile_depol, C["green"]),
        ("readout  $e_{10}$", np.linspace(0, 0.1, 26), _profile_readout, C["purple"]),
    ]

    # ---- (a) bias: naive (per channel) vs noise-aware (flat 0) ----
    for label, xs, mk, col in sweeps:
        biases = [naive_bias(theta, mk(x)) for x in xs]
        axA.plot(xs / xs.max(), biases, "-", color=col, lw=2, label=f"naive, {label}")
    # noise-aware, one representative channel (all give ~0); verify a few points
    xs = np.linspace(0, 0.5, 11)
    aware = [noise_aware_bias(theta, _profile_T2(x)) for x in xs]
    axA.plot(xs / xs.max(), aware, "k--", lw=2, label="noise-aware (all channels)")
    axA.axhline(0, color="0.6", lw=0.8, zorder=0)
    axA.set_xlabel("noise strength  (fraction of swept max)")
    axA.set_ylabel(r"bias  $\hat\theta - \theta^*$  (rad)")
    axA.set_title(r"(a) Bias: naive vs noise-aware ($\theta^*=%.1f$)" % theta)
    axA.legend(fontsize=8, loc="lower left")

    # ---- (b) Fisher information vs noise (graceful degradation) ----
    for label, xs, mk, col in sweeps:
        F = [noisy_labeled_fisher(theta, mk(x)) for x in xs]
        axB.plot(xs / xs.max(), F, "-", color=col, lw=2, label=label)
    axB.axhline(1.0, color="0.6", lw=0.8, ls=":", label=r"$F_0=1$ (quantum limit)")
    axB.set_xlabel("noise strength  (fraction of swept max)")
    axB.set_ylabel(r"per-shot labeled Fisher  $F_{\mathrm{noisy}}$")
    axB.set_title(r"(b) Information degrades gracefully")
    axB.set_ylim(0, 1.05)
    axB.legend(fontsize=8, loc="lower left")

    # ---- (c) unbiasedness under readout offset: labeled vs marginal ----
    e10s = np.linspace(0, 0.12, 26)
    lab_full = [naive_bias(theta, _profile_readout(e), p=0.5) for e in e10s]      # full twirl
    marg_partial = [marginal_bias(theta, _profile_readout(e), p=0.8, mu=0.6) for e in e10s]
    aware_full = [noise_aware_bias(theta, _profile_readout(e)) for e in e10s]
    axC.plot(e10s, marg_partial, "-s", color=C["orange"], ms=3,
             label=r"marginal (partial twirl $\mu=0.6$)")
    axC.plot(e10s, lab_full, "-^", color=C["blue"], ms=3,
             label=r"labeled naive (full twirl)")
    axC.plot(e10s, aware_full, "-o", color=C["black"], ms=3,
             label=r"labeled noise-aware (full twirl)")
    axC.axhline(0, color="0.6", lw=0.8, zorder=0)
    axC.set_xlabel(r"readout asymmetry  $e_{10}$  ($e_{01}=e_{10}/4$)")
    axC.set_ylabel(r"bias  $\hat\theta - \theta^*$  (rad)")
    axC.set_title(r"(c) Label-independent offset: cancels in full twirl")
    axC.legend(fontsize=8, loc="upper left")

    save(fig, "fig6_robustness")
    plt.close(fig)


def report(theta=0.3):
    print("Robustness of labeled recovery (theta* = %.2f):" % theta)
    print(f"  {'channel':16s} {'strength':>9s} {'F_noisy':>8s} {'naive_bias':>11s} {'aware_bias':>11s}")
    cases = [("T2 (dt/T2=0.3)", _profile_T2(0.3)),
             ("T1 (dt/T1=0.3)", _profile_T1(0.3)),
             ("depol r=0.05", _profile_depol(0.05)),
             ("readout e10=0.08", _profile_readout(0.08))]
    for name, pr in cases:
        F = noisy_labeled_fisher(theta, pr)
        print(f"  {name:16s} {'':>9s} {F:8.4f} {naive_bias(theta, pr):+11.4f} "
              f"{noise_aware_bias(theta, pr):+11.4f}")
    # unbiasedness headline: readout offset cancellation
    e = 0.08
    off = e - 0.25 * e
    mb = marginal_bias(theta, _profile_readout(e), p=0.8, mu=0.6)
    lb = noise_aware_bias(theta, _profile_readout(e))
    print(f"readout e10={e}: offset(e10-e01)={off:.3f}  marginal bias={mb:+.4f}  "
          f"labeled(noise-aware) bias={lb:+.4f}")


def main():
    report()
    # sanity: noise-aware unbiased to grid resolution across channels
    max_aware = max(abs(noise_aware_bias(0.3, p))
                    for p in [_profile_T2(0.4), _profile_T1(0.4), _profile_depol(0.08),
                              _profile_readout(0.1)])
    print(f"max |noise-aware bias| over channels = {max_aware:.2e}")
    if max_aware > 2e-3:
        raise SystemExit(f"noise-aware estimator not unbiased: {max_aware:.2e}")
    print("Noise-aware recovery is unbiased under the full profile.\nGenerating Figure 6 ...")
    make_figure()
    print("Figure 6 done.")


if __name__ == "__main__":
    main()
