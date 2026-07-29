"""Experiment 10 / Figure 10 — Shot-cost positioning vs RPE (GAPS_CODE.md GAP 5).

When is labeled-RC the cheaper way to learn a coherent parameter?

Analytical anchors (also in the paper):
  * Labeled-RC precision (CRB with visibility):  Var(theta_hat) = 1 / (M v^2 F0),
    where M shots are ALREADY spent on the RC run and F0 = 1. The marginal cost of the
    coherent map is ~0: it rides on RC. Dedicated shots to reach precision delta beyond
    an RC budget N_run:  max(0, 1/(delta^2 v^2) - N_run).
  * Robust phase estimation (RPE): Heisenberg-like Var ~ 1/(M d^2 c^{2d}) using
    DEDICATED circuits of growing depth d (decoherence per application c). Optimal depth
    d* = -1/ln(c); dedicated shots  ~ 1/(delta^2 d*^2 c^{2 d*}).
  * Cycle benchmarking / Pauli-noise learning measures the STOCHASTIC (diagonal) rates
    only -- NOT the coherent parameter -- so it is not a competitor here (stated, not plotted).

Numerical confirmation: labeled-RC variance matches the CRB (extends the Fig-6 check);
a minimal single-optimal-depth RPE shows the Heisenberg ~1/d^2 improvement (rolling off
once c^{2d} decoherence dominates). Fig 10 plots dedicated shots-to-precision vs delta for
both, shades where labeled-RC is free (rides on RC), and annotates the crossover.

Model functions are import-safe; tests import them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# Analytical shot costs
# --------------------------------------------------------------------------
def labeled_rc_shots(delta, v, F0=1.0):
    """Shots for labeled-RC to reach 1-sigma precision delta: 1/(delta^2 v^2 F0)."""
    return 1.0 / (delta**2 * v**2 * F0)


def rpe_optimal_depth(c):
    """d* = argmax_d d^2 c^{2d} = -1/ln(c)."""
    return -1.0 / np.log(c)


def rpe_fisher_per_shot(d, c):
    """Per-shot Fisher of a depth-d amplified Ramsey estimate of theta: d^2 c^{2d}."""
    return d**2 * c ** (2 * d)


def rpe_shots(delta, c, ladder_overhead=True):
    """Dedicated shots for a minimal RPE to reach precision delta at optimal depth d*."""
    dstar = rpe_optimal_depth(c)
    F = rpe_fisher_per_shot(dstar, c)
    n = 1.0 / (delta**2 * F)
    if ladder_overhead:                       # ~log2(d*) generations in the depth ladder
        n *= max(1.0, np.log2(max(dstar, 2.0)))
    return n


# --------------------------------------------------------------------------
# Numerical verification (Monte-Carlo)
# --------------------------------------------------------------------------
def labeled_rc_mc_variance(theta, v, M, trials=800, rng=None):
    """MC Var(theta_hat) for labeled-RC: outcome mean m = v sin(theta), estimator
    theta_hat = arcsin( mean(y) / v ). Should match the CRB 1/(M v^2)."""
    if rng is None:
        rng = np.random.default_rng(0)
    m = v * np.sin(theta)                      # E[y] with y = b*s in {+1,-1}
    ests = []
    for _ in range(trials):
        y = np.where(rng.random(M) < (1 + m) / 2, 1, -1)
        ests.append(np.arcsin(np.clip(np.mean(y) / v, -1, 1)))
    return float(np.var(ests))


def rpe_depth_mc_variance(theta, d, c, M, trials=800, rng=None):
    """MC Var(theta_hat) for a single-depth (d) two-quadrature Ramsey estimate.
    P_cos = (1 + c^d cos(d theta))/2, P_sin = (1 + c^d sin(d theta))/2 ; theta small so
    d*theta < pi (no unwrapping). Heisenberg: Var ~ 1/(M d^2 c^{2d})."""
    if rng is None:
        rng = np.random.default_rng(0)
    cd = c ** d
    pc = (1 + cd * np.cos(d * theta)) / 2
    ps = (1 + cd * np.sin(d * theta)) / 2
    half = max(M // 2, 1)
    ests = []
    for _ in range(trials):
        xc = np.mean(np.where(rng.random(half) < pc, 1, -1))   # = cd cos(d theta) + noise
        xs = np.mean(np.where(rng.random(half) < ps, 1, -1))
        phi = np.arctan2(xs, xc)               # estimates d*theta
        ests.append((phi % (2 * np.pi)) / d if phi >= 0 else phi / d)
    return float(np.var(ests))


# --------------------------------------------------------------------------
# Figure 10
# --------------------------------------------------------------------------
def make_figure(v_rc=0.85, c_gate=0.99, N_run=1e5):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    deltas = np.logspace(-1, -3.3, 60)         # target precision (rad)
    rc_total = labeled_rc_shots(deltas, v_rc)
    rc_dedicated = np.maximum(rc_total - N_run, 0.0)
    rpe_ded = np.array([rpe_shots(d, c_gate) for d in deltas])
    delta_c = 1.0 / np.sqrt(N_run * v_rc**2)   # RC budget just reaches this precision

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    # (a) dedicated shots-to-precision
    axA.loglog(deltas, np.maximum(rpe_ded, 0.5), "-s", color=C["vermil"], ms=3,
               label="RPE (dedicated deep circuits)")
    axA.loglog(deltas, np.maximum(rc_dedicated, 0.5), "-o", color=C["blue"], ms=3,
               label="labeled-RC (extra shots beyond RC run)")
    axA.axvspan(delta_c, deltas.max(), color=C["green"], alpha=0.15,
                label="labeled-RC free (rides on RC)")
    axA.axvline(delta_c, color=C["green"], lw=1.2, ls="--")
    axA.annotate(rf"crossover $\delta_c={delta_c:.1e}$", xy=(delta_c, 5),
                 xytext=(delta_c * 1.3, 50), fontsize=9, color=C["green"],
                 arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0))
    axA.set_xlabel(r"target precision  $\delta$  (rad, 1-$\sigma$)")
    axA.set_ylabel(r"dedicated shots to reach $\delta$")
    axA.set_title(rf"(a) Shot cost ($v_{{RC}}={v_rc}$, $c={c_gate}$, RC run $={N_run:.0e}$)")
    axA.invert_xaxis()
    axA.set_ylim(0.5, None)
    axA.legend(loc="upper left", fontsize=8.5)

    # (b) verification: RPE Heisenberg 1/d^2 scaling, with c^{2d} decoherence rolloff.
    # (The labeled-RC CRB saturation is the Fig-6 result and is printed to stdout.)
    rng = np.random.default_rng(0)
    ds = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    var_meas, ideal = [], []
    for d in ds:
        v = rpe_depth_mc_variance(0.01, int(d), c_gate, 8000, trials=1500, rng=rng)
        var_meas.append(v)
        ideal.append(1.0 / (8000 // 2 * rpe_fisher_per_shot(int(d), c_gate)))
    var_meas = np.array(var_meas)
    axB.loglog(ds, var_meas, "-s", color=C["vermil"], ms=5, base=2,
               label=r"RPE Var$(\hat\theta)$ (simulated)")
    axB.loglog(ds, 1.0 / (var_meas[0] ** -1 * ds.astype(float) ** 2), ":", color=C["blue"],
               base=2, lw=1.5, label=r"$\propto 1/d^2$ (Heisenberg)")
    dstar = rpe_optimal_depth(c_gate)
    axB.axvline(dstar, color=C["green"], lw=1.2, ls="--",
                label=rf"$d^*=-1/\ln c \approx {dstar:.0f}$")
    axB.set_xlabel(r"RPE depth  $d$  (gate repetitions)")
    axB.set_ylabel(r"Var$(\hat\theta)$  per fixed shot budget")
    axB.set_title(r"(b) RPE Heisenberg $1/d^2$ scaling, decoherence rolloff at $d^*$")
    axB.legend(loc="lower left", fontsize=8.5)

    save(fig, "fig10_shotcost")
    plt.close(fig)


def main():
    v_rc, c_gate = 0.85, 0.99
    # verification 1: RC MC variance matches CRB
    rng = np.random.default_rng(1)
    rc_ratios = []
    for v in [0.6, 0.8, 1.0]:
        var = labeled_rc_mc_variance(0.3, v, 5000, trials=2000, rng=rng)
        crb = 1.0 / (5000 * v**2)
        rc_ratios.append(var / crb)
        print(f"labeled-RC v={v}: Var*={var:.3e} CRB={crb:.3e} ratio={var/crb:.3f}")
    # verification 2: RPE Heisenberg 1/d^2 (d^2*Var ~ const before decoherence)
    print("RPE d^2*Var (should be ~flat for small d, rise as c^{2d} rolls off):")
    base = None
    for d in [1, 2, 4, 8, 16]:
        var = rpe_depth_mc_variance(0.02, d, c_gate, 8000, trials=2000, rng=rng)
        s = var * d**2
        base = s if base is None else base
        print(f"  d={d:2d}: d^2*Var={s:.3e}  (ratio to d=1: {s/base:.2f})")
    if not all(0.6 < r < 1.6 for r in rc_ratios):
        raise SystemExit("labeled-RC does not match CRB")
    print("Verifications passed.\nGenerating Figure 10 ...")
    make_figure(v_rc, c_gate)
    print("Figure 10 done.")


if __name__ == "__main__":
    main()
