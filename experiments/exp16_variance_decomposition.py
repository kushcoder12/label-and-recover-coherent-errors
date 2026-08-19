"""Experiment 16 / Figure 18 — The measured (not assumed) precision cost of resolving
phi_1j.

Two checks, both against Monte Carlo, neither against a hard-coded number
(DRIVE_CROSSTALK_SPEC Sec 2.5, 5):

  1. The structural variance relation Var(theta_1_0_hat) = Var(Theta_hat)
     + 2.25*Var(Phi_hat) (exact because theta_1_0_hat = Theta_hat - 1.5*Phi_hat and
     c_j _|_ p_j makes Theta_hat, Phi_hat independent) holds within Monte Carlo noise.
  2. The real "precision cost of resolving phi_1j" figure: Var(theta_1_0_hat) from the
     3-label estimator vs Var(theta0_hat) from Part 2's 2-label estimator ALONE (no
     p_j, no phi_1j to resolve), at the SAME total shot budget. This ratio is measured
     here, not assumed -- Sec 7 explicitly forbids hard-coding any specific multiplier.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import three_label_probe, two_qubit_zz_probe
from core.multiqubit_zz import stratified_three_label_estimator, joint_correlation_estimator

import exp14_drive_crosstalk_oracle as exp14

THETA0 = exp14.THETA0
KAPPA_1J = exp14.KAPPA_1J
PHI_1J = exp14.PHI_1J
M_PROBE = 8000
TRIALS = 300
SEED = 20260817


def _three_label_trial(rng):
    recs = exp14.sample_shots(THETA0, KAPPA_1J, PHI_1J, M_PROBE, rng=rng)
    fit = stratified_three_label_estimator(recs, B=2, seed=int(rng.integers(1, 1 << 30)))
    Theta_hat = fit["theta_1_0_hat"] + 1.5 * fit["phi_1j_hat"]
    return fit["theta_1_0_hat"], Theta_hat, fit["phi_1j_hat"]


def _two_label_baseline_trial(rng):
    """Part 2's 2-label estimator alone, at the same M_PROBE shots, phi_1j=0 (nothing
    to resolve) -- the 'cost of resolving only 2 parameters' baseline."""
    s = np.where(rng.random(M_PROBE) < 0.5, 1, -1)
    c = np.where(rng.random(M_PROBE) < 0.5, 1, -1)
    u = rng.random(M_PROBE)
    table = {(si, ci): two_qubit_zz_probe(THETA0, KAPPA_1J, si, ci) for si in (1, -1) for ci in (1, -1)}
    records = []
    for si, ci, uu in zip(s, c, u):
        y = table[(int(si), int(ci))]
        p_plus = np.clip((1.0 + y) / 2.0, 0.0, 1.0)
        b = 1 if uu < p_plus else -1
        records.append((int(si), {2: int(ci)}, b))
    fit = joint_correlation_estimator(records, 1, [2], B=2, seed=int(rng.integers(1, 1 << 30)))
    return fit["theta0_hat"]


def run_variance_decomposition(trials=TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    theta_ests, Theta_ests, phi_ests = [], [], []
    for _ in range(trials):
        th, Th, ph = _three_label_trial(rng)
        theta_ests.append(th)
        Theta_ests.append(Th)
        phi_ests.append(ph)
    return (np.array(theta_ests), np.array(Theta_ests), np.array(phi_ests))


def run_two_label_baseline(trials=TRIALS, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    return np.array([_two_label_baseline_trial(rng) for _ in range(trials)])


# ---------------------------------------------------------------------------
# Figure 18
# ---------------------------------------------------------------------------
def make_figure(var_theta_1_0, var_baseline):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    ratio = var_theta_1_0 / var_baseline
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.bar([0, 1], [var_baseline, var_theta_1_0], width=0.55,
          color=[C["grey"], C["blue"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"2-label baseline" "\n" r"(no $\phi_{1j}$ to resolve)",
                        r"3-label" "\n" r"(resolves $\phi_{1j}$ too)"])
    ax.set_ylabel(r"measured $\mathrm{Var}(\hat\theta_1^{(0)})$" f"  (M={M_PROBE} shots/trial)")
    ax.set_title(f"Measured precision cost of resolving " r"$\phi_{1j}$"
                f"\nratio = {ratio:.2f}x  ({TRIALS} trials)")
    for i, v in enumerate([var_baseline, var_theta_1_0]):
        ax.annotate(f"{v:.2e}", (i, v), textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=10)
    save(fig, "fig18_variance_cost")
    plt.close(fig)


def main():
    print(f"Monte Carlo: {TRIALS} trials, M={M_PROBE} shots/trial, "
          f"theta_1_0={THETA0}, kappa_1j={KAPPA_1J}, phi_1j={PHI_1J}.\n")

    theta_ests, Theta_ests, phi_ests = run_variance_decomposition()
    var_theta = float(np.var(theta_ests, ddof=1))
    var_Theta = float(np.var(Theta_ests, ddof=1))
    var_phi = float(np.var(phi_ests, ddof=1))
    predicted = var_Theta + 2.25 * var_phi
    ratio_struct = var_theta / predicted

    print("Structural relation Var(theta_1_0_hat) = Var(Theta_hat) + 2.25*Var(Phi_hat):")
    print(f"  Var(theta_1_0_hat) measured  = {var_theta:.4e}")
    print(f"  Var(Theta_hat) + 2.25*Var(Phi_hat) = {var_Theta:.4e} + 2.25*{var_phi:.4e} "
          f"= {predicted:.4e}")
    print(f"  ratio = {ratio_struct:.4f}  (predicts 1.0)")
    if not (0.75 < ratio_struct < 1.35):
        raise SystemExit(f"variance decomposition FAILED: ratio {ratio_struct:.4f} "
                         f"too far from 1.0 at {TRIALS} trials -- either the "
                         f"independence assumption or the trial count needs revisiting.")
    print("  PASSED (within Monte Carlo tolerance).\n")

    print("Real precision cost: Var(theta_1_0_hat) [3-label] vs Var(theta0_hat) "
          "[Part 2's 2-label estimator alone, same shot budget, nothing to resolve]:")
    baseline_ests = run_two_label_baseline()
    var_baseline = float(np.var(baseline_ests, ddof=1))
    cost_ratio = var_theta / var_baseline
    print(f"  Var(theta_1_0_hat), 3-label  = {var_theta:.4e}")
    print(f"  Var(theta0_hat), 2-label     = {var_baseline:.4e}")
    print(f"  measured ratio = {cost_ratio:.3f}x  <- the real number, not assumed.")
    print("  (Sec 7: no specific multiplier is hard-coded anywhere; this is what the "
          "simulation actually produced.)")

    print("\nGenerating Figure 18 ...")
    make_figure(var_theta, var_baseline)
    print("Figure 18 done.")


if __name__ == "__main__":
    main()
