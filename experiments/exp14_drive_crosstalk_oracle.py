"""Experiment 14 — Drive-crosstalk oracles DC1-DC3, noiseless.

Reproduces, in order, every numeric oracle DRIVE_CROSSTALK_SPEC Sec 3 specifies before
anything downstream (exp15's noisy sweep, exp16's variance decomposition, or the
hardware extension) is allowed to run:

  DC1 -- the three exact correlations (X, Y_c, Y_p) match brute-force averaging over
         the four (c_j, p_j) label combinations, sign of Y_p included.
  DC2 -- the full stratified recovery (core.multiqubit_zz.stratified_three_label_
         estimator) reproduces the true (theta_1_0, kappa_1j, phi_1j) to floating-point
         precision from population-exact records. The primary correctness gate.
  DC3 -- the contamination this whole document exists to fix: a naive (s_1, c_j)
         estimator that never varied p_j recovers (Theta, K), not the true
         (theta_1_0, kappa_1j) -- demonstrated numerically, not merely asserted.

Also provides `sample_shots`, the finite-shot (s_1, c_j, p_j, b_1) sampler reused by
exp15's noisy sweep, exp16's variance decomposition, and tests/test_drive_crosstalk.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import three_label_probe
from core.multiqubit_zz import stratified_three_label_estimator, _point_estimate

THETA0 = 0.10
KAPPA_1J = 0.05
PHI_1J = 0.02
SHOTS = 20000
SEED = 20260817


def combined_params(theta0=THETA0, kappa=KAPPA_1J, phi=PHI_1J):
    """Theta, K, Phi from the Sec 1 substitution."""
    return theta0 + 1.5 * phi, kappa - 0.5 * phi, phi


# ---------------------------------------------------------------------------
# Oracle DC1
# ---------------------------------------------------------------------------
def oracle_dc1_correlations():
    """Brute-force (X, Y_c, Y_p) over the 4 (c,p) combos, and the exact closed forms."""
    Theta, K, Phi = combined_params()
    X = Y_c = Y_p = 0.0
    for c in (1, -1):
        for p in (1, -1):
            m = np.sin(Theta + K * c - Phi * p)
            X += m
            Y_c += c * m
            Y_p += p * m
    X, Y_c, Y_p = X / 4, Y_c / 4, Y_p / 4
    closed = (
        np.sin(Theta) * np.cos(K) * np.cos(Phi),
        np.cos(Theta) * np.sin(K) * np.cos(Phi),
        -np.cos(Theta) * np.cos(K) * np.sin(Phi),
    )
    return (X, Y_c, Y_p), closed


# ---------------------------------------------------------------------------
# Oracle DC2
# ---------------------------------------------------------------------------
def population_records(theta0=THETA0, kappa=KAPPA_1J, phi=PHI_1J):
    """All 8 (s_1, c_j, p_j) combinations with the EXACT probe outcome as b_1 --
    population-exact (infinite-shot) records, the model identity rather than a
    statistical estimate."""
    records = []
    for s1 in (1, -1):
        for c in (1, -1):
            for p in (1, -1):
                y = three_label_probe(theta0, kappa, phi, s1, c, p)
                records.append((s1, c, p, y))
    return records


def oracle_dc2_recovery(theta0=THETA0, kappa=KAPPA_1J, phi=PHI_1J):
    fit = stratified_three_label_estimator(population_records(theta0, kappa, phi), B=2)
    return fit


# ---------------------------------------------------------------------------
# Oracle DC3
# ---------------------------------------------------------------------------
def oracle_dc3_naive_contamination(theta0=THETA0, kappa=KAPPA_1J, phi=PHI_1J):
    """The naive (s_1, c_j) estimator on data averaged over p_j (i.e. what a real
    un-padded 2-label experiment would see if phi != 0 were present but unaccounted
    for): returns (naive_theta_hat, naive_kappa_hat, Theta, K)."""
    Theta, K, _ = combined_params(theta0, kappa, phi)
    s = np.array([1, 1, -1, -1], dtype=float)
    c = np.array([1, -1, 1, -1], dtype=float)
    b = np.array([
        0.5 * (three_label_probe(theta0, kappa, phi, int(si), int(ci), 1)
              + three_label_probe(theta0, kappa, phi, int(si), int(ci), -1))
        for si, ci in zip((1, 1, -1, -1), (1, -1, 1, -1))
    ])
    naive_theta, naive_kappa = _point_estimate(s, {2: c}, b, [2])
    return float(naive_theta), float(naive_kappa[2]), float(Theta), float(K)


# ---------------------------------------------------------------------------
# Finite-shot sampler (reused by exp15, exp16, tests)
# ---------------------------------------------------------------------------
def _outcome_table(theta0, kappa, phi, noise=None):
    return {(s1, c, p): three_label_probe(theta0, kappa, phi, s1, c, p, noise=noise)
            for s1 in (1, -1) for c in (1, -1) for p in (1, -1)}


def sample_shots(theta0, kappa, phi, shots, noise=None, v=1.0, rng=None):
    """Draw `shots` owned-label records (s_1, c_j, p_j, b_1) from the exact model."""
    rng = np.random.default_rng() if rng is None else rng
    table = _outcome_table(theta0, kappa, phi, noise)
    s = np.where(rng.random(shots) < 0.5, 1, -1)
    c = np.where(rng.random(shots) < 0.5, 1, -1)
    p = np.where(rng.random(shots) < 0.5, 1, -1)
    records = []
    for si, ci, pi in zip(s, c, p):
        y = table[(int(si), int(ci), int(pi))] / v
        p_plus = np.clip((1.0 + y) / 2.0, 0.0, 1.0)
        b = 1 if rng.random() < p_plus else -1
        records.append((int(si), int(ci), int(pi), b))
    return records


def _hardcoded_recovery_view():
    """(true, naive-2-label, 3-label) for theta_1_0 and kappa_1j -- the numbers Fig 17
    plots, computed the same way main() reports them."""
    naive_theta, naive_kappa, _, _ = oracle_dc3_naive_contamination()
    fit = oracle_dc2_recovery()
    return {
        "theta_1_0": (THETA0, naive_theta, fit["theta_1_0_hat"]),
        "kappa_1j": (KAPPA_1J, naive_kappa, fit["kappa_1j_hat"]),
    }


# ---------------------------------------------------------------------------
# Figure 17
# ---------------------------------------------------------------------------
def make_figure():
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    view = _hardcoded_recovery_view()
    labels = [r"$\theta_1^{(0)}$", r"$\kappa_{1j}$"]
    true_v = [view["theta_1_0"][0], view["kappa_1j"][0]]
    naive_v = [view["theta_1_0"][1], view["kappa_1j"][1]]
    fit_v = [view["theta_1_0"][2], view["kappa_1j"][2]]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.bar(x - w, true_v, w, color=C["grey"], label="true value")
    ax.bar(x, naive_v, w, color=C["vermil"], label="naive 2-label (s_1, c_j only)")
    ax.bar(x + w, fit_v, w, color=C["blue"], label="3-label (s_1, c_j, p_j)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("recovered value (rad)")
    ax.set_title("Fig 17 — Padding (p_j) removes the drive-leakage contamination")
    ax.legend(fontsize=9.5, loc="upper left")
    save(fig, "fig17_drive_contamination")
    plt.close(fig)


def main():
    print(f"Combined parameters: theta_1_0={THETA0}, kappa_1j={KAPPA_1J}, phi_1j={PHI_1J}")
    Theta, K, Phi = combined_params()
    print(f"  -> Theta={Theta}, K={K}, Phi={Phi}\n")

    print("Oracle DC1 -- exact correlations vs brute-force averaging:")
    (X, Y_c, Y_p), closed = oracle_dc1_correlations()
    labels = ["X", "Y_c", "Y_p"]
    ref = [0.129505, 0.039644, -0.019814]
    max_err_dc1 = 0.0
    for name, bf, cl, rv in zip(labels, (X, Y_c, Y_p), closed, ref):
        err_form = abs(bf - cl)
        err_ref = abs(bf - rv)
        max_err_dc1 = max(max_err_dc1, err_form)
        print(f"  {name}: brute-force={bf:+.6f}  closed-form={cl:+.6f}  "
              f"spec-reference={rv:+.6f}  |diff|={err_form:.2e}")
    if max_err_dc1 > 1e-12:
        raise SystemExit(f"Oracle DC1 FAILED: max |brute-force - closed-form| "
                         f"{max_err_dc1:.2e} > 1e-12")
    print(f"  Oracle DC1 PASSED (max diff {max_err_dc1:.2e}).\n")

    print("Oracle DC2 -- stratified recovery from population-exact records:")
    fit = oracle_dc2_recovery()
    err_theta = abs(fit["theta_1_0_hat"] - THETA0)
    err_kappa = abs(fit["kappa_1j_hat"] - KAPPA_1J)
    err_phi = abs(fit["phi_1j_hat"] - PHI_1J)
    print(f"  theta_1_0_hat={fit['theta_1_0_hat']:.9f}  (true {THETA0}, |err|={err_theta:.2e})")
    print(f"  kappa_1j_hat ={fit['kappa_1j_hat']:.9f}  (true {KAPPA_1J}, |err|={err_kappa:.2e})")
    print(f"  phi_1j_hat   ={fit['phi_1j_hat']:.9f}  (true {PHI_1J}, |err|={err_phi:.2e})")
    print(f"  K_disagreement={fit['K_disagreement']:.2e}")
    if max(err_theta, err_kappa, err_phi) > 1e-9:
        raise SystemExit("Oracle DC2 FAILED: recovery did not match the true inputs "
                         "to <1e-9 in the noiseless limit -- do not proceed further.")
    print("  Oracle DC2 PASSED.\n")

    print("Oracle DC3 -- naive 2-label (s_1, c_j) contamination, p_j unvaried:")
    naive_theta, naive_kappa, Theta, K = oracle_dc3_naive_contamination()
    bias_theta_pct = 100.0 * abs(naive_theta - THETA0) / THETA0
    bias_kappa_pct = 100.0 * abs(naive_kappa - KAPPA_1J) / KAPPA_1J
    print(f"  naive theta_1_0_hat={naive_theta:.6f}  (true theta_1_0={THETA0}, "
          f"lands near Theta={Theta:.6f}) -- bias {bias_theta_pct:.1f}%")
    print(f"  naive kappa_1j_hat ={naive_kappa:.6f}  (true kappa_1j={KAPPA_1J}, "
          f"lands near K={K:.6f}) -- bias {bias_kappa_pct:.1f}%")
    if abs(naive_theta - Theta) > 1e-4 or abs(naive_kappa - K) > 1e-4:
        raise SystemExit("Oracle DC3 FAILED: naive estimate did not land near (Theta, K) "
                         "-- the contamination model (Sec 1) may be wrong.")
    if bias_theta_pct < 10.0 or bias_kappa_pct < 10.0:
        raise SystemExit("Oracle DC3 FAILED: bias too small to be a convincing "
                         "demonstration -- the whole exercise this document justifies "
                         "would not be motivated.")
    print(f"  Oracle DC3 PASSED: clear, reproducible bias confirmed "
          f"({bias_theta_pct:.0f}% theta, {bias_kappa_pct:.0f}% kappa).")
    print("\nAll oracles DC1-DC3 PASSED.\nGenerating Figure 17 ...")
    make_figure()
    print("Figure 17 done.")


if __name__ == "__main__":
    main()
