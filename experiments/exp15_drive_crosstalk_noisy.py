"""Experiment 15 / Figure 16 — Drive-crosstalk recovery under the full device noise
profile, and the noiseless reference sweep it is checked against.

Repeats Oracle DC2's recovery, sweeping phi_1j over {0.0, 0.01, 0.02, 0.05, 0.10} at
fixed theta_1_0=0.10, kappa_1j=0.05:

  (a) noiseless, finite shots, against the exact closed-form model;
  (b) under core.noise.NoiseProfile, debiased by the visibility v measured from a
      theta_1_0=pi/2, kappa_1j=0, phi_1j=0 calibration probe -- the same mechanical
      reuse of the single-qubit visibility-debiasing idea exp11 already applies to
      the two-label ZZ recovery, extended unchanged to the three-label case.

A single incoherent noise layer scales <Y_1> by a label- and angle-independent scalar
(the same fact exp11 verifies for two_qubit_zz_probe); this is re-checked here for
three_label_probe before it is relied on in the sweep.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import three_label_probe
from core.multiqubit_zz import stratified_three_label_estimator
from core.noise import NoiseProfile

import exp14_drive_crosstalk_oracle as exp14

THETA0 = exp14.THETA0
KAPPA_1J = exp14.KAPPA_1J
PHI_SWEEP = [0.0, 0.01, 0.02, 0.05, 0.10]
SHOTS = 20000
SEED = 20260817

sample_shots = exp14.sample_shots


def device_profile():
    """Representative single-qubit-tier device profile (reuses core.noise as-is,
    same parameters exp11 uses for the two-label ZZ recovery)."""
    return NoiseProfile(dt=1.0, T1=60.0, T2=40.0, r_1q=0.004, e01=0.01, e10=0.02)


def measure_visibility(profile):
    """v = <Y_1> at theta_1_0=pi/2, kappa_1j=phi_1j=0, s_1=c_j=p_j=+1: sin(pi/2)=1
    exactly noiseless, so the noisy probe's value IS the visibility."""
    return three_label_probe(np.pi / 2, 0.0, 0.0, 1, 1, 1, noise=profile)


def check_visibility_is_label_and_angle_independent(profile):
    """One incoherent layer applied after the coherent evolution must scale <Y_1> by
    a pure scalar v, independent of theta_1_0/kappa_1j/phi_1j and of every label --
    verified numerically before the debiasing in run_noisy_sweep is trusted."""
    v_ref = measure_visibility(profile)
    max_err = 0.0
    for theta0 in (0.05, 0.10, 0.20):
        for kappa in (0.0, 0.05, -0.03):
            for phi in (0.0, 0.02, 0.08):
                for s1 in (1, -1):
                    for c in (1, -1):
                        for p in (1, -1):
                            noiseless = three_label_probe(theta0, kappa, phi, s1, c, p)
                            noisy = three_label_probe(theta0, kappa, phi, s1, c, p, noise=profile)
                            max_err = max(max_err, abs(noisy - v_ref * noiseless))
    return v_ref, max_err


def run_noiseless_sweep(shots=SHOTS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for phi in PHI_SWEEP:
        recs = sample_shots(THETA0, KAPPA_1J, phi, shots, rng=rng)
        fit = stratified_three_label_estimator(recs, B=600, seed=seed + int(phi * 1000))
        rows.append({"phi_true": phi, **fit})
    return rows


def run_noisy_sweep(shots=SHOTS, seed=SEED + 1):
    profile = device_profile()
    v = measure_visibility(profile)
    rng = np.random.default_rng(seed)
    rows = []
    for phi in PHI_SWEEP:
        recs = sample_shots(THETA0, KAPPA_1J, phi, shots, noise=profile, v=v, rng=rng)
        fit = stratified_three_label_estimator(recs, B=600, seed=seed + int(phi * 1000))
        rows.append({"phi_true": phi, "v": v, **fit})
    return rows, v


# ---------------------------------------------------------------------------
# Figure 16
# ---------------------------------------------------------------------------
def make_figure(noiseless_rows, noisy_rows):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    lo, hi = 0.0, max(PHI_SWEEP) * 1.15

    for ax, rows, title in [(axA, noiseless_rows, "(a) noiseless"),
                            (axB, noisy_rows, "(b) full device noise, debiased")]:
        ax.plot([lo, hi], [lo, hi], "--", color=C["black"], lw=1.2, zorder=1,
                label=r"ideal recovery ($\hat\phi_{1j}=\phi_{1j}$)")
        xs = [r["phi_true"] for r in rows]
        ys = [r["phi_1j_hat"] for r in rows]
        lo_ci = [r["phi_1j_hat"] - r["bootstrap_ci"]["phi_1j"][0] for r in rows]
        hi_ci = [r["bootstrap_ci"]["phi_1j"][1] - r["phi_1j_hat"] for r in rows]
        ax.errorbar(xs, ys, yerr=[lo_ci, hi_ci], fmt="o", color=C["blue"], ms=6,
                    mec="k", mew=0.5, capsize=3, lw=1.3, label=r"recovered $\hat\phi_{1j}$")
        theta_ys = [r["theta_1_0_hat"] for r in rows]
        ax.plot(xs, theta_ys, "^", color=C["orange"], ms=6, mec="k", mew=0.5,
               label=r"recovered $\hat\theta_1^{(0)}$ (true " f"{THETA0}" r", should stay flat)")
        ax.set_xlabel(r"true drive-leakage angle  $\phi_{1j}$  (rad)")
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo - 0.03, hi + 0.04)
        ax.legend(fontsize=8.5, loc="upper left")
    axA.set_ylabel(r"recovered value  (rad)")
    fig.suptitle(r"Fig 16 — Drive-crosstalk recovery ($\theta_1^{(0)}=0.10$, "
                r"$\kappa_{1j}=0.05$ held fixed)", fontsize=12.5)
    save(fig, "fig16_drive_crosstalk_sweep")
    plt.close(fig)


def main():
    profile = device_profile()
    print("Checking the noise layer scales <Y_1> by a label/angle-independent visibility:")
    v_ref, max_err = check_visibility_is_label_and_angle_independent(profile)
    print(f"  v = {v_ref:.6f}   max |noisy - v*noiseless| = {max_err:.2e}")
    if max_err > 1e-9:
        raise SystemExit(f"visibility-scalar check FAILED: max err {max_err:.2e} > 1e-9 "
                         f"-- debiasing below would be unjustified.")
    print("  PASSED.\n")

    print(f"Noiseless recovery, phi_1j sweep {PHI_SWEEP} ({SHOTS} shots/point):")
    noiseless_rows = run_noiseless_sweep()
    max_err_noiseless = 0.0
    for r in noiseless_rows:
        et = abs(r["theta_1_0_hat"] - THETA0)
        ek = abs(r["kappa_1j_hat"] - KAPPA_1J)
        ep = abs(r["phi_1j_hat"] - r["phi_true"])
        max_err_noiseless = max(max_err_noiseless, et, ek, ep)
        print(f"  phi_1j={r['phi_true']:.3f}: theta_1_0_hat={r['theta_1_0_hat']:+.4f} "
              f"kappa_1j_hat={r['kappa_1j_hat']:+.4f} phi_1j_hat={r['phi_1j_hat']:+.4f} "
              f"K_disagreement={r['K_disagreement']:.2e}")
    if max_err_noiseless > 0.02:
        raise SystemExit(f"noiseless sweep FAILED: max|err|={max_err_noiseless:.4f} > 0.02")
    print(f"  max|err|={max_err_noiseless:.4f}  PASSED.\n")

    print(f"Noisy recovery (full device profile, visibility-debiased), same sweep:")
    noisy_rows, v = run_noisy_sweep()
    print(f"  visibility v = {v:.4f}")
    max_err_noisy = 0.0
    for r in noisy_rows:
        et = abs(r["theta_1_0_hat"] - THETA0)
        ek = abs(r["kappa_1j_hat"] - KAPPA_1J)
        ep = abs(r["phi_1j_hat"] - r["phi_true"])
        max_err_noisy = max(max_err_noisy, et, ek, ep)
        print(f"  phi_1j={r['phi_true']:.3f}: theta_1_0_hat={r['theta_1_0_hat']:+.4f} "
              f"kappa_1j_hat={r['kappa_1j_hat']:+.4f} phi_1j_hat={r['phi_1j_hat']:+.4f} "
              f"K_disagreement={r['K_disagreement']:.2e}")
    if max_err_noisy > 0.025:
        raise SystemExit(f"noisy sweep FAILED: max|err|={max_err_noisy:.4f} > 0.025")
    print(f"  max|err|={max_err_noisy:.4f}  PASSED.\n")

    print("Generating Figure 16 ...")
    make_figure(noiseless_rows, noisy_rows)
    print("Figure 16 done.")


if __name__ == "__main__":
    main()
