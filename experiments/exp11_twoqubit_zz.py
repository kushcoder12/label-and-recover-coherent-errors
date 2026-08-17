"""Experiment 11 / Figure 12 — Two-qubit static ZZ crosstalk: labeled recovery.

Extends the single-qubit sign-flip theorem (Fig 1-3) to a DIFFERENT, more tractable
mechanism than a neighbor's real drive: static ZZ coupling. Qubit 2 is deliberately
prepared each shot in a known, independently randomized |0>/|1> (label c_2), alongside
qubit 1's own RC sign s_1 -- a second owned label, not a passive read of an unmodified
circuit. core.circuits.two_qubit_zz_probe evolves the FULL two-qubit Hamiltonian and
reproduces <Y1> = s_1 sin(theta0 + kappa_12 c_2) exactly (checked against the closed
form in tests/test_multiqubit_zz.py, not assumed here); this experiment verifies the
INVERSE problem, recovering (theta0, kappa_12) from finite-shot labeled data:

  (a) noiseless, across a kappa_12 sweep, against the exact closed-form Oracle MQ1;
  (b) under the full device noise profile, debiased by the visibility v measured from
      a theta0=pi/2, kappa=0 calibration probe -- a direct, mechanical reuse of the
      single-qubit noise-aware approach (core.estimator), not new physics.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import two_qubit_zz_probe
from core.multiqubit_zz import joint_correlation_estimator
from core.noise import NoiseProfile

THETA0 = 0.10
KAPPA_SWEEP = [0.0, 0.02, 0.05, 0.10, 0.15]
SHOTS = 20000
SEED = 20260816


def device_profile():
    """Representative single-qubit-tier device profile (reuses core.noise as-is)."""
    return NoiseProfile(dt=1.0, T1=60.0, T2=40.0, r_1q=0.004, e01=0.01, e10=0.02)


def _outcome_table(theta0, kappa12, noise):
    """Exact <Y1> for all 4 (s_1, c_2) combinations -- the model is deterministic
    given the label pair, so per-shot outcomes are Bernoulli draws around this table
    rather than re-evolving the Hamiltonian per shot."""
    return {(s1, c2): two_qubit_zz_probe(theta0, kappa12, s1, c2, noise=noise)
            for s1 in (1, -1) for c2 in (1, -1)}


def sample_shots(theta0, kappa12, shots, noise=None, v=1.0, rng=None):
    """Draw `shots` owned-label records (s_1, {2: c_2}, b_1) from the exact model."""
    rng = np.random.default_rng() if rng is None else rng
    table = _outcome_table(theta0, kappa12, noise)
    s = np.where(rng.random(shots) < 0.5, 1, -1)
    c2 = np.where(rng.random(shots) < 0.5, 1, -1)
    records = []
    for si, ci in zip(s, c2):
        y = table[(int(si), int(ci))] / v
        p_plus = np.clip((1.0 + y) / 2.0, 0.0, 1.0)
        b = 1 if rng.random() < p_plus else -1
        records.append((int(si), {2: int(ci)}, b))
    return records


def exact_recovery_check(theta0, kappa12, noise=None):
    """Oracle MQ1-style check: population-exact (infinite-shot) records recover
    (theta0, kappa12) to floating-point precision -- the model identity, not a
    statistical estimate."""
    records = [(s1, {2: c2}, two_qubit_zz_probe(theta0, kappa12, s1, c2, noise=noise))
               for s1 in (1, -1) for c2 in (1, -1)]
    fit = joint_correlation_estimator(records, 1, [2], B=2)
    return abs(fit["theta0_hat"] - theta0), abs(fit["kappa_hat"][2] - kappa12)


def run_noiseless_sweep(shots=SHOTS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for kappa12 in KAPPA_SWEEP:
        err_theta, err_kappa = exact_recovery_check(THETA0, kappa12)
        recs = sample_shots(THETA0, kappa12, shots, rng=rng)
        fit = joint_correlation_estimator(recs, 1, [2], B=600, seed=seed + int(kappa12 * 1000))
        rows.append({"kappa_true": kappa12, "exact_err_theta": err_theta,
                     "exact_err_kappa": err_kappa, **fit})
    return rows


def run_noisy_sweep(shots=SHOTS, seed=SEED + 1):
    profile = device_profile()
    v = two_qubit_zz_probe(np.pi / 2, 0.0, 1, 1, noise=profile)
    rng = np.random.default_rng(seed)
    rows = []
    for kappa12 in KAPPA_SWEEP:
        recs = sample_shots(THETA0, kappa12, shots, noise=profile, v=v, rng=rng)
        fit = joint_correlation_estimator(recs, 1, [2], B=600, seed=seed + int(kappa12 * 1000))
        rows.append({"kappa_true": kappa12, "v": v, **fit})
    return rows, v


# ---------------------------------------------------------------------------
# Figure 12
# ---------------------------------------------------------------------------
def make_figure(noiseless_rows, noisy_rows):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    lo, hi = 0.0, max(KAPPA_SWEEP) * 1.1

    for ax, rows, title in [(axA, noiseless_rows, "(a) noiseless"),
                             (axB, noisy_rows, "(b) full device noise, debiased")]:
        ax.plot([lo, hi], [lo, hi], "--", color=C["black"], lw=1.2, zorder=1,
                label=r"ideal recovery ($\hat\kappa_{12}=\kappa_{12}$)")
        xs = [r["kappa_true"] for r in rows]
        ys = [r["kappa_hat"][2] for r in rows]
        lo_ci = [r["kappa_hat"][2] - r["bootstrap_ci"]["kappa"][2][0] for r in rows]
        hi_ci = [r["bootstrap_ci"]["kappa"][2][1] - r["kappa_hat"][2] for r in rows]
        ax.errorbar(xs, ys, yerr=[lo_ci, hi_ci], fmt="o", color=C["blue"], ms=6,
                    mec="k", mew=0.5, capsize=3, lw=1.3, label=r"recovered $\hat\kappa_{12}$")
        ax.set_xlabel(r"true ZZ coupling  $\kappa_{12}$  (rad)")
        ax.set_title(title)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=9, loc="upper left")
    axA.set_ylabel(r"recovered  $\hat\kappa_{12}$  (rad)")
    fig.suptitle(r"Fig 12 — Two-qubit static ZZ crosstalk recovery ($\theta_1^{(0)}=0.10$)",
                fontsize=12.5)
    save(fig, "fig12_twoqubit_zz")
    plt.close(fig)


def main():
    print(f"Oracle MQ1 sweep (theta0={THETA0}), noiseless, exact (infinite-shot) records:")
    max_exact_err = 0.0
    for kappa12 in KAPPA_SWEEP:
        et, ek = exact_recovery_check(THETA0, kappa12)
        max_exact_err = max(max_exact_err, et, ek)
        print(f"  kappa_12={kappa12:.3f}: |err theta0|={et:.2e}  |err kappa12|={ek:.2e}")
    if max_exact_err > 1e-9:
        raise SystemExit(f"Oracle MQ1 FAILED: max exact error {max_exact_err:.2e} > 1e-9")
    print(f"  max exact error = {max_exact_err:.2e}  (Oracle MQ1 PASSED)")

    print(f"\nFinite-shot recovery ({SHOTS} shots/point), noiseless:")
    noiseless_rows = run_noiseless_sweep()
    max_shot_err = 0.0
    for r in noiseless_rows:
        err = abs(r["kappa_hat"][2] - r["kappa_true"])
        max_shot_err = max(max_shot_err, err)
        lo, hi = r["bootstrap_ci"]["kappa"][2]
        print(f"  kappa_12={r['kappa_true']:.3f}: hat={r['kappa_hat'][2]:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  |err|={err:.4f}")
    if max_shot_err > 0.02:
        raise SystemExit(f"finite-shot recovery FAILED: max|err|={max_shot_err:.4f} > 0.02")

    print(f"\nFinite-shot recovery ({SHOTS} shots/point), full device noise + visibility debiasing:")
    noisy_rows, v = run_noisy_sweep()
    print(f"  visibility v = {v:.4f}")
    max_noisy_err = 0.0
    for r in noisy_rows:
        err = abs(r["kappa_hat"][2] - r["kappa_true"])
        max_noisy_err = max(max_noisy_err, err)
        lo, hi = r["bootstrap_ci"]["kappa"][2]
        print(f"  kappa_12={r['kappa_true']:.3f}: hat={r['kappa_hat'][2]:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  |err|={err:.4f}")
    if max_noisy_err > 0.02:
        raise SystemExit(f"noisy recovery FAILED: max|err|={max_noisy_err:.4f} > 0.02")

    print("\nAll recovery checks PASSED.\nGenerating Figure 12 ...")
    make_figure(noiseless_rows, noisy_rows)
    print("Figure 12 done.")


if __name__ == "__main__":
    main()
