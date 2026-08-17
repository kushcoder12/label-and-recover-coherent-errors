"""Experiment 12 / Figure 13 — Three-qubit star: joint multi-neighbor ZZ recovery.

Qubit 1 is coupled to TWO neighbors (qubits 2, 3; star topology, no 2-3 coupling).
Both neighbor labels c_2, c_3 are drawn independently every shot, alongside qubit 1's
own RC sign s_1 -- THREE owned labels per shot. core.circuits.multi_qubit_zz_probe
evolves the full 3-qubit Hamiltonian; this experiment checks three things the spec
calls out explicitly, none of which may be assumed:

  (a) Oracle MQ2 -- the exact closed forms for X=E[s1 b1], Y2=E[s1 c2 b1],
      Y3=E[s1 c3 b1] match what a direct Monte Carlo simulation of the circuit
      converges to as shots -> infinity;
  (b) Oracle MQ3 -- the Fisher matrix of the design vector (s1, s1 c2, s1 c3) is
      exactly diagonal, verified from the EMPIRICAL covariance of the actual sampled
      labels (catches a labeling bug that the formula alone would not);
  (c) theta_1^(0), kappa_12, AND kappa_13 are recovered jointly from ONE shot pool.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import multi_qubit_zz_probe
from core.multiqubit_zz import joint_correlation_estimator

THETA0 = 0.10
KAPPA = {2: 0.05, 3: 0.03}
SWEEP = [(0.02, 0.012), (0.05, 0.030), (0.08, 0.048), (0.12, 0.072)]  # (kappa12, kappa13)
SHOTS = 40000
SEED = 20260817


def _outcome_table(theta0, kappa):
    return {(s1, c2, c3): multi_qubit_zz_probe(theta0, kappa, s1, {2: c2, 3: c3})
            for s1 in (1, -1) for c2 in (1, -1) for c3 in (1, -1)}


def sample_shots(theta0, kappa, shots, rng):
    """Draw `shots` owned-label records (s_1, {2:c_2, 3:c_3}, b_1)."""
    table = _outcome_table(theta0, kappa)
    s1 = np.where(rng.random(shots) < 0.5, 1, -1)
    c2 = np.where(rng.random(shots) < 0.5, 1, -1)
    c3 = np.where(rng.random(shots) < 0.5, 1, -1)
    u = rng.random(shots)
    records = []
    for a, b2, b3, uu in zip(s1, c2, c3, u):
        y = table[(int(a), int(b2), int(b3))]
        p_plus = min(max((1.0 + y) / 2.0, 0.0), 1.0)
        bit = 1 if uu < p_plus else -1
        records.append((int(a), {2: int(b2), 3: int(b3)}, bit))
    return records


def oracle_mq2_targets(theta0, kappa):
    """Exact closed forms (Sec 2.3), evaluated with math.sin/cos, not transcribed."""
    k2, k3 = kappa[2], kappa[3]
    X = math.sin(theta0) * math.cos(k2) * math.cos(k3)
    Y2 = math.cos(theta0) * math.sin(k2) * math.cos(k3)
    Y3 = math.cos(theta0) * math.cos(k2) * math.sin(k3)
    return X, Y2, Y3


def empirical_XY(records):
    s1 = np.array([r[0] for r in records], dtype=float)
    c2 = np.array([r[1][2] for r in records], dtype=float)
    c3 = np.array([r[1][3] for r in records], dtype=float)
    b1 = np.array([r[2] for r in records], dtype=float)
    X = float(np.mean(s1 * b1))
    Y2 = float(np.mean(s1 * c2 * b1))
    Y3 = float(np.mean(s1 * c3 * b1))
    return X, Y2, Y3


def empirical_fisher(records):
    """Fisher matrix of the design vector d=(s1, s1*c2, s1*c3), Sec 2.5, from the
    ACTUAL sampled labels (not the theoretical E[d_a d_b]=delta_ab identity)."""
    s1 = np.array([r[0] for r in records], dtype=float)
    c2 = np.array([r[1][2] for r in records], dtype=float)
    c3 = np.array([r[1][3] for r in records], dtype=float)
    d = np.stack([s1, s1 * c2, s1 * c3], axis=1)
    return (d.T @ d) / len(records)


def run_oracle_checks(shots=SHOTS, seed=SEED):
    rng = np.random.default_rng(seed)
    X_exact, Y2_exact, Y3_exact = oracle_mq2_targets(THETA0, KAPPA)
    records = sample_shots(THETA0, KAPPA, shots, rng)
    X_mc, Y2_mc, Y3_mc = empirical_XY(records)
    tol = 6.0 / math.sqrt(shots)   # ~6-sigma band on a +/-1-valued mean at this M
    errs = {"X": abs(X_mc - X_exact), "Y2": abs(Y2_mc - Y2_exact), "Y3": abs(Y3_mc - Y3_exact)}
    F = empirical_fisher(records)
    fisher_diag_err = float(np.max(np.abs(np.diag(F) - 1.0)))
    fisher_offdiag_err = float(np.max(np.abs(F - np.diag(np.diag(F)))))
    return records, errs, tol, F, fisher_diag_err, fisher_offdiag_err


def run_joint_recovery(shots=SHOTS, seed=SEED + 1):
    """Sweep (kappa12, kappa13) pairs; ONE shot pool per point recovers both jointly."""
    rng = np.random.default_rng(seed)
    rows = []
    for k12, k13 in SWEEP:
        kappa = {2: k12, 3: k13}
        records = sample_shots(THETA0, kappa, shots, rng)
        fit = joint_correlation_estimator(records, 1, [2, 3], B=600,
                                          seed=seed + int(1000 * k12))
        rows.append({"kappa12_true": k12, "kappa13_true": k13, **fit})
    return rows


# ---------------------------------------------------------------------------
# Figure 13
# ---------------------------------------------------------------------------
def make_figure(rows):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    lo, hi = 0.0, max(r["kappa12_true"] for r in rows) * 1.15
    ax.plot([lo, hi], [lo, hi], "--", color=C["black"], lw=1.2, zorder=1, label="ideal $y=x$")

    xs12 = [r["kappa12_true"] for r in rows]
    ys12 = [r["kappa_hat"][2] for r in rows]
    e12lo = [r["kappa_hat"][2] - r["bootstrap_ci"]["kappa"][2][0] for r in rows]
    e12hi = [r["bootstrap_ci"]["kappa"][2][1] - r["kappa_hat"][2] for r in rows]
    ax.errorbar(xs12, ys12, yerr=[e12lo, e12hi], fmt="o", color=C["blue"], ms=6,
                mec="k", mew=0.5, capsize=3, lw=1.3, label=r"$\hat\kappa_{12}$ (neighbor 2)")

    xs13 = [r["kappa13_true"] for r in rows]
    ys13 = [r["kappa_hat"][3] for r in rows]
    e13lo = [r["kappa_hat"][3] - r["bootstrap_ci"]["kappa"][3][0] for r in rows]
    e13hi = [r["bootstrap_ci"]["kappa"][3][1] - r["kappa_hat"][3] for r in rows]
    ax.errorbar(xs13, ys13, yerr=[e13lo, e13hi], fmt="s", color=C["vermil"], ms=6,
                mec="k", mew=0.5, capsize=3, lw=1.3, label=r"$\hat\kappa_{13}$ (neighbor 3)")

    ax.set_xlabel(r"true ZZ coupling  $\kappa_{1j}$  (rad)")
    ax.set_ylabel(r"recovered  $\hat\kappa_{1j}$  (rad)")
    ax.set_title(r"Fig 13 — Joint recovery of two neighbors from one shot pool"
                "\n" r"($\theta_1^{(0)}=0.10$, 3-qubit star)")
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "fig13_threequbit_zz")
    plt.close(fig)


def main():
    print("Oracle MQ2 (3-qubit exact closed forms vs Monte Carlo convergence):")
    records, errs, tol, F, fdiag, foff = run_oracle_checks()
    for name, err in errs.items():
        print(f"  {name}: |MC - exact| = {err:.4f}  (tol {tol:.4f})")
    if any(err > tol for err in errs.values()):
        raise SystemExit(f"Oracle MQ2 FAILED: {errs} vs tol {tol:.4f}")
    print("  Oracle MQ2 PASSED.")

    print("\nOracle MQ3 (empirical Fisher matrix of the sampled labels):")
    print(np.array2string(F, precision=4, suppress_small=True))
    print(f"  max|diag-1| = {fdiag:.4f}, max|offdiag| = {foff:.4f}  (tol {tol:.4f} each)")
    if fdiag > tol or foff > tol:
        raise SystemExit(f"Oracle MQ3 FAILED: diag err {fdiag:.4f}, offdiag err {foff:.4f}")
    print("  Oracle MQ3 PASSED (Fisher matrix diagonal to statistical precision).")

    print(f"\nJoint recovery of (kappa_12, kappa_13) from ONE shot pool ({SHOTS} shots/point):")
    rows = run_joint_recovery()
    max_err = 0.0
    for r in rows:
        e12 = abs(r["kappa_hat"][2] - r["kappa12_true"])
        e13 = abs(r["kappa_hat"][3] - r["kappa13_true"])
        max_err = max(max_err, e12, e13)
        print(f"  true=({r['kappa12_true']:.3f},{r['kappa13_true']:.3f})  "
              f"hat=({r['kappa_hat'][2]:+.4f},{r['kappa_hat'][3]:+.4f})  "
              f"theta0_hat={r['theta0_hat']:+.4f}  |err|=({e12:.4f},{e13:.4f})")
    if max_err > 0.02:
        raise SystemExit(f"joint recovery FAILED: max|err|={max_err:.4f} > 0.02")
    print("  joint recovery PASSED.")

    print("\nAll checks PASSED.\nGenerating Figure 13 ...")
    make_figure(rows)
    print("Figure 13 done.")


if __name__ == "__main__":
    main()
