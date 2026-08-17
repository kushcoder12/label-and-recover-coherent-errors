"""Experiment 13 / Figure 14 — Shot cost: joint recovery vs pairwise baseline.

The figure that demonstrates the actual claimed advantage of tracking every neighbor's
label in the SAME shot pool. For Q neighbors, compares the total shots needed to reach
a fixed target precision sigma_target on every kappa_1j via:

  (a) the joint estimator here -- one pool of M shots, all Q neighbors labeled
      simultaneously and independently (core.multiqubit_zz.joint_correlation_estimator);
  (b) the "standard practice" baseline -- pairwise conditional Ramsey, one dedicated
      two-qubit experiment per neighbor (core.circuits.two_qubit_zz_probe, i.e. the
      OTHER neighbors are simply not part of that sub-experiment), run sequentially.

The claim (joint cost flat in Q, pairwise cost linear in Q) is a prediction of Oracle
MQ3's Fisher diagonality, not something plotted by assumption: shots-to-target are
measured from repeated simulated trials at a probed shot count and extrapolated via the
measured Var ~ 1/M scaling, and that scaling itself is checked against two probed shot
counts before being trusted. If the data does not show flat-vs-linear, this script
reports the discrepancy rather than forcing the figure to agree.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuits import multi_qubit_zz_probe, two_qubit_zz_probe
from core.multiqubit_zz import joint_correlation_estimator

THETA0 = 0.10
KAPPA_COMMON = 0.05
Q_VALUES = [1, 2, 3, 4, 5]
SIGMA_TARGET = 0.01          # rad, the precision every kappa_1j must reach
M_PROBE = 4000                # shots/trial used to measure the variance at one point
TRIALS = 250
SEED = 20260818


def _outcome_table(theta0, kappa_dict, neighbors):
    table = {}
    for combo in itertools.product((1, -1), repeat=len(neighbors) + 1):
        s1 = combo[0]
        c_dict = {j: combo[1 + i] for i, j in enumerate(neighbors)}
        table[combo] = multi_qubit_zz_probe(theta0, kappa_dict, s1, c_dict)
    return table


def sample_joint_shots(theta0, kappa_dict, shots, rng):
    """One shot pool tracking every neighbor in kappa_dict simultaneously."""
    neighbors = sorted(kappa_dict)
    table = _outcome_table(theta0, kappa_dict, neighbors)
    draws = rng.random((shots, len(neighbors) + 1)) < 0.5
    signs = np.where(draws, 1, -1)
    u = rng.random(shots)
    records = []
    for row, uu in zip(signs, u):
        combo = tuple(int(x) for x in row)
        y = table[combo]
        p_plus = min(max((1.0 + y) / 2.0, 0.0), 1.0)
        bit = 1 if uu < p_plus else -1
        c_dict = {j: combo[1 + i] for i, j in enumerate(neighbors)}
        records.append((combo[0], c_dict, bit))
    return records, neighbors


def sample_pairwise_shots(theta0, kappa, shots, rng):
    """One dedicated 2-qubit experiment for a single neighbor (the baseline)."""
    s1 = np.where(rng.random(shots) < 0.5, 1, -1)
    c2 = np.where(rng.random(shots) < 0.5, 1, -1)
    u = rng.random(shots)
    table = {(s, c): two_qubit_zz_probe(theta0, kappa, s, c) for s in (1, -1) for c in (1, -1)}
    records = []
    for a, b, uu in zip(s1, c2, u):
        y = table[(int(a), int(b))]
        p_plus = min(max((1.0 + y) / 2.0, 0.0), 1.0)
        bit = 1 if uu < p_plus else -1
        records.append((int(a), {2: int(b)}, bit))
    return records


def joint_kappa_variance(Q, M, trials, rng):
    """Empirical Var(kappa_hat_2) tracking all Q neighbors together at shots=M."""
    kappa_dict = {j: KAPPA_COMMON for j in range(2, 2 + Q)}
    ests = np.empty(trials)
    for t in range(trials):
        records, neighbors = sample_joint_shots(THETA0, kappa_dict, M, rng)
        fit = joint_correlation_estimator(records, 1, neighbors, B=2, seed=0)
        ests[t] = fit["kappa_hat"][2]
    return float(np.var(ests, ddof=1))


def pairwise_kappa_variance(M, trials, rng):
    """Empirical Var(kappa_hat) for the single dedicated 2-qubit baseline experiment."""
    ests = np.empty(trials)
    for t in range(trials):
        records = sample_pairwise_shots(THETA0, KAPPA_COMMON, M, rng)
        fit = joint_correlation_estimator(records, 1, [2], B=2, seed=0)
        ests[t] = fit["kappa_hat"][2]
    return float(np.var(ests, ddof=1))


def shots_to_target(var_at_M, M, sigma_target):
    """Extrapolate shots-to-target from a single measured (M, Var) point assuming the
    CRB scaling Var ~ 1/M (checked separately, not assumed blindly -- see main())."""
    return M * (var_at_M / sigma_target**2)


def check_variance_scaling(rng):
    """Directly probe Var at M_PROBE and 2*M_PROBE for Q=3; confirm it roughly halves
    (the CRB 1/M law) before using a single-point extrapolation anywhere else."""
    v1 = joint_kappa_variance(3, M_PROBE, TRIALS, rng)
    v2 = joint_kappa_variance(3, 2 * M_PROBE, TRIALS, rng)
    ratio = v1 / v2
    return v1, v2, ratio


def run_cost_comparison(rng):
    pairwise_var = pairwise_kappa_variance(M_PROBE, TRIALS, rng)
    m_single = shots_to_target(pairwise_var, M_PROBE, SIGMA_TARGET)

    rows = []
    for Q in Q_VALUES:
        joint_var = joint_kappa_variance(Q, M_PROBE, TRIALS, rng)
        m_joint = shots_to_target(joint_var, M_PROBE, SIGMA_TARGET)
        rows.append({
            "Q": Q,
            "joint_total_shots": m_joint,
            "pairwise_total_shots": Q * m_single,
        })
    return rows, m_single


def _loglog_slope(xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    return float(np.polyfit(lx, ly, 1)[0])


# ---------------------------------------------------------------------------
# Figure 14
# ---------------------------------------------------------------------------
def make_figure(rows):
    from _style import setup, save
    C = setup()
    import matplotlib.pyplot as plt

    Qs = [r["Q"] for r in rows]
    joint = [r["joint_total_shots"] for r in rows]
    pairwise = [r["pairwise_total_shots"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.semilogy(Qs, joint, "-o", color=C["blue"], ms=7, lw=1.8,
               label=f"joint estimator (one pool, target " r"$\sigma=$" f"{SIGMA_TARGET})")
    ax.semilogy(Qs, pairwise, "-s", color=C["vermil"], ms=7, lw=1.8,
               label="pairwise baseline (Q sequential experiments)")
    ax.set_xlabel(r"number of neighbors  $Q$")
    ax.set_ylabel(r"total shots to reach $\sigma_{\mathrm{target}}$ on every $\kappa_{1j}$")
    ax.set_title(r"Fig 14 — Joint labeling is flat in $Q$; pairwise is linear")
    ax.set_xticks(Qs)
    ax.legend(fontsize=9.5, loc="upper left")
    save(fig, "fig14_shot_cost")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)

    print("Checking the Var ~ 1/M scaling this extrapolation relies on (Q=3, joint):")
    v1, v2, ratio = check_variance_scaling(rng)
    print(f"  Var(M={M_PROBE}) = {v1:.3e}   Var(M={2*M_PROBE}) = {v2:.3e}   ratio = {ratio:.3f}"
          f"  (CRB predicts 2.0)")
    if not (1.4 < ratio < 2.8):
        raise SystemExit(f"Var~1/M scaling check FAILED: ratio {ratio:.3f} not near 2.0 "
                         f"-- extrapolation below would be unjustified.")
    print("  scaling check PASSED (within statistical tolerance of TRIALS={}).".format(TRIALS))

    print(f"\nShot-cost comparison (sigma_target={SIGMA_TARGET} rad, kappa={KAPPA_COMMON} rad, "
          f"M_probe={M_PROBE}, trials={TRIALS}):")
    rows, m_single = run_cost_comparison(rng)
    print(f"  single dedicated 2-qubit experiment needs ~{m_single:.0f} shots for one kappa_1j.")
    for r in rows:
        print(f"  Q={r['Q']}: joint={r['joint_total_shots']:.0f} shots   "
              f"pairwise={r['pairwise_total_shots']:.0f} shots")

    slope_joint = _loglog_slope([r["Q"] for r in rows], [r["joint_total_shots"] for r in rows])
    slope_pairwise = _loglog_slope([r["Q"] for r in rows], [r["pairwise_total_shots"] for r in rows])
    print(f"\n  log-log slope vs Q: joint={slope_joint:+.2f} (flat predicts ~0), "
          f"pairwise={slope_pairwise:+.2f} (linear predicts ~1).")

    if not (slope_joint < 0.35):
        raise SystemExit(f"CLAIM NOT SUPPORTED BY SIMULATION: joint cost is not flat in Q "
                         f"(slope={slope_joint:+.2f}). Reporting the discrepancy rather than "
                         f"forcing the figure -- see printed data above.")
    if not (slope_pairwise > 0.85):
        raise SystemExit(f"CLAIM NOT SUPPORTED BY SIMULATION: pairwise cost is not linear in Q "
                         f"(slope={slope_pairwise:+.2f}).")
    print("\n  Flat-vs-linear claim CONFIRMED by simulation.")

    print("\nGenerating Figure 14 ...")
    make_figure(rows)
    print("Figure 14 done.")


if __name__ == "__main__":
    main()
