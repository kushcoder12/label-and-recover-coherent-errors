"""GAP 5 verification (GAPS_CODE.md): shot-cost positioning.

- labeled-RC MC variance matches the CRB 1/(M v^2 F0) (extends the Fig-6 check).
- Minimal RPE shows the Heisenberg ~1/d^2 improvement (d^2 * Var ~ const before the
  c^{2d} decoherence rolloff).
- Analytic crossover and optimal-depth helpers behave sensibly.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from exp10_shotcost import (labeled_rc_shots, labeled_rc_mc_variance, rpe_optimal_depth,
                            rpe_fisher_per_shot, rpe_depth_mc_variance, rpe_shots)


def test_labeled_rc_matches_crb():
    rng = np.random.default_rng(0)
    for v in [0.6, 0.8, 1.0]:
        M = 6000
        var = labeled_rc_mc_variance(0.3, v, M, trials=2500, rng=rng)
        crb = 1.0 / (M * v**2)
        assert 0.75 < var / crb < 1.3, f"v={v}: Var/CRB = {var/crb:.3f}"


def test_rpe_heisenberg_scaling():
    """d^2 * Var stays ~flat for small d (Var ~ 1/d^2) before c^{2d} rolls off."""
    rng = np.random.default_rng(1)
    c = 0.99
    scaled = []
    for d in [1, 2, 4, 8, 16]:
        var = rpe_depth_mc_variance(0.01, d, c, 8000, trials=2000, rng=rng)
        scaled.append(var * d**2)
    base = scaled[0]
    for s in scaled:
        assert 0.6 < s / base < 1.7, f"d^2*Var not flat: {[round(x/base,2) for x in scaled]}"


def test_optimal_depth_and_costs():
    c = 0.99
    dstar = rpe_optimal_depth(c)
    assert abs(dstar - (-1 / np.log(c))) < 1e-9
    # d* maximizes d^2 c^{2d}
    grid = np.arange(1, 400)
    F = grid**2 * c ** (2 * grid)
    assert abs(grid[np.argmax(F)] - dstar) <= 2
    # labeled-RC shots scale as 1/delta^2
    assert abs(labeled_rc_shots(0.01, 0.9) / labeled_rc_shots(0.02, 0.9) - 4.0) < 1e-6
    assert rpe_shots(0.001, c) > 0


def run_all():
    for t in [test_labeled_rc_matches_crb, test_rpe_heisenberg_scaling,
              test_optimal_depth_and_costs]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_shotcost: ALL PASS")


if __name__ == "__main__":
    run_all()
