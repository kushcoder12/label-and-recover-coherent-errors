"""Estimator + recover-and-correct payoff verification.

- Labeled MLE saturates the Cramer-Rao bound 1/(M F0) at the quantum limit F0=1.
- Closed-form MLE agrees with the scipy-free golden-section maximizer.
- Marginal estimate is unidentifiable at full twirl (mu=0).
- Recover-and-correct payoff: labeled RC + correction beats plain RC in the
  meaningful regime (theta* >= 0.1) and is never meaningfully worse.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.estimator import (labeled_mle, labeled_mle_golden, marginal_estimate,
                            sample_labeled_shots, estimator_variance_vs_crb)
from exp5_recover_and_correct import payoff_curves


def test_crb_saturation():
    rng = np.random.default_rng(3)
    for theta in [0.2, 0.5]:
        var, crb, bias = estimator_variance_vs_crb(theta, 2000, F0=1.0,
                                                   trials=600, rng=rng)
        assert 0.8 < var / crb < 1.25, f"MLE does not saturate CRB: {var/crb:.3f}"
        assert abs(bias) < 0.01


def test_closed_form_matches_golden():
    rng = np.random.default_rng(5)
    shots = sample_labeled_shots(0.4, 4000, rng=rng)
    assert abs(labeled_mle(shots) - labeled_mle_golden(shots)) < 1e-6


def test_marginal_unidentifiable_at_full_twirl():
    rng = np.random.default_rng(7)
    shots = sample_labeled_shots(0.4, 2000, rng=rng)
    est, identifiable = marginal_estimate(shots, mu=0.0)
    assert not identifiable and np.isnan(est)


def test_payoff_beats_plain_rc():
    data = payoff_curves(alpha=0.7, L=6, thetas=np.linspace(0.0, 0.7, 9),
                         M=2000, trials=80, seed=1)
    diff = data["corrected"] - data["plainRC"]
    meaningful = data["theta"] >= 0.1
    assert np.min(diff[meaningful]) > 0, "correction fails to beat plain RC where it matters"
    assert np.min(diff) > -3e-3, "correction is meaningfully worse somewhere"
    assert np.max(diff) > 0.05, "no substantial payoff observed"


def run_all():
    for t in [test_crb_saturation, test_closed_form_matches_golden,
              test_marginal_unidentifiable_at_full_twirl, test_payoff_beats_plain_rc]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_estimator: ALL PASS")


if __name__ == "__main__":
    run_all()
