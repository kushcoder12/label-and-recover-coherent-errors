"""Phase 3 verification (CLAUDE.md §5, Oracle O1).

Reproduce the single-qubit conservation decomposition to 1e-6, both from the
numeric simulation and the analytic model, and confirm the conservation law
F_marg + E_b[F_{s|b}] = F_0 = 1 at several (theta, mu).

Runnable directly or via pytest. Seed-independent (exact enumeration/analytics).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from exp1_conservation import (numeric_decomposition, analytic_decomposition,
                               verify_oracle_O1)


def test_oracle_O1():
    max_err = verify_oracle_O1(verbose=False)
    assert max_err < 1e-6, f"Oracle O1 max error {max_err:.2e} exceeds 1e-6"


def test_numeric_matches_analytic():
    for theta in [0.05, 0.2, 0.5, 1.0]:
        for mu in [0.0, 0.3, 0.5, 0.8, 1.0]:
            p = (1 + mu) / 2
            num = numeric_decomposition(theta, p)
            ana = analytic_decomposition(theta, mu)
            assert abs(num["F_marg"] - ana["F_marg"]) < 1e-6
            assert abs(num["E_b"] - ana["E_b"]) < 1e-6
            assert abs(num["F0"] - 1.0) < 1e-6
            assert abs(num["F_joint"] - 1.0) < 1e-6


def test_conservation_law():
    for theta in [0.05, 0.2, 0.5, 1.0, 1.5]:
        for mu in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = (1 + mu) / 2
            num = numeric_decomposition(theta, p)
            assert abs((num["F_marg"] + num["E_b"]) - 1.0) < 1e-6, \
                f"conservation broken at theta={theta}, mu={mu}"


def run_all():
    for t in [test_oracle_O1, test_numeric_matches_analytic, test_conservation_law]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_conservation: ALL PASS")


if __name__ == "__main__":
    run_all()
