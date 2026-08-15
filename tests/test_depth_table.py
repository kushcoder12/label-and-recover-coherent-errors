"""Oracles O2 & O3 verification.

Reproduce the depth tables:
  - Oracle O2 (alpha = 1.0 rad) to 1e-6,
  - Oracle O3 (alpha = pi/2, integers) exactly,
and confirm labeled analytic == enumeration and Method A == Method B.

Runnable directly or via pytest. Seed-independent for the oracle checks.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from exp2_depth_scaling import (verify_oracle_O2, verify_oracle_O3,
                                verify_methods_agree, labeled_EFQ,
                                labeled_EFQ_enumerate, coherent_FQ,
                                labeled_analytic, coherent_analytic)


def test_oracle_O2():
    assert verify_oracle_O2(verbose=False) < 1e-6


def test_oracle_O3():
    assert verify_oracle_O3(verbose=False) < 1e-6


def test_labeled_matches_analytic_and_enumeration():
    for alpha in [0.7, 1.0, np.pi / 2, 2.3]:
        for L in range(1, 9):
            lab = labeled_EFQ(L, alpha)
            assert abs(lab - labeled_analytic(L, alpha)) < 1e-9
            assert abs(lab - labeled_EFQ_enumerate(L, alpha)) < 1e-9
            assert abs(coherent_FQ(L, alpha) - coherent_analytic(L, alpha)) < 1e-9


def test_methods_agree():
    assert verify_methods_agree(verbose=False) < 1e-3


def run_all():
    for t in [test_oracle_O2, test_oracle_O3,
              test_labeled_matches_analytic_and_enumeration, test_methods_agree]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_depth_table: ALL PASS")


if __name__ == "__main__":
    run_all()
