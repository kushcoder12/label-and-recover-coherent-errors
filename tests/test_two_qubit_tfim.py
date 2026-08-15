"""Oracle O5 verification (two-qubit TFIM).

- Conservation chain-rule residual ~machine precision at every L (2-qubit TFIM).
- Labeled Sum(1-<G>^2) matches exact 2^L enumeration.
- O5 kernel: XX, YY, ZZ are exactly silent on |Phi+>; the other six are active.
- g->0 decoupling: chi is a product state and every ZZ error is silent.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from exp4_two_qubit_tfim import (conservation_split, labeled_EFQ_tfim,
                                 labeled_EFQ_enumerate, bell_kernel_demo,
                                 decoupling_sanity)


def test_conservation_residual():
    max_res = max(conservation_split(L, 0.3, 0.8, 0.15)[3] for L in range(1, 8))
    assert max_res < 1e-10, f"chain-rule residual too large: {max_res:.2e}"


def test_labeled_matches_enumeration():
    for (tau, g) in [(0.3, 0.8), (0.5, 1.1), (0.2, 0.4)]:
        for L in range(1, 7):
            assert abs(labeled_EFQ_tfim(L, tau, g)
                       - labeled_EFQ_enumerate(L, tau, g)) < 1e-9


def test_O5_kernel():
    demo = bell_kernel_demo()
    for name in ["XX", "YY", "ZZ"]:
        assert demo[name][1] < 1e-9, f"{name} should be silent"
    for name in ["XY", "XZ", "YX", "YZ", "ZX", "ZY"]:
        assert abs(demo[name][1] - 1.0) < 1e-9, f"{name} should be fully active"


def test_decoupling():
    S, cmax = decoupling_sanity(0.3)
    assert S < 1e-9 and cmax < 1e-9


def run_all():
    for t in [test_conservation_residual, test_labeled_matches_enumeration,
              test_O5_kernel, test_decoupling]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_two_qubit_tfim: ALL PASS")


if __name__ == "__main__":
    run_all()
