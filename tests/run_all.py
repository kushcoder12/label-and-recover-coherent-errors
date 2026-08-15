"""Run every oracle-backed test file in sequence. Exits non-zero on first failure.

Usage:  python tests/run_all.py
"""

import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = [
    "test_gates",
    "test_fisher_identities",
    "test_conservation",
    "test_depth_table",
    "test_noisy_saturation",
    "test_two_qubit_tfim",
    "test_estimator",
    "test_noise",             # noise-module oracles N1-N6
    "test_noise_estimation",  # noise robustness / unbiasedness
    "test_error_map",         # coherent-error map recovery
    "test_corpus",            # corpus generality
    "test_shotcost",          # shot-cost vs RPE
]


def main():
    for name in MODULES:
        print(f"\n===== {name} =====")
        mod = importlib.import_module(name)
        mod.run_all()
    print("\nALL TESTS PASSED (Oracles O1-O5 verified).")


if __name__ == "__main__":
    main()
