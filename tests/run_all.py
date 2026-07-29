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
    "test_noise",             # NOISE_MODULE.md oracles N1-N6
    "test_noise_estimation",  # NOISE_MODULE.md robustness / unbiasedness
    "test_error_map",         # GAPS_CODE.md GAP 3 (coherent-error map recovery)
    "test_corpus",            # GAPS_CODE.md GAP 4 (corpus generality)
    "test_shotcost",          # GAPS_CODE.md GAP 5 (shot-cost vs RPE)
]


def main():
    for name in MODULES:
        print(f"\n===== {name} =====")
        mod = importlib.import_module(name)
        mod.run_all()
    print("\nALL TESTS PASSED (Oracles O1-O5 verified).")


if __name__ == "__main__":
    main()
