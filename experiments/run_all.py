"""Regenerate every figure (Fig 1-5) in sequence, printing each oracle comparison.

Usage:  python experiments/run_all.py
"""

import os
import sys
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXPERIMENTS = [
    ("exp1_conservation", "Figure 1"),
    ("exp2_depth_scaling", "Figure 2"),
    ("exp3_noisy_saturation", "Figure 3"),
    ("exp4_two_qubit_tfim", "Figure 4"),
    ("exp5_recover_and_correct", "Figure 5"),
    ("exp6_robustness", "Figure 6"),
    ("exp7_saturation_full", "Figure 7"),
    ("exp8_error_map", "Figure 8"),
    ("exp9_corpus", "Figure 9"),
    ("exp10_shotcost", "Figure 10"),
]


def main():
    for name, fig in EXPERIMENTS:
        print(f"\n########## {name}  ->  {fig} ##########")
        mod = importlib.import_module(name)
        mod.main()
    print("\nAll figures regenerated in figures/.")


if __name__ == "__main__":
    main()
