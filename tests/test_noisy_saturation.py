"""Noisy-saturation sanity checks (no exact oracle).

- gamma_z=0 recovers Oracle O2 labeled (sum sin^2(ell*alpha)) to machine precision.
- Saturation depth L*(T2) increases (weakly monotone) as gamma_z decreases (T2 grows).
- Dephasing Kraus set is trace preserving.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.channels import dephasing_kraus, amplitude_damping_kraus, is_trace_preserving
from exp3_noisy_saturation import (noisy_labeled_EFQ, noiseless_labeled,
                                   saturation_depth)


def test_channels_trace_preserving():
    assert is_trace_preserving(dephasing_kraus(0.3, 1.0))
    assert is_trace_preserving(amplitude_damping_kraus(0.2, 1.0))


def test_gamma0_recovers_O2():
    err = max(abs(noisy_labeled_EFQ(L, 1.0, 0.0) - noiseless_labeled(L, 1.0))
              for L in range(1, 13))
    assert err < 1e-9, f"gamma_z=0 does not recover O2: {err:.2e}"


def test_Lstar_monotone_in_T2():
    gammas = [0.4, 0.2, 0.1, 0.05, 0.02]          # decreasing gamma -> increasing T2
    Lstars = [saturation_depth(1.0, g, L_max=80)[0] for g in gammas]
    for a, b in zip(Lstars, Lstars[1:]):
        assert b >= a, f"L* not monotone in T2: {Lstars}"


def run_all():
    for t in [test_channels_trace_preserving, test_gamma0_recovers_O2,
              test_Lstar_monotone_in_T2]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_noisy_saturation: ALL PASS")


if __name__ == "__main__":
    run_all()
