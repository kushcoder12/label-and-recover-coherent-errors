"""Coherent-error MAP recovery verification.

- Noiseless smoke (ties to O-series): exact recovery of every angle.
- Under the full NoiseProfile the recovery stays unbiased (visibility-debiased,
  bias at the probe finite-difference floor, far below shot-noise).
- Silent (kernel) generators are detected and flagged.
- Visibility decreases with noise strength.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.error_map import CoherentError, recover_error_vector, _embed_1q
from core.single_qubit_gates import Hadamard
from core.two_qubit_gates import CX_gate
from core.noise import NoiseProfile
from exp8_error_map import build_circuit, N as EXP8_N


def _small_setup():
    H = Hadamard()
    ideal = [_embed_1q(H, 1, 2), _embed_1q(H, 2, 2), CX_gate()]
    errs = [CoherentError(pos=2, qubits=(1,), gen="Z", theta=0.22),
            CoherentError(pos=2, qubits=(2,), gen="Y", theta=0.14),
            CoherentError(pos=2, qubits=(1, 2), gen="ZZ", theta=0.08)]
    return ideal, errs


def test_noiseless_exact_recovery():
    ideal, errs = _small_setup()
    res = recover_error_vector(2, ideal, errs, NoiseProfile(1.0, T1=np.inf, T2=np.inf))
    assert max(abs(r.theta_hat - r.theta_true) for r in res) < 1e-5


def test_noisy_unbiased():
    ideal, errs = _small_setup()
    for prof in [NoiseProfile(1.0, r_1q=0.01, r_2q=0.008, T1=50.0, T2=40.0),
                 NoiseProfile(1.0, r_1q=0.03, r_2q=0.02, T1=5.0, T2=8.0)]:  # incl. strong
        res = recover_error_vector(2, ideal, errs, prof)
        assert max(abs(r.theta_hat - r.theta_true) for r in res) < 1e-3


def test_silent_detected():
    H = Hadamard()
    ideal = [_embed_1q(H, 1, 2), _embed_1q(H, 2, 2), CX_gate()]
    # X on q2 is a stabilizer of |++> before the CX -> silent (g=+1)
    errs = [CoherentError(pos=2, qubits=(2,), gen="X", theta=0.2)]
    res = recover_error_vector(2, ideal, errs, NoiseProfile(1.0))
    assert res[0].silent and abs(abs(res[0].g) - 1.0) < 1e-6


def test_visibility_decreases_with_noise():
    ideal, errs = _small_setup()
    v_clean = recover_error_vector(2, ideal, errs, NoiseProfile(1.0, T1=np.inf, T2=np.inf))
    v_noisy = recover_error_vector(2, ideal, errs, NoiseProfile(1.0, r_1q=0.05))
    for rc, rn in zip(v_clean, v_noisy):
        assert rn.visibility < rc.visibility


def test_exp8_full_vector_noiseless():
    ops, errors = build_circuit(seed=1)
    res = recover_error_vector(EXP8_N, ops, errors, NoiseProfile(1.0, T1=np.inf, T2=np.inf))
    assert max(abs(r.theta_hat - r.theta_true) for r in res if not r.silent) < 1e-5


def run_all():
    for t in [test_noiseless_exact_recovery, test_noisy_unbiased, test_silent_detected,
              test_visibility_decreases_with_noise, test_exp8_full_vector_noiseless]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_error_map: ALL PASS")


if __name__ == "__main__":
    run_all()
