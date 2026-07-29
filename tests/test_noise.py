"""Noise-module oracles N1-N6 (NOISE_MODULE.md §6).

N1 trace preservation; N2 T1 limit; N3 T2 limit; N4 depolarizing fidelity;
N5 noiseless limit (identity layer + O1-O5 under noise=None, checked in the
    existing suite / test_noise_noiseless_reproduces_oracles); N6 DM/trajectory
    agreement.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.noise import (NoiseProfile, amplitude_damping_kraus, dephasing_kraus,
                        depolarizing_kraus_1q, depolarizing_kraus_2q,
                        is_trace_preserving, avg_gate_infidelity,
                        apply_incoherent_layer, readout_confusion,
                        _embed_two_qubit_op)
from core.quantum_errors import state_to_density_matrix

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _profile(**kw):
    base = dict(dt=30e-9, T1=150e-6, T2=100e-6, r_1q=2e-4, r_2q=3e-3,
                e01=0.008, e10=0.02)
    base.update(kw)
    return NoiseProfile(**base)


def test_N1_trace_preserving():
    p = _profile()
    for K in [amplitude_damping_kraus(p), dephasing_kraus(p),
              depolarizing_kraus_1q(p), depolarizing_kraus_2q(p)]:
        assert is_trace_preserving(K, tol=1e-12)


def test_N2_T1_limit():
    dt, T1 = 1e-6, 10e-6
    p = NoiseProfile(dt=dt, T1=T1, T2=2 * T1)     # pure amplitude damping
    rho = np.array([[0, 0], [0, 1]], dtype=complex)  # excited |1>
    for L in range(1, 12):
        rho = apply_incoherent_layer(rho, p, 1, mode="dm")
        assert abs(rho[1, 1].real - np.exp(-L * dt / T1)) < 1e-12


def test_N3_T2_limit():
    dt, T1, T2 = 1e-6, 10e-6, 12e-6
    p = NoiseProfile(dt=dt, T1=T1, T2=T2)
    rho = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)   # |+><+|
    for L in range(1, 12):
        rho = apply_incoherent_layer(rho, p, 1, mode="dm")
        assert abs(2 * abs(rho[0, 1]) - np.exp(-L * dt / T2)) < 1e-12


def test_N4_depolarizing_fidelity():
    for r in [1e-4, 2e-4, 1e-3, 3e-3, 1e-2]:
        p = NoiseProfile(dt=30e-9, T1=150e-6, T2=100e-6, r_1q=r, r_2q=r)
        assert abs(avg_gate_infidelity(depolarizing_kraus_1q(p), 2) - r) < 1e-6
        assert abs(avg_gate_infidelity(depolarizing_kraus_2q(p), 4) - r) < 1e-6


def test_N5_noiseless_identity_layer():
    p = NoiseProfile(dt=30e-9)                    # T1=T2=inf, all rates 0
    assert p.is_noiseless()
    rng = np.random.default_rng(0)
    rho = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
    assert np.allclose(apply_incoherent_layer(rho, p, 1, mode="dm"), rho)
    # zero-strength depolarizing has exactly zero infidelity
    assert abs(avg_gate_infidelity(depolarizing_kraus_1q(p), 2)) < 1e-15


def test_N5_noiseless_reproduces_oracles():
    """With noise=None the depth builder reproduces Oracle O2 exactly (wiring check)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "experiments"))
    from exp2_depth_scaling import verify_oracle_O2
    assert verify_oracle_O2(verbose=False) < 1e-6


def test_N6_dm_vs_trajectory():
    dt = 1e-6
    p = NoiseProfile(dt=dt, T1=8e-6, T2=6e-6, r_1q=0.01)
    L = 4
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = state_to_density_matrix(plus)
    for _ in range(L):
        rho = apply_incoherent_layer(rho, p, 1, mode="dm")
    zdm = np.trace(rho @ _Z).real

    rng = np.random.default_rng(0)
    N = 40000
    zs = np.empty(N)
    for i in range(N):
        st = plus.copy()
        for _ in range(L):
            st, _o = apply_incoherent_layer(st, p, 1, mode="trajectory", rng=rng)
        zs[i] = (abs(st[0]) ** 2 - abs(st[1]) ** 2)
    se = zs.std() / np.sqrt(N)
    assert abs(zdm - zs.mean()) < 5 * se, f"DM {zdm} vs traj {zs.mean()}+-{se}"


def test_N6_embed_matches_kron():
    p = NoiseProfile(dt=1e-6, T1=1e6, T2=1e6, r_2q=0.02)
    K = depolarizing_kraus_2q(p)[5]
    assert np.allclose(_embed_two_qubit_op(K, 1, 2, 3), np.kron(K, np.eye(2)))
    assert np.allclose(_embed_two_qubit_op(K, 2, 3, 3), np.kron(np.eye(2), K))


def test_readout_confusion():
    p = _profile(e01=0.01, e10=0.03)
    R = readout_confusion(p)
    assert np.allclose(R.sum(axis=0), [1, 1])          # columns are distributions
    # true |0> -> measured P(1) = e01
    assert abs((R @ np.array([1.0, 0.0]))[1] - 0.01) < 1e-12
    assert abs((R @ np.array([0.0, 1.0]))[0] - 0.03) < 1e-12


def run_all():
    for t in [test_N1_trace_preserving, test_N2_T1_limit, test_N3_T2_limit,
              test_N4_depolarizing_fidelity, test_N5_noiseless_identity_layer,
              test_N5_noiseless_reproduces_oracles, test_N6_dm_vs_trajectory,
              test_N6_embed_matches_kron, test_readout_confusion]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_noise: ALL PASS")


if __name__ == "__main__":
    run_all()
