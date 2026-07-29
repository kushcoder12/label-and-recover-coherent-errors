"""Phase 1 verification (CLAUDE.md §5, Oracle O5).

- Every rotation gate R_P(theta) is unitary.
- Rz(pi) == diag(-1j, 1j) (= Z up to global phase).
- Rzz / CZ / CPhase are correct.
- Pauli propagation table: CX X_c CX = X_c X_t, CX Z_c CX = Z_c, etc.
- Bell-state stabilizer silence: exp(-i theta X X)|Phi+> = e^{-i theta}|Phi+>.

Runnable directly (`python tests/test_gates.py`) or via pytest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Rx, Ry, Rz, U3
from core.two_qubit_gates import Rzz, CZ, CPhase, CX_gate
from core.pauli import pauli, pauli_string, propagate_generator, expectation


TOL = 1e-12


def _is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=TOL)


def test_rotations_unitary():
    for theta in [0.0, 0.2, 1.0, np.pi, 2.7, -1.3]:
        assert _is_unitary(Rx(theta)), f"Rx({theta}) not unitary"
        assert _is_unitary(Ry(theta)), f"Ry({theta}) not unitary"
        assert _is_unitary(Rz(theta)), f"Rz({theta}) not unitary"
    assert _is_unitary(U3(0.7, 1.1, -0.4)), "U3 not unitary"


def test_rz_pi_is_Z_up_to_phase():
    got = Rz(np.pi)
    want = np.diag([-1j, 1j])
    assert np.allclose(got, want, atol=TOL), f"Rz(pi)={got}, want {want}"
    # Z up to global phase: Rz(pi) = -1j * Z
    assert np.allclose(got, -1j * pauli("Z"), atol=TOL)


def test_rzz_cz_cphase():
    # Rzz(theta) = diag(e^-it/2, e^it/2, e^it/2, e^-it/2)
    t = 0.9
    a, b = np.exp(-1j * t / 2), np.exp(1j * t / 2)
    assert np.allclose(Rzz(t), np.diag([a, b, b, a]), atol=TOL)
    assert _is_unitary(Rzz(t))
    # Rzz = exp(-i t ZZ/2) directly
    ZZ = pauli_string("ZZ")
    from_expm = _expm_hermitian(-1j * t / 2 * ZZ)
    assert np.allclose(Rzz(t), from_expm, atol=1e-10)
    # CZ == diag(1,1,1,-1); CPhase(pi) == CZ
    assert np.allclose(CZ(), np.diag([1, 1, 1, -1]), atol=TOL)
    assert np.allclose(CPhase(np.pi), CZ(), atol=1e-12)


def _expm_hermitian(A):
    """exp(A) for anti-Hermitian A via eigendecomposition (no scipy)."""
    # A = -i H with H Hermitian -> eig of iA is real
    w, V = np.linalg.eigh(1j * A)          # iA Hermitian
    return V @ np.diag(np.exp(-1j * w)) @ V.conj().T


def test_pauli_propagation_O5():
    """Oracle O5: CNOT conjugation table (control=q1, target=q2)."""
    CX = CX_gate()
    checks = {
        "CX Xc CX = Xc Xt": (propagate_generator(pauli_string("XI"), CX), pauli_string("XX")),
        "CX Zc CX = Zc":    (propagate_generator(pauli_string("ZI"), CX), pauli_string("ZI")),
        "CX Xt CX = Xt":    (propagate_generator(pauli_string("IX"), CX), pauli_string("IX")),
        "CX Zt CX = Zc Zt": (propagate_generator(pauli_string("IZ"), CX), pauli_string("ZZ")),
    }
    for name, (got, want) in checks.items():
        assert np.allclose(got, want, atol=TOL), f"{name} failed"


def test_bell_stabilizer_silence_O5():
    """Oracle O5: X_1 X_2 is a stabilizer of |Phi+>, so exp(-i t XX)|Phi+> = e^{-i t}|Phi+>."""
    # |Phi+> = (|00> + |11>)/sqrt2
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1 / np.sqrt(2)
    XX = pauli_string("XX")
    assert np.allclose(XX @ phi, phi, atol=TOL), "XX is not a stabilizer of |Phi+>"
    for t in [0.3, 1.0, 2.2]:
        U = _expm_hermitian(-1j * t * XX)      # exp(-i t XX)
        got = U @ phi
        want = np.exp(-1j * t) * phi
        assert np.allclose(got, want, atol=1e-10), f"stabilizer silence failed at t={t}"


def run_all():
    tests = [
        test_rotations_unitary,
        test_rz_pi_is_Z_up_to_phase,
        test_rzz_cz_cphase,
        test_pauli_propagation_O5,
        test_bell_stabilizer_silence_O5,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_gates: ALL PASS")


if __name__ == "__main__":
    run_all()
