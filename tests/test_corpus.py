"""GAP 4 verification (GAPS_CODE.md): corpus generality.

- The native QASM loader parses bundled QASMBench circuits into unitary circuits.
- Noiseless smoke test: exact coherent-error recovery on several circuit families.
- Native random-Clifford circuit is unitary and recovers exactly.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.qasm_loader import load_circuit
from core.error_map import recover_error_vector
from core.noise import NoiseProfile
from core._compat import quiet_accelerate
from exp9_corpus import _QDIR, _inject, _native_random_clifford, CORPUS

_CLEAN = NoiseProfile(1.0, T1=np.inf, T2=np.inf)


def _unitary(ops, n):
    with quiet_accelerate():
        U = np.eye(2**n, dtype=complex)
        for o in ops:
            U = o @ U
        return np.allclose(U.conj().T @ U, np.eye(2**n), atol=1e-9)


def test_loader_parses_unitary():
    for _disp, sub in CORPUS[:6]:
        n, ops, gl = load_circuit(os.path.join(_QDIR, sub, sub + ".qasm"))
        assert _unitary(ops, n), f"{sub} not unitary"


def test_noiseless_recovery_across_families():
    for _disp, sub in [("QFT", "qft_n4"), ("GHZ/cat", "cat_state_n4"),
                       ("Grover", "grover_n2"), ("Teleport", "teleportation_n3")]:
        n, ops, gl = load_circuit(os.path.join(_QDIR, sub, sub + ".qasm"))
        errs = _inject(n, len(ops))
        res = recover_error_vector(n, ops, errs, _CLEAN)
        active = [r for r in res if not r.silent]
        assert active, f"{sub}: no active errors"
        assert max(abs(r.theta_hat - r.theta_true) for r in active) < 1e-5


def test_native_random_clifford():
    n, ops, gl = _native_random_clifford()
    assert _unitary(ops, n)
    errs = _inject(n, len(ops))
    res = recover_error_vector(n, ops, errs, _CLEAN)
    active = [r for r in res if not r.silent]
    assert max(abs(r.theta_hat - r.theta_true) for r in active) < 1e-5


def run_all():
    for t in [test_loader_parses_unitary, test_noiseless_recovery_across_families,
              test_native_random_clifford]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_corpus: ALL PASS")


if __name__ == "__main__":
    run_all()
