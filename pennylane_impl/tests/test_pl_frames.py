"""Twirl frames: P' G P must equal G (up to global phase) for every Pauli P."""

from itertools import product

import numpy as np
import pennylane as qp
import pytest

from labeled_rc.frames import PAULI_LABELS, conjugate_pauli_pair, pauli_pair_matrix, twirl_frames


def _equal_up_to_phase(A, B, atol=1e-10):
    overlap = np.trace(B.conj().T @ A) / A.shape[0]
    return abs(abs(overlap) - 1) < atol


@pytest.mark.parametrize("gate_name", ["CNOT", "CZ"])
def test_all_16_frames_leave_gate_unchanged(gate_name):
    G = qp.matrix(getattr(qp, gate_name)(wires=[0, 1]), wire_order=[0, 1])
    frames = twirl_frames(gate_name)
    assert len(frames) == 16
    for (a, b), (c, d) in frames.items():
        dressed = pauli_pair_matrix(c, d) @ G @ pauli_pair_matrix(a, b)
        assert _equal_up_to_phase(dressed, G), (gate_name, a, b, c, d)


def test_known_cnot_propagation():
    # X on the control spreads to both qubits; Z on the target spreads to both.
    frames = twirl_frames("CNOT")
    assert frames[("X", "I")] == ("X", "X")
    assert frames[("I", "Z")] == ("Z", "Z")
    assert frames[("Z", "I")] == ("Z", "I")
    assert frames[("I", "X")] == ("I", "X")


def test_non_clifford_is_rejected():
    T2 = np.kron(np.diag([1, np.exp(1j * np.pi / 4)]), np.eye(2))
    with pytest.raises(ValueError):
        conjugate_pauli_pair(T2, "X", "I")


def test_frames_are_a_permutation_of_the_paulis():
    for gate_name in ("CNOT", "CZ"):
        images = set(twirl_frames(gate_name).values())
        assert images == set(product(PAULI_LABELS, repeat=2))
