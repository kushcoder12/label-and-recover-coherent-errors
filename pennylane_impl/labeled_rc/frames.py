"""Pauli twirl frames for two-qubit Clifford gates.

For a Clifford gate G and a two-qubit Pauli P, the frame P' = G P G^dagger is
again a Pauli (up to a phase), and P' G P = G. Inserting P before G and P'
after it therefore leaves the ideal circuit unchanged, while any coherent
error attached to G is randomized. This is the standard randomized-compiling
(Pauli twirling) construction.

Frames are computed by conjugation, never typed in by hand, so they are correct
for any two-qubit Clifford that PennyLane can give a matrix for.
"""

from functools import lru_cache
from itertools import product

import numpy as np
import pennylane as qp

PAULI_LABELS = ("I", "X", "Y", "Z")

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_pair_matrix(a, b):
    """Matrix of the two-qubit Pauli a (first wire) tensor b (second wire)."""
    return np.kron(_PAULI_MATRICES[a], _PAULI_MATRICES[b])


def conjugate_pauli_pair(gate_matrix, a, b, atol=1e-10):
    """Return (c, d) with G (a x b) G^dagger = phase * (c x d) for a Clifford G.

    Raises ValueError if the result is not a Pauli (G is not a Clifford).
    """
    conj = gate_matrix @ pauli_pair_matrix(a, b) @ gate_matrix.conj().T
    for c, d in product(PAULI_LABELS, repeat=2):
        target = pauli_pair_matrix(c, d)
        # conj = phase * target  <=>  |<target, conj>| / 4 == 1
        overlap = np.trace(target.conj().T @ conj) / 4
        if abs(abs(overlap) - 1) < atol:
            return c, d
    raise ValueError("gate is not a two-qubit Clifford: conjugate is not a Pauli")


@lru_cache(maxsize=None)
def twirl_frames(gate_name):
    """All 16 frames {(a, b): (c, d)} for a named two-qubit Clifford gate.

    The matrix is taken in PennyLane's own wire order for the gate, so
    (a, b) acts on (wires[0], wires[1]).
    """
    gate = getattr(qp, gate_name)(wires=[0, 1])
    G = qp.matrix(gate, wire_order=[0, 1])
    return {(a, b): conjugate_pauli_pair(G, a, b) for a, b in product(PAULI_LABELS, repeat=2)}


def pauli_op(label, wire):
    """PennyLane operator for a single-qubit Pauli label, or None for identity."""
    if label == "I":
        return None
    return {"X": qp.PauliX, "Y": qp.PauliY, "Z": qp.PauliZ}[label](wires=wire)
