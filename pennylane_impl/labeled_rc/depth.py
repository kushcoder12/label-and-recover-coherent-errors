"""Depth theorem (no washout of coherent information), with PennyLane's quantum Fisher information.

The single-qubit L-cycle model of ``core/circuits.trotter_single_qubit``: start in |0>, and in
each cycle apply the ideal rotation RX(alpha) followed by the coherent error RZ(s_l * theta).
A label s_l = -1 is realised as an X ... X dressing of the error, as in ``sign_label_twirl``.

At theta = 0 the paper gives (experiments/exp2_depth_scaling.py, Oracles O2/O3)

    labeled   E_s[F_Q] = sum_l sin^2(l * alpha)      (grows ~ L/2)
    coherent  F_Q      = (sum_l sin(l * alpha))^2    (bounded)

Here both are computed with ``qp.gradients.quantum_fisher`` on the PennyLane circuit: the
labeled value by averaging the QFI of every label vector (exact enumeration), the coherent
value from the untwirled circuit (all labels +1).
"""

from itertools import product

import numpy as np
import pennylane as qp
from pennylane import numpy as pnp


def trotter_qnode(dev, alpha, signs):
    """QNode f(theta) -> state of the L-cycle model for one fixed label vector ``signs``."""
    signs = [int(s) for s in signs]
    if any(s not in (1, -1) for s in signs):
        raise ValueError("signs must be +/-1")

    @qp.qnode(dev)
    def circuit(theta):
        for s in signs:
            qp.RX(alpha, wires=0)
            if s == -1:
                qp.PauliX(wires=0)
            qp.RZ(theta, wires=0)
            if s == -1:
                qp.PauliX(wires=0)
        return qp.state()

    return circuit


def qfi(qnode, theta=0.0):
    """Quantum Fisher information of a single-parameter QNode, via ``qp.gradients.quantum_fisher``."""
    th = pnp.array(theta, requires_grad=True)
    return float(np.squeeze(qp.gradients.quantum_fisher(qnode)(th)))


def labeled_qfi(L, alpha, theta=0.0, p_plus=0.5, dev=None):
    """E_s[F_Q] over all 2^L label vectors, weighted by P(s_l = +1) = p_plus."""
    dev = dev or qp.device("default.qubit", wires=1)
    total = 0.0
    for signs in product((1, -1), repeat=L):
        n_plus = sum(1 for s in signs if s == 1)
        weight = p_plus**n_plus * (1 - p_plus) ** (L - n_plus)
        if weight:
            total += weight * qfi(trotter_qnode(dev, alpha, signs), theta)
    return total


def coherent_qfi(L, alpha, theta=0.0, dev=None):
    """F_Q of the untwirled circuit (every label +1)."""
    dev = dev or qp.device("default.qubit", wires=1)
    return qfi(trotter_qnode(dev, alpha, [1] * L), theta)


def labeled_closed_form(L, alpha):
    """sum_{l=1..L} sin^2(l alpha)."""
    return float(sum(np.sin(l * alpha) ** 2 for l in range(1, L + 1)))


def coherent_closed_form(L, alpha):
    """(sum_{l=1..L} sin(l alpha))^2."""
    return float(sum(np.sin(l * alpha) for l in range(1, L + 1)) ** 2)
