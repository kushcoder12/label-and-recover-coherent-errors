"""Circuits used by the paper, written as PennyLane QNodes.

``signal_tape_qnode`` mirrors the hardware experiment in ``hardware/build_circuits.py``:
prepare |+>, apply L coherent Z-errors RZ(theta), rotate Y into Z (S^dagger, H) and
measure. With labels s_l from ``sign_label_twirl`` the ideal result is
<Y> = sin(theta * sum_l s_l).

``visibility_qnode`` is the matching reference: the same L layers with a known net
phase pi/2, so its ideal result is 1 and its measured value is the visibility v(L).

``joint_label_qnode`` stores the twirl label in an ancilla, so the joint distribution
p(label, outcome) and the label-averaged marginal can both be read from one circuit
and differentiated exactly with PennyLane's autodiff.
"""

import numpy as np
import pennylane as qp


def _measure_y(wire):
    qp.adjoint(qp.S(wires=wire))
    qp.Hadamard(wires=wire)


def signal_tape_qnode(dev, L):
    """QNode f(theta) -> <Y> after L coherent RZ(theta) errors on |+>."""

    @qp.qnode(dev)
    def circuit(theta):
        qp.Hadamard(wires=0)
        for _ in range(L):
            qp.RZ(theta, wires=0)
        _measure_y(0)
        return qp.expval(qp.PauliZ(0))

    return circuit


def visibility_qnode(dev, L):
    """QNode f() -> measured <Y> for the net-phase-pi/2 reference at depth L."""

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=0)
        for _ in range(L):
            qp.RZ(np.pi / (2 * L), wires=0)
        _measure_y(0)
        return qp.expval(qp.PauliZ(0))

    return circuit


def joint_label_qnode(dev, mu, marginal=False):
    """Single-qubit model with the twirl label purified into an ancilla (wire 0).

    The ancilla is prepared with P(label = +1) = (1 + mu) / 2. If the ancilla is |1>
    (label -1), the error RZ(theta) on the system (wire 1) is dressed with X on both
    sides. Returns probs over (ancilla, system) or, if ``marginal``, over the system
    only. Ancilla index 0 <-> s = +1; system index 0 <-> b = +1.
    """
    phi = 2 * np.arccos(np.sqrt((1 + mu) / 2))

    @qp.qnode(dev)
    def circuit(theta):
        qp.RY(phi, wires=0)
        qp.Hadamard(wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RZ(theta, wires=1)
        qp.CNOT(wires=[0, 1])
        _measure_y(1)
        return qp.probs(wires=[1]) if marginal else qp.probs(wires=[0, 1])

    return circuit
