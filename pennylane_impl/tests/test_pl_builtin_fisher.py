"""Conservation law checked with PennyLane's own Fisher-information transforms.

- ``qp.gradients.classical_fisher`` on the label-in-ancilla circuit gives F_joint, the Fisher
  information of the joint (label, outcome) distribution.
- ``qp.gradients.quantum_fisher`` on the untwirled state gives F_0, the quantum Fisher
  information of the coherent error.

The paper's claim is F_joint = F_0 at every twirl strength: keeping the labels recovers all of it.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane import numpy as pnp

from labeled_rc import fisher_decomposition, joint_label_qnode

DEV2 = qp.device("default.qubit", wires=2)
DEV1 = qp.device("default.qubit", wires=1)


@qp.qnode(DEV1)
def untwirled_state(theta):
    qp.Hadamard(wires=0)
    qp.RZ(theta, wires=0)
    return qp.state()


def _scalar(x):
    return float(np.squeeze(x))


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.7])
def test_quantum_fisher_of_untwirled_error(theta):
    F0 = _scalar(qp.gradients.quantum_fisher(untwirled_state)(pnp.array(theta, requires_grad=True)))
    assert F0 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.2, 0.7])
@pytest.mark.parametrize("mu", [0.0, 0.4, 1.0])
def test_classical_fisher_of_joint_distribution_equals_qfi(theta, mu):
    th = pnp.array(theta, requires_grad=True)
    qnode = joint_label_qnode(DEV2, mu)
    F_joint_builtin = _scalar(qp.gradients.classical_fisher(qnode)(th))
    F0 = _scalar(qp.gradients.quantum_fisher(untwirled_state)(th))
    F_joint, F_marg, delta = fisher_decomposition(qnode, theta)

    assert F_joint_builtin == pytest.approx(F_joint, abs=1e-10)  # two independent routes
    assert F_joint_builtin == pytest.approx(F0, abs=1e-10)  # labels recover the full QFI
    assert F_marg + delta == pytest.approx(F0, abs=1e-10)
