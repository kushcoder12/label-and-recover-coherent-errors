"""Cross-check the PennyLane layer against the repository's numpy reference engine.

Three independent agreements are tested:
1. per-instance <Y> from the PennyLane transform vs the core/ state-vector engine;
2. autodiff Fisher terms vs core/fisher.py (finite differences);
3. the PennyLane estimator vs the paper's hardware estimator in hardware/analyze.py.
"""

import os
import sys

import numpy as np
import pennylane as qp
import pytest

from core.fisher import decompose_joint_fisher
from core.operator_on_state import single_qubit_op
from core.single_qubit_gates import Hadamard, Rz
from labeled_rc import (
    fisher_decomposition,
    fit_theta,
    joint_label_qnode,
    sample_signs,
    sign_label_twirl,
    signal_tape_qnode,
)
from labeled_rc.fisher import joint_matrix

_Y = np.array([[0, -1j], [1j, 0]])


def _core_expval_y(theta, signs):
    """<Y> for one label vector, computed with the core/ numpy engine (no PennyLane)."""
    state = single_qubit_op(np.array([1, 0], dtype=complex), Hadamard(), 1, 1)
    for s in signs:
        state = single_qubit_op(state, Rz(s * theta), 1, 1)
    return float(np.real(state.conj() @ _Y @ state))


@pytest.mark.parametrize("L", [1, 4, 8])
def test_transform_matches_core_engine(L):
    signs = sample_signs(20, L, seed=100 + L)
    dev = qp.device("default.qubit", wires=1)
    pl = sign_label_twirl(signal_tape_qnode(dev, L), signs=signs)(0.17)
    core = np.array([_core_expval_y(0.17, row) for row in signs])
    assert np.allclose(pl, core, atol=1e-12)


@pytest.mark.parametrize("theta, mu", [(0.0, 0.0), (0.2, 0.5), (0.6, 0.8)])
def test_autodiff_fisher_matches_core_finite_differences(theta, mu):
    dev = qp.device("default.qubit", wires=2)
    qnode = joint_label_qnode(dev, mu)
    pl = fisher_decomposition(qnode, theta)
    core = decompose_joint_fisher(lambda t: joint_matrix(qnode(t)), theta)
    # core uses central differences (eps = 1e-6), so agreement is ~1e-6, not 1e-12
    assert np.allclose(pl, core, atol=1e-5)


def test_estimator_matches_paper_hardware_analysis():
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hardware")
    )
    analyze = pytest.importorskip("analyze")
    rng = np.random.default_rng(7)
    for _ in range(5):
        m = rng.choice(np.arange(-8, 9, 2), size=40)
        y = 0.97 * np.sin(0.12 * m) + 0.02 * rng.standard_normal(40)
        assert fit_theta(m, y, 0.97) == pytest.approx(analyze._fit_theta(m, y, 0.97), abs=1e-12)
