"""Fisher-information conservation law with exact (autodiff) derivatives."""

import numpy as np
import pennylane as qp
import pytest

from labeled_rc import exact_marginal_fisher, fisher_decomposition, joint_label_qnode

DEV = qp.device("default.qubit", wires=2)


@pytest.mark.parametrize("theta", [0.0, 0.05, 0.3, 0.9])
@pytest.mark.parametrize("T", [0.0, 0.25, 0.5, 1.0])
def test_conservation_is_exact(theta, T):
    mu = 1 - T
    F_joint, F_marg, delta = fisher_decomposition(joint_label_qnode(DEV, mu), theta)
    assert abs(F_joint - 1.0) < 1e-10  # F0 = 1 for this model
    assert abs(F_joint - (F_marg + delta)) < 1e-10


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.9])
@pytest.mark.parametrize("mu", [0.0, 0.3, 0.75, 1.0])
def test_exact_marginal_formula(theta, mu):
    _, F_marg, _ = fisher_decomposition(joint_label_qnode(DEV, mu), theta)
    assert abs(F_marg - exact_marginal_fisher(theta, mu)) < 1e-10


@pytest.mark.parametrize("T", [0.0, 0.3, 0.5, 1.0])
def test_paper_closed_forms_hold_at_theta_zero(T):
    F_joint, F_marg, delta = fisher_decomposition(joint_label_qnode(DEV, 1 - T), 0.0)
    assert abs(F_marg - (1 - T) ** 2) < 1e-10
    assert abs(delta - (2 * T - T**2)) < 1e-10


def test_closed_forms_are_leading_order_only():
    # At finite theta the (1-T)^2 form is not exact; the exact marginal is smaller.
    _, F_marg, _ = fisher_decomposition(joint_label_qnode(DEV, 0.5), 0.3)
    assert F_marg < 0.25 - 1e-3
