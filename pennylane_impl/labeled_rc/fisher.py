"""Fisher-information decomposition with exact (autodiff) derivatives.

For a joint distribution P[b, s] over outcome b and twirl label s, the chain rule gives

    F_joint = F_marg + E_b[ F_{s|b} ]

exactly (the conservation law of arXiv:2607.26756). ``core/fisher.py`` evaluates the
same three terms with finite differences; here the derivatives come from
``qp.jacobian``, so there is no finite-difference error floor.

Note: ``qp.gradients.classical_fisher`` measures *all* device wires (documented
behaviour), so it cannot give the label-averaged marginal. The marginal term is
therefore computed directly from the Jacobian of the joint probabilities.
"""

import numpy as np
import pennylane as qp
from pennylane import numpy as pnp


def joint_matrix(joint_probs):
    """Reshape probs over (ancilla, system) into P[b, s] (rows b, columns s)."""
    return np.asarray(joint_probs, float).reshape(2, 2).T


def fisher_decomposition(joint_qnode, theta):
    """Return (F_joint, F_marg, Delta) at theta, with Delta = E_b[F_{s|b}].

    ``joint_qnode(theta)`` must return probs over (ancilla label, system outcome),
    for example ``labeled_rc.joint_label_qnode``.
    """
    th = pnp.array(theta, requires_grad=True)
    P = joint_matrix(joint_qnode(th))
    dP = joint_matrix(qp.jacobian(joint_qnode)(th))

    mask = P > 1e-15
    F_joint = float(np.sum(dP[mask] ** 2 / P[mask]))

    Pb, dPb = P.sum(axis=1), dP.sum(axis=1)
    mb = Pb > 1e-15
    F_marg = float(np.sum(dPb[mb] ** 2 / Pb[mb]))

    delta = 0.0
    for b in range(2):
        if Pb[b] <= 1e-15:
            continue
        for s in range(2):
            if P[b, s] <= 1e-15:
                continue
            p_cond = P[b, s] / Pb[b]
            dp_cond = (dP[b, s] * Pb[b] - P[b, s] * dPb[b]) / Pb[b] ** 2
            delta += Pb[b] * dp_cond**2 / p_cond
    return F_joint, F_marg, float(delta)


def exact_marginal_fisher(theta, mu):
    """Closed form of F_marg for the single-qubit model at any theta.

    F_marg = mu^2 cos^2(theta) / (1 - mu^2 sin^2(theta)); it equals (1 - T)^2 with
    T = 1 - |mu| only to leading order in theta.
    """
    return float(mu**2 * np.cos(theta) ** 2 / (1 - mu**2 * np.sin(theta) ** 2))
