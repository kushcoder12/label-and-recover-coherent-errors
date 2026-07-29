"""Fisher-information utilities — the core science primitives (CLAUDE.md §2.4, §4).

Classical Fisher information of measurement distributions and quantum Fisher
information (QFI) of pure states, computed two independent ways (fidelity and
generator variance) so they can be cross-checked (mandatory, CLAUDE.md §8).
"""

import numpy as np


def binary_fisher(mean, dmean):
    """Fisher information of a binary (+/-1) outcome with mean m(theta) (E3, CLAUDE.md §2.2).

    F = (d m / d theta)^2 / (1 - m^2).

    mean, dmean may be scalars or numpy arrays (elementwise).
    """
    mean = np.asarray(mean, dtype=float)
    dmean = np.asarray(dmean, dtype=float)
    denom = 1.0 - mean**2
    # Guard the deterministic-outcome limit |m|->1 where F is a 0/0 that must be
    # handled by the caller's model; here we return inf only if numerator != 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        F = dmean**2 / denom
    F = np.where(np.isclose(denom, 0.0),
                 np.where(np.isclose(dmean, 0.0), 0.0, np.inf),
                 F)
    return F if F.ndim else float(F)


def fisher_from_hist(prob_fn, theta, eps=1e-5):
    """Classical Fisher information of a discrete distribution by finite difference.

    F = sum_b (d p_b / d theta)^2 / p_b  (CLAUDE.md §4).

    prob_fn(theta) -> 1-D numpy array of outcome probabilities (sums to 1).
    Central difference for the derivative; probabilities evaluated at `theta`.
    Zero-probability outcomes are skipped (they contribute nothing in the limit).
    """
    p = np.asarray(prob_fn(theta), dtype=float)
    p_plus = np.asarray(prob_fn(theta + eps), dtype=float)
    p_minus = np.asarray(prob_fn(theta - eps), dtype=float)
    dp = (p_plus - p_minus) / (2.0 * eps)
    mask = p > 1e-15
    return float(np.sum(dp[mask] ** 2 / p[mask]))


def qfi_fidelity(state_fn, theta, eps=1e-4):
    """QFI via the finite-difference fidelity, Method A (CLAUDE.md §2.4).

    F_Q ~= 8 (1 - |<psi(theta)|psi(theta+eps)>|) / eps^2.

    state_fn(theta) -> normalized complex state vector.
    """
    psi = state_fn(theta)
    psi_eps = state_fn(theta + eps)
    overlap = abs(np.vdot(psi, psi_eps))
    return 8.0 * (1.0 - overlap) / eps**2


def qfi_variance(chi_state, generator):
    """QFI via generator variance, Method B (CLAUDE.md §2.4).

    F_Q = 4 Var_{|chi>}(G) = 4 (<G^2> - <G>^2),  G = generator (dense operator).

    For a state exp(-i theta G)|chi>, `generator` is exactly G. For the labeled
    depth quantity, pass the fixed-pattern generator G = (1/2) sum_ell s_ell G-tilde_ell.
    """
    Gpsi = generator @ chi_state
    exp_G = np.vdot(chi_state, Gpsi).real
    exp_G2 = np.vdot(Gpsi, Gpsi).real          # <G^2> = <G psi|G psi> for Hermitian G
    return 4.0 * (exp_G2 - exp_G**2)


def qfi_mixed(rho_fn, theta, eps=1e-5, tol=1e-12):
    """Quantum Fisher information of a MIXED state via the SLD formula (CLAUDE.md §2.4 ext).

    F_Q(rho) = 2 sum_{j,k : lam_j+lam_k>0} |<j|drho/dtheta|k>|^2 / (lam_j + lam_k),

    where rho = sum_j lam_j |j><j|. rho_fn(theta) -> density matrix (2^n x 2^n).
    This is the information extractable by the OPTIMAL measurement on the mixed
    state that standard RC produces after forgetting the labels (the 'marginal'
    QFI), and it reduces to the pure-state QFI when rho is pure.
    """
    rho = np.asarray(rho_fn(theta), dtype=complex)
    drho = (np.asarray(rho_fn(theta + eps), dtype=complex)
            - np.asarray(rho_fn(theta - eps), dtype=complex)) / (2 * eps)
    lam, V = np.linalg.eigh(rho)
    lam = np.clip(lam.real, 0.0, None)
    D = V.conj().T @ drho @ V              # drho in eigenbasis
    F = 0.0
    for j in range(len(lam)):
        for k in range(len(lam)):
            s = lam[j] + lam[k]
            if s > tol:
                F += 2.0 * (abs(D[j, k]) ** 2) / s
    return float(F.real if np.iscomplex(F) else F)


def decompose_joint_fisher(joint_fn, theta, eps=1e-6):
    """Decompose the Fisher information of a joint distribution over (outcome b, label s).

    joint_fn(theta) -> 2-D numpy array P[b, s] (nonnegative, sums to 1), where the
    first axis indexes the measurement outcome b and the second the twirl label s.

    Returns (F_joint, F_marg, E_b_cond) with, per the Fisher chain rule (R1, §2.2):

        F_joint = sum_{b,s} (dP_{bs})^2 / P_{bs}
        F_marg  = sum_b     (dP_b)^2   / P_b        [marginal over labels: P_b = sum_s P_{bs}]
        E_b_cond= sum_b P_b sum_s (dP_{s|b})^2 / P_{s|b}   [label info hidden in b-correlation]

    and F_joint == F_marg + E_b_cond exactly. This is the numerical engine behind
    the conservation law: F_marg is what standard RC keeps, E_b_cond is what it
    discards but which remains recoverable from the retained labels.
    """
    P = np.asarray(joint_fn(theta), dtype=float)
    dP = (np.asarray(joint_fn(theta + eps), dtype=float)
          - np.asarray(joint_fn(theta - eps), dtype=float)) / (2 * eps)

    mask = P > 1e-15
    F_joint = float(np.sum(dP[mask] ** 2 / P[mask]))

    Pb = P.sum(axis=1)
    dPb = dP.sum(axis=1)
    mb = Pb > 1e-15
    F_marg = float(np.sum(dPb[mb] ** 2 / Pb[mb]))

    # P(s|b) = P[b,s]/Pb[b]; quotient rule for its theta-derivative
    with np.errstate(divide="ignore", invalid="ignore"):
        Psb = P / Pb[:, None]
        dPsb = (dP * Pb[:, None] - P * dPb[:, None]) / Pb[:, None] ** 2
    valid = (P > 1e-15) & (Pb[:, None] > 1e-15)
    contrib = np.where(valid, dPsb**2 / np.where(valid, Psb, 1.0), 0.0)
    E_b_cond = float(np.sum(Pb[:, None] * contrib))
    return F_joint, F_marg, E_b_cond


def chain_rule_check(F_joint, F_marg, F_cond_term, tol=1e-9):
    """Assert the Fisher chain rule F_joint = F_marg + F_cond_term (R1, CLAUDE.md §2.2).

    Returns the residual |F_joint - (F_marg + F_cond_term)| and raises if > tol.
    """
    residual = abs(F_joint - (F_marg + F_cond_term))
    if residual > tol:
        raise AssertionError(
            f"Fisher chain rule violated: |{F_joint} - ({F_marg}+{F_cond_term})| "
            f"= {residual} > {tol}"
        )
    return residual
