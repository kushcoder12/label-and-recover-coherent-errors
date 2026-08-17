"""Multi-qubit static-ZZ crosstalk verification (Oracles MQ1-MQ3).

- MQ1: exact 2-label inversion is a floating-point identity (2 qubits).
- MQ2: exact 3-qubit closed forms match Monte Carlo convergence.
- MQ3: the Fisher matrix of (s1, s1*c2, s1*c3) is diagonal, from the EMPIRICAL
  covariance of actually-sampled labels (catches a label-tracking bug the formula
  alone would not).
- The full-Hamiltonian circuit builders reproduce the closed-form outcome model
  exactly -- the model is being validated against the circuit, never hard-coded
  into it (Sec 0 guardrail).
- Lightweight versions of the exp11-exp13 recovery/cost checks.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.circuits import two_qubit_zz_probe, multi_qubit_zz_probe
from core.multiqubit_zz import build_zz_hamiltonian, joint_correlation_estimator
from core.pauli import pauli

import exp11_twoqubit_zz as exp11
import exp12_threequbit_zz as exp12
import exp13_joint_vs_pairwise_cost as exp13


# ---------------------------------------------------------------------------
# Oracle MQ1 -- 2 qubits, exact identity
# ---------------------------------------------------------------------------
def test_oracle_MQ1_closed_form_identity():
    theta0, kappa12 = 0.10, 0.05
    X = math.sin(theta0) * math.cos(kappa12)
    Y = math.cos(theta0) * math.sin(kappa12)
    theta_rec = (math.asin(X + Y) + math.asin(X - Y)) / 2.0
    kappa_rec = (math.asin(X + Y) - math.asin(X - Y)) / 2.0
    assert abs(theta_rec - theta0) < 1e-12, theta_rec
    assert abs(kappa_rec - kappa12) < 1e-12, kappa_rec


def test_oracle_MQ1_matches_code():
    """The circuit builder + estimator reproduce the same identity from population-
    exact (infinite-shot) records, for a sweep of kappa values."""
    theta0 = 0.10
    for kappa12 in (0.0, 0.02, 0.05, 0.10, 0.15):
        err_theta, err_kappa = exp11.exact_recovery_check(theta0, kappa12)
        assert err_theta < 1e-9, (kappa12, err_theta)
        assert err_kappa < 1e-9, (kappa12, err_kappa)


# ---------------------------------------------------------------------------
# Oracle MQ2 -- 3 qubits, exact forms vs Monte Carlo convergence
# ---------------------------------------------------------------------------
def test_oracle_MQ2_closed_forms_independent():
    """Independently re-derive the 3-qubit closed forms (not by calling exp12's
    helper) and cross-check them against it."""
    theta0, k2, k3 = 0.10, 0.05, 0.03
    X = math.sin(theta0) * math.cos(k2) * math.cos(k3)
    Y2 = math.cos(theta0) * math.sin(k2) * math.cos(k3)
    Y3 = math.cos(theta0) * math.cos(k2) * math.sin(k3)
    X2, Y2b, Y3b = exp12.oracle_mq2_targets(theta0, {2: k2, 3: k3})
    assert abs(X - X2) < 1e-12 and abs(Y2 - Y2b) < 1e-12 and abs(Y3 - Y3b) < 1e-12


def test_oracle_MQ2_monte_carlo_convergence():
    rng = np.random.default_rng(11)
    theta0, kappa = 0.10, {2: 0.05, 3: 0.03}
    shots = 60000
    X_exact, Y2_exact, Y3_exact = exp12.oracle_mq2_targets(theta0, kappa)
    records = exp12.sample_shots(theta0, kappa, shots, rng)
    X_mc, Y2_mc, Y3_mc = exp12.empirical_XY(records)
    tol = 6.0 / math.sqrt(shots)
    assert abs(X_mc - X_exact) < tol
    assert abs(Y2_mc - Y2_exact) < tol
    assert abs(Y3_mc - Y3_exact) < tol


# ---------------------------------------------------------------------------
# Oracle MQ3 -- Fisher diagonality from empirical label statistics
# ---------------------------------------------------------------------------
def test_oracle_MQ3_fisher_diagonal_from_data():
    rng = np.random.default_rng(12)
    theta0, kappa = 0.10, {2: 0.05, 3: 0.03}
    records = exp12.sample_shots(theta0, kappa, 60000, rng)
    F = exp12.empirical_fisher(records)
    tol = 6.0 / math.sqrt(60000)
    assert np.max(np.abs(np.diag(F) - 1.0)) < tol
    assert np.max(np.abs(F - np.diag(np.diag(F)))) < tol


# ---------------------------------------------------------------------------
# Circuit builders reproduce the closed-form model exactly (Sec 0 guardrail:
# the effective-angle formula is what is being VALIDATED, never hard-coded).
# ---------------------------------------------------------------------------
def test_two_qubit_probe_matches_theory():
    for theta0 in (0.0, 0.1, -0.2):
        for kappa12 in (0.0, 0.05, -0.12):
            for s1 in (1, -1):
                for c2 in (1, -1):
                    y = two_qubit_zz_probe(theta0, kappa12, s1, c2)
                    expected = s1 * math.sin(theta0 + kappa12 * c2)
                    assert abs(y - expected) < 1e-10, (theta0, kappa12, s1, c2, y, expected)


def test_multi_qubit_probe_matches_theory_three_neighbors():
    theta0 = 0.10
    kappa = {2: 0.05, 3: -0.03, 4: 0.07}
    for s1 in (1, -1):
        for c2 in (1, -1):
            for c3 in (1, -1):
                for c4 in (1, -1):
                    y = multi_qubit_zz_probe(theta0, kappa, s1, {2: c2, 3: c3, 4: c4})
                    expected = s1 * math.sin(theta0 + kappa[2] * c2 + kappa[3] * c3 + kappa[4] * c4)
                    assert abs(y - expected) < 1e-10


def test_two_qubit_probe_is_the_one_neighbor_case():
    """two_qubit_zz_probe must literally be the Q=1 case of multi_qubit_zz_probe."""
    theta0, kappa12, s1, c2 = 0.12, -0.04, -1, 1
    a = two_qubit_zz_probe(theta0, kappa12, s1, c2)
    b = multi_qubit_zz_probe(theta0, {2: kappa12}, s1, {2: c2})
    assert a == b


def test_zz_hamiltonian_hermitian_and_matches_hand_derivation():
    H = build_zz_hamiltonian([0.2, -0.1], [[0.0, 0.05], [0.05, 0.0]], 2)
    assert H.shape == (4, 4)
    assert np.allclose(H, H.conj().T)
    Z = pauli("Z")
    expected = (0.2 / 2.0) * np.kron(Z, np.eye(2)) + (-0.1 / 2.0) * np.kron(np.eye(2), Z) \
        + (0.05 / 4.0) * np.kron(Z, Z)
    assert np.allclose(H, expected)


# ---------------------------------------------------------------------------
# Lightweight recovery / cost-scaling sanity (fast versions of exp11-13)
# ---------------------------------------------------------------------------
def test_exp11_style_recovery_small():
    rng = np.random.default_rng(21)
    theta0, kappa12 = 0.10, 0.06
    recs = exp11.sample_shots(theta0, kappa12, 15000, rng=rng)
    fit = joint_correlation_estimator(recs, 1, [2], B=200, seed=21)
    assert abs(fit["theta0_hat"] - theta0) < 0.03
    assert abs(fit["kappa_hat"][2] - kappa12) < 0.03


def test_exp12_style_joint_recovery_small():
    rng = np.random.default_rng(22)
    theta0, kappa = 0.10, {2: 0.05, 3: 0.03}
    recs = exp12.sample_shots(theta0, kappa, 30000, rng)
    fit = joint_correlation_estimator(recs, 1, [2, 3], B=200, seed=22)
    assert abs(fit["theta0_hat"] - theta0) < 0.03
    assert abs(fit["kappa_hat"][2] - kappa[2]) < 0.03
    assert abs(fit["kappa_hat"][3] - kappa[3]) < 0.03


def test_exp13_style_cost_direction():
    """Coarse, fast version of the flat-vs-linear check: joint(Q=4)/joint(Q=1) stays
    well under the ~4x growth that a linear law would predict, while a single
    dedicated pairwise experiment costs about the same as joint(Q=1)."""
    rng = np.random.default_rng(23)
    M, trials = 1500, 80
    var_pairwise = exp13.pairwise_kappa_variance(M, trials, rng)
    var_joint_q1 = exp13.joint_kappa_variance(1, M, trials, rng)
    var_joint_q4 = exp13.joint_kappa_variance(4, M, trials, rng)
    # same physics at Q=1 (one neighbor either way) -> variances should be comparable
    assert 0.4 < var_joint_q1 / var_pairwise < 2.5
    # Fisher orthogonality -> Q=4's per-parameter variance should NOT blow up vs Q=1
    assert var_joint_q4 / var_joint_q1 < 2.0


def run_all():
    tests = [
        test_oracle_MQ1_closed_form_identity,
        test_oracle_MQ1_matches_code,
        test_oracle_MQ2_closed_forms_independent,
        test_oracle_MQ2_monte_carlo_convergence,
        test_oracle_MQ3_fisher_diagonal_from_data,
        test_two_qubit_probe_matches_theory,
        test_multi_qubit_probe_matches_theory_three_neighbors,
        test_two_qubit_probe_is_the_one_neighbor_case,
        test_zz_hamiltonian_hermitian_and_matches_hand_derivation,
        test_exp11_style_recovery_small,
        test_exp12_style_joint_recovery_small,
        test_exp13_style_cost_direction,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_multiqubit_zz: ALL PASS")


if __name__ == "__main__":
    run_all()
