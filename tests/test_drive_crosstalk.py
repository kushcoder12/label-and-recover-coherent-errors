"""Drive-induced leakage recovery: the third label p_j (Oracles DC1-DC3).

- DC1: the three exact correlations (X, Y_c, Y_p) match brute-force averaging over the
  four (c_j, p_j) combinations, including the leading minus sign on Y_p.
- DC2: the full stratified recovery reproduces the true (theta_1_0, kappa_1j, phi_1j)
  to floating-point precision from population-exact (infinite-shot) records -- the
  primary correctness gate (DRIVE_CROSSTALK_SPEC Sec 3).
- DC3: the contamination this fix solves -- a naive two-label (s_1, c_j) estimator that
  never varied p_j recovers a biased (Theta, K), not the true (theta_1_0, kappa_1j).
  This is what justifies the whole exercise, not an afterthought.
- The circuit builder (core.circuits.three_label_probe) reproduces the closed-form
  outcome model exactly -- the model is being validated against the circuit, never
  hard-coded into it (same discipline as test_multiqubit_zz.py).
- Lightweight versions of the exp14-exp16 recovery/variance checks.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from core.circuits import three_label_probe
from core.drive_crosstalk import n_real_pulses, apply_padding, sample_padding_label
from core.multiqubit_zz import stratified_three_label_estimator

import exp14_drive_crosstalk_oracle as exp14
import exp15_drive_crosstalk_noisy as exp15

THETA0, KAPPA, PHI = 0.10, 0.05, 0.02


def _combined(theta0=THETA0, kappa=KAPPA, phi=PHI):
    return theta0 + 1.5 * phi, kappa - 0.5 * phi, phi


# ---------------------------------------------------------------------------
# n_real_pulses -- the exact table (Sec 1), checked before anything downstream
# trusts it.
# ---------------------------------------------------------------------------
def test_n_real_pulses_table():
    assert n_real_pulses(1, 1) == 0
    assert n_real_pulses(1, -1) == 2
    assert n_real_pulses(-1, 1) == 1
    assert n_real_pulses(-1, -1) == 3


def test_apply_padding_is_identity_on_the_logical_state():
    """X_pi . X_pi = -I: padding must not change the neighbor's prepared state,
    only the real-pulse count (which n_real_pulses tracks separately)."""
    n = 2
    for c in (1, -1):
        from core.multiqubit_zz import prepare_neighbor_state
        from core.single_qubit_gates import Hadamard
        from core.operator_on_state import single_qubit_op
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        state = single_qubit_op(state, Hadamard(), 1, n)
        state = prepare_neighbor_state(state, c, 2, n)
        padded = apply_padding(state.copy(), -1, 2, n)
        unpadded = apply_padding(state.copy(), 1, 2, n)
        assert np.allclose(padded, state), (c, padded, state)
        assert np.array_equal(unpadded, state)


def test_sample_padding_label_is_uniform_pm1():
    rng = np.random.default_rng(0)
    draws = [sample_padding_label(rng) for _ in range(2000)]
    assert set(draws) <= {1, -1}
    assert abs(np.mean(draws)) < 0.06


# ---------------------------------------------------------------------------
# Oracle DC1 -- exact correlations match brute-force averaging (Sec 2.2, 3)
# ---------------------------------------------------------------------------
def test_oracle_DC1_correlations_match_bruteforce():
    Theta, K, Phi = _combined()
    X = Y_c = Y_p = 0.0
    for c in (1, -1):
        for p in (1, -1):
            m = math.sin(Theta + K * c - Phi * p)
            X += m
            Y_c += c * m
            Y_p += p * m
    X, Y_c, Y_p = X / 4, Y_c / 4, Y_p / 4

    assert abs(X - math.sin(Theta) * math.cos(K) * math.cos(Phi)) < 1e-12
    assert abs(Y_c - math.cos(Theta) * math.sin(K) * math.cos(Phi)) < 1e-12
    # leading minus sign on Y_p -- not a typo, checked against brute force directly.
    assert abs(Y_p - (-math.cos(Theta) * math.cos(K) * math.sin(Phi))) < 1e-12


def test_oracle_DC1_reference_values():
    Theta, K, Phi = _combined()
    X = math.sin(Theta) * math.cos(K) * math.cos(Phi)
    Y_c = math.cos(Theta) * math.sin(K) * math.cos(Phi)
    Y_p = -math.cos(Theta) * math.cos(K) * math.sin(Phi)
    assert abs(X - 0.129505) < 5e-6
    assert abs(Y_c - 0.039644) < 5e-6
    assert abs(Y_p - (-0.019814)) < 5e-6


# ---------------------------------------------------------------------------
# Circuit builder reproduces the closed-form model exactly (never hard-coded
# into the builder -- checked here, same discipline as multi_qubit_zz_probe).
# ---------------------------------------------------------------------------
def test_three_label_probe_matches_closed_form():
    for theta0 in (0.0, 0.10, -0.15):
        for kappa in (0.0, 0.05, -0.03):
            for phi in (0.0, 0.02, -0.04):
                Theta, K, Phi = theta0 + 1.5 * phi, kappa - 0.5 * phi, phi
                for s1 in (1, -1):
                    for c in (1, -1):
                        for p in (1, -1):
                            y = three_label_probe(theta0, kappa, phi, s1, c, p)
                            expected = s1 * math.sin(Theta + K * c - Phi * p)
                            assert abs(y - expected) < 1e-10, (theta0, kappa, phi, s1, c, p)


def test_three_label_probe_reduces_to_two_qubit_probe_when_phi_zero():
    from core.circuits import two_qubit_zz_probe
    for s1 in (1, -1):
        for c in (1, -1):
            for p in (1, -1):
                a = three_label_probe(0.10, 0.05, 0.0, s1, c, p)
                b = two_qubit_zz_probe(0.10, 0.05, s1, c)
                assert abs(a - b) < 1e-12


# ---------------------------------------------------------------------------
# Oracle DC2 -- full stratified recovery, noiseless, exact (Sec 3)
# ---------------------------------------------------------------------------
def _population_records(theta0, kappa, phi):
    records = []
    for s1 in (1, -1):
        for c in (1, -1):
            for p in (1, -1):
                y = three_label_probe(theta0, kappa, phi, s1, c, p)
                records.append((s1, c, p, y))
    return records


def test_oracle_DC2_stratified_recovery_exact():
    records = _population_records(THETA0, KAPPA, PHI)
    fit = stratified_three_label_estimator(records, B=2)
    assert abs(fit["theta_1_0_hat"] - THETA0) < 1e-9
    assert abs(fit["kappa_1j_hat"] - KAPPA) < 1e-9
    assert abs(fit["phi_1j_hat"] - PHI) < 1e-9
    assert fit["K_disagreement"] < 1e-9


def test_oracle_DC2_matches_spec_reference_values():
    Theta, K, Phi = _combined()
    assert abs(Theta - 0.13) < 1e-12
    assert abs(K - 0.04) < 1e-12
    assert abs(Phi - 0.02) < 1e-12


def test_oracle_DC2_requires_both_strata():
    only_plus = [(1, 1, 1, three_label_probe(THETA0, KAPPA, PHI, 1, 1, 1))]
    try:
        stratified_three_label_estimator(only_plus, B=2)
        assert False, "expected ValueError for a single-stratum record set"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Oracle DC3 -- the contamination this fix solves (Sec 3), demonstrated, not
# merely asserted: a naive (s_1, c_j) estimator that never varied p_j recovers
# a biased (Theta, K) instead of the true (theta_1_0, kappa_1j).
# ---------------------------------------------------------------------------
def test_oracle_DC3_naive_two_label_estimator_is_biased():
    from core.multiqubit_zz import _point_estimate
    Theta, K, _ = _combined()

    s = np.array([1, 1, -1, -1], dtype=float)
    c = np.array([1, -1, 1, -1], dtype=float)
    b = np.array([three_label_probe(THETA0, KAPPA, PHI, int(si), int(ci), 1)
                 for si, ci in zip((1, 1, -1, -1), (1, -1, 1, -1))], dtype=float)
    # average the two p-strata together, as an un-padded 2-label experiment would
    b2 = np.array([0.5 * (three_label_probe(THETA0, KAPPA, PHI, int(si), int(ci), 1)
                          + three_label_probe(THETA0, KAPPA, PHI, int(si), int(ci), -1))
                  for si, ci in zip((1, 1, -1, -1), (1, -1, 1, -1))])

    naive_theta, naive_kappa = _point_estimate(s, {2: c}, b2, [2])
    naive_kappa = naive_kappa[2]

    assert abs(naive_theta - 0.129974) < 5e-6
    assert abs(naive_kappa - 0.039992) < 5e-6
    # the naive estimate lands near (Theta, K), not near the true (theta_1_0, kappa_1j)
    assert abs(naive_theta - Theta) < 1e-4
    assert abs(naive_theta - THETA0) > 0.02
    assert abs(naive_kappa - K) < 1e-4
    assert abs(naive_kappa - KAPPA) > 0.008


def test_oracle_DC3_matches_exp14_helper():
    """exp14 reproduces this same contamination; both must agree (not two competing
    implementations of the same check)."""
    naive_theta, naive_kappa, Theta, K = exp14.oracle_dc3_naive_contamination()
    assert abs(naive_theta - Theta) < 1e-4
    assert abs(naive_kappa - K) < 1e-4
    assert abs(naive_theta - THETA0) > 0.02


# ---------------------------------------------------------------------------
# Lightweight recovery / variance sanity (fast versions of exp14-16)
# ---------------------------------------------------------------------------
def test_finite_shot_recovery_small():
    rng = np.random.default_rng(31)
    recs = exp14.sample_shots(THETA0, KAPPA, PHI, 20000, rng=rng)
    fit = stratified_three_label_estimator(recs, B=200, seed=31)
    assert abs(fit["theta_1_0_hat"] - THETA0) < 0.03
    assert abs(fit["kappa_1j_hat"] - KAPPA) < 0.04
    assert abs(fit["phi_1j_hat"] - PHI) < 0.04


def test_noisy_recovery_stays_unbiased_small():
    rng = np.random.default_rng(32)
    profile = exp15.device_profile()
    v = exp15.measure_visibility(profile)
    recs = exp15.sample_shots(THETA0, KAPPA, PHI, 20000, noise=profile, v=v, rng=rng)
    fit = stratified_three_label_estimator(recs, B=200, seed=32)
    assert abs(fit["theta_1_0_hat"] - THETA0) < 0.04
    assert abs(fit["kappa_1j_hat"] - KAPPA) < 0.05
    assert abs(fit["phi_1j_hat"] - PHI) < 0.05


def test_variance_decomposition_relation_small():
    """Var(theta_1_0_hat) ~= Var(Theta_hat) + 2.25*Var(Phi_hat) (Sec 2.5), measured
    from a small Monte Carlo batch -- coarse tolerance, exp16 does the real check."""
    rng = np.random.default_rng(33)
    trials, M = 120, 4000
    theta_ests, phi_ests = [], []
    for _ in range(trials):
        recs = exp14.sample_shots(THETA0, KAPPA, PHI, M, rng=rng)
        fit = stratified_three_label_estimator(recs, B=2, seed=int(rng.integers(1, 1 << 30)))
        theta_ests.append(fit["theta_1_0_hat"])
        phi_ests.append(fit["phi_1j_hat"])
    var_theta = float(np.var(theta_ests, ddof=1))
    var_phi = float(np.var(phi_ests, ddof=1))
    # Theta_hat = theta_1_0_hat + 1.5*phi_1j_hat by construction (Sec 2.4); recompute
    # its variance directly rather than assuming independence holds at this trial count.
    Theta_ests = [t + 1.5 * p for t, p in zip(theta_ests, phi_ests)]
    var_Theta = float(np.var(Theta_ests, ddof=1))
    predicted = var_Theta + 2.25 * var_phi
    ratio = var_theta / predicted
    assert 0.5 < ratio < 2.0, (var_theta, predicted, ratio)


def run_all():
    tests = [
        test_n_real_pulses_table,
        test_apply_padding_is_identity_on_the_logical_state,
        test_sample_padding_label_is_uniform_pm1,
        test_oracle_DC1_correlations_match_bruteforce,
        test_oracle_DC1_reference_values,
        test_three_label_probe_matches_closed_form,
        test_three_label_probe_reduces_to_two_qubit_probe_when_phi_zero,
        test_oracle_DC2_stratified_recovery_exact,
        test_oracle_DC2_matches_spec_reference_values,
        test_oracle_DC2_requires_both_strata,
        test_oracle_DC3_naive_two_label_estimator_is_biased,
        test_oracle_DC3_matches_exp14_helper,
        test_finite_shot_recovery_small,
        test_noisy_recovery_stays_unbiased_small,
        test_variance_decomposition_relation_small,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_drive_crosstalk: ALL PASS")


if __name__ == "__main__":
    run_all()
