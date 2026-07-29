"""Phase 2 verification (CLAUDE.md §5, Oracle O4).

- qfi_fidelity and qfi_variance agree to 1e-3 on Rz(theta)|+> (both give F_Q = 1).
- Fisher chain rule F_joint = F_marg + E_b[F_{s|b}] holds numerically on a random
  2-outcome / 2-label joint distribution.

Runnable directly or via pytest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.single_qubit_gates import Rz
from core.pauli import pauli
from core.fisher import qfi_fidelity, qfi_variance, chain_rule_check


PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)


def _rz_on_plus(theta):
    return Rz(theta) @ PLUS


def test_qfi_methods_agree_O4():
    """Oracle O4: Rz(theta)|+> has F_Q = 1 by both methods."""
    G = pauli("Z") / 2.0                      # generator of Rz(theta) is Z/2
    for theta in [0.0, 0.3, 1.0, 2.0]:
        chi = _rz_on_plus(theta)
        fa = qfi_fidelity(_rz_on_plus, theta)
        fb = qfi_variance(chi, G)
        assert abs(fa - 1.0) < 1e-3, f"Method A F_Q={fa} != 1 at theta={theta}"
        assert abs(fb - 1.0) < 1e-12, f"Method B F_Q={fb} != 1 at theta={theta}"
        assert abs(fa - fb) < 1e-3, f"A vs B disagree ({fa} vs {fb}) at theta={theta}"


def _random_joint(seed=7):
    """Return P(theta) -> (2,2) array indexed [b, s], smooth and normalized."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2, 2))
    c = rng.normal(size=(2, 2))
    d = rng.normal(size=(2, 2))

    def P(theta):
        f = np.exp(a + c * theta + d * theta**2)
        return f / f.sum()

    return P


def _fisher_pieces(P, theta, eps=1e-6):
    """Compute (F_joint, F_marg_over_b, E_b[F_{s|b}]) by finite difference."""
    p = P(theta)
    dp = (P(theta + eps) - P(theta - eps)) / (2 * eps)     # d/dtheta of joint [b,s]

    # Joint Fisher: sum over (b,s) of (dP)^2 / P
    F_joint = np.sum(dp**2 / p)

    # Marginal over b: p_b = sum_s P(b,s)
    pb = p.sum(axis=1)
    dpb = dp.sum(axis=1)
    F_marg = np.sum(dpb**2 / pb)

    # Conditional term E_b[F_{s|b}]: P(s|b) = P(b,s)/p_b
    Psb = p / pb[:, None]
    dPsb = (dp * pb[:, None] - p * dpb[:, None]) / pb[:, None] ** 2   # quotient rule
    F_cond_per_b = np.sum(dPsb**2 / Psb, axis=1)          # Fisher of s given each b
    E_b = np.sum(pb * F_cond_per_b)
    return F_joint, F_marg, E_b


def test_chain_rule_random():
    for seed in [1, 7, 42, 99]:
        P = _random_joint(seed)
        for theta in [-0.5, 0.0, 0.4, 1.1]:
            Fj, Fm, Eb = _fisher_pieces(P, theta)
            res = chain_rule_check(Fj, Fm, Eb, tol=1e-5)
            assert res < 1e-5, f"chain rule residual {res} at seed={seed}, theta={theta}"


def run_all():
    for t in [test_qfi_methods_agree_O4, test_chain_rule_random]:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_fisher_identities: ALL PASS")


if __name__ == "__main__":
    run_all()
