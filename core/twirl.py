"""Randomized-compiling twirl for the sign-label model (CLAUDE.md §2.2, §4).

The twirl is a per-shot sign label s in {+1, -1} on a coherent Z-error: an
X-dressing flips the error sign, since X exp(-i theta Z/2) X = exp(+i theta Z/2).

CRITICAL INVARIANT: labels are RETAINED and returned to the caller. This module
NEVER averages over the labels internally. Marginalization (standard RC) happens
only where the marginal quantity is explicitly wanted, in the caller.
"""

import numpy as np

from .single_qubit_gates import Rz
from .operator_on_state import single_qubit_op


def sample_sign_pattern(L, p, rng=None):
    """Draw a length-L sign pattern s_ell in {+1,-1} with P(+1)=p (CLAUDE.md §4).

    Returns the labels (an integer numpy array of +/-1). The mean sign is
    mu = 2p - 1; p=0.5 is the fully-twirled (mu=0) limit, p=1 is no twirl.
    """
    if rng is None:
        rng = np.random.default_rng()
    draws = rng.random(L)
    return np.where(draws < p, 1, -1).astype(int)


def all_sign_patterns(L):
    """Enumerate all 2^L sign patterns as a (2^L, L) array of +/-1.

    Used for EXACT averages at small L (CLAUDE.md §8: enumerate for L <= ~12).
    """
    idx = np.arange(2**L)
    bits = ((idx[:, None] >> np.arange(L)[::-1]) & 1)
    return np.where(bits == 1, 1, -1)


def pattern_probability(pattern, p):
    """Probability of a given +/-1 sign pattern under i.i.d. P(+1)=p."""
    pattern = np.asarray(pattern)
    n_plus = np.sum(pattern == 1)
    n_minus = np.sum(pattern == -1)
    return p**n_plus * (1.0 - p) ** n_minus


def apply_signed_z_error(state, theta, s, qubit, n):
    """Apply the signed coherent Z-error Rz(s*theta) on `qubit` (1-indexed).

    The label s in {+1,-1} is supplied by the caller and is NOT averaged here
    (see module docstring). Returns the new state vector.
    """
    return single_qubit_op(state, Rz(s * theta), qubit, n)
