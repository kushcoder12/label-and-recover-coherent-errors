"""Device-representative noise profile.

A single dataclass `NoiseProfile` parametrizes the full standard channel set, and
helper functions build the Kraus operators and apply them either exactly
(density-matrix mode, small n) or by sampling (trajectory mode, larger n).

Conceptual anchor: RC's per-shot twirl sorts errors -- coherent
(Hamiltonian) errors become label-correlated (recoverable from labels), while
stochastic Pauli errors are label-independent (they sit in the marginal). This
module supplies both sectors: coherent errors are injected as unitaries in the
circuit builders (circuits.py), incoherent Pauli/damping/readout errors live here.

Built on the existing Kraus machinery in core/quantum_errors.py (numpy only).
Conventions: 1-indexed qubits, big-endian, matching operator_on_state.py.
"""

from dataclasses import dataclass
from math import inf

import numpy as np

from .quantum_errors import (
    kraus_amplitude_damping,
    kraus_depolarizing,
    kraus_two_qubit_depolarizing,
    _check_completeness,
    apply_single_qubit_channel_to_dm,
    apply_channel_density_matrix,
    apply_channel_stochastic,
    apply_two_qubit_channel_stochastic,
)

_I = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ===========================================================================
# Profile
# ===========================================================================
@dataclass
class NoiseProfile:
    """Per-layer physical noise parameters.

    dt     : layer duration (same time units as T1, T2).
    T1     : relaxation time (inf = off).
    T2     : coherence time (inf = off); require T2 <= 2 T1.
    r_1q   : average single-qubit gate error -> depolarizing strength.
    r_2q   : average two-qubit gate error   -> depolarizing strength.
    e01    : readout P(measure 1 | true 0).
    e10    : readout P(measure 0 | true 1).
    p_plus : thermal excitation probability (optional; off by default).

    Coherent (ground-truth signal) parameters live with the circuit, not here.
    """
    dt: float
    T1: float = inf
    T2: float = inf
    r_1q: float = 0.0
    r_2q: float = 0.0
    e01: float = 0.0
    e10: float = 0.0
    p_plus: float = 0.0

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.T1 <= 0 or self.T2 <= 0:
            raise ValueError("T1, T2 must be positive")
        if self.T2 > 2 * self.T1 + 1e-12:
            raise ValueError(f"T2={self.T2} violates T2 <= 2 T1 = {2 * self.T1}")
        for name in ("r_1q", "r_2q", "e01", "e10", "p_plus"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} must lie in [0, 1]")

    def is_noiseless(self):
        return (self.T1 == inf and self.T2 == inf and self.r_1q == 0.0
                and self.r_2q == 0.0 and self.e01 == 0.0 and self.e10 == 0.0
                and self.p_plus == 0.0)


# ===========================================================================
# Kraus builders
# ===========================================================================
def amplitude_damping_kraus(profile):
    """T1 amplitude damping, p1 = 1 - exp(-dt/T1).

    Returns [I, 0]-equivalent (no-op) when T1 is infinite.
    """
    if profile.T1 == inf:
        return [_I.copy(), np.zeros((2, 2), dtype=complex)]
    return kraus_amplitude_damping(profile.T1, profile.dt)


def dephasing_kraus(profile):
    """Pure dephasing (phase-flip form) from T1, T2.

    1/Tphi = 1/T2 - 1/(2 T1)  (require T2 <= 2 T1);  lam = (1 - exp(-dt/Tphi))/2.
    K0 = sqrt(1-lam) I,  K1 = sqrt(lam) Z.  Coherence factor (1-2 lam) = exp(-dt/Tphi),
    so combined with amplitude damping the total off-diagonal decays as exp(-dt/T2).
    """
    if profile.T2 == inf:
        return [_I.copy(), np.zeros((2, 2), dtype=complex)]
    inv_tphi = 1.0 / profile.T2 - 1.0 / (2.0 * profile.T1)
    if inv_tphi <= 0:                      # T2 == 2 T1 exactly -> no pure dephasing
        return [_I.copy(), np.zeros((2, 2), dtype=complex)]
    tphi = 1.0 / inv_tphi
    lam = 0.5 * (1.0 - np.exp(-profile.dt / tphi))
    return [np.sqrt(1.0 - lam) * _I, np.sqrt(lam) * _Z]


def _depol_strength_1q(r):
    """Depolarizing strength giving average gate infidelity r (single qubit).

    Analytic (exact for the depolarizing channel): 1 - F_avg = 2 p / 3, so p = 1.5 r.
    Verified numerically by avg_gate_infidelity in the unit tests (N4).
    """
    return min(1.5 * r, 1.0)


def _depol_strength_2q(r):
    """Depolarizing strength giving average gate infidelity r (two qubits).

    Analytic (exact): 1 - F_avg = 4 p / 5, so p = 1.25 r.
    """
    return min(1.25 * r, 1.0)


def depolarizing_kraus_1q(profile):
    """Single-qubit depolarizing Kraus set calibrated to profile.r_1q."""
    return kraus_depolarizing(_depol_strength_1q(profile.r_1q))


def depolarizing_kraus_2q(profile):
    """Two-qubit depolarizing Kraus set calibrated to profile.r_2q."""
    return kraus_two_qubit_depolarizing(_depol_strength_2q(profile.r_2q))


def readout_confusion(profile):
    """Asymmetric readout confusion matrix R.

    Columns = true state, rows = measured state:
        R = [[1 - e01,   e10  ],
             [  e01  , 1 - e10]]
    P_meas = R @ P_true.
    """
    e01, e10 = profile.e01, profile.e10
    return np.array([[1 - e01, e10], [e01, 1 - e10]], dtype=float)


# ===========================================================================
# Average gate fidelity (for the depolarizing calibration check, N4)
# ===========================================================================
def avg_gate_infidelity(kraus_list, d):
    """1 - F_avg of an error channel (ideal = identity) from its Kraus operators.

    F_pro = (1/d^2) sum_k |Tr K_k|^2 ;  F_avg = (d F_pro + 1)/(d+1).
    """
    F_pro = sum(abs(np.trace(K)) ** 2 for K in kraus_list) / d**2
    F_avg = (d * F_pro + 1.0) / (d + 1.0)
    return float(1.0 - F_avg)


# ===========================================================================
# Trace-preservation validator (N1)
# ===========================================================================
def is_trace_preserving(kraus_list, tol=1e-12):
    """True iff sum_k K_k^dagger K_k = I."""
    return _check_completeness(kraus_list, tol=tol)


# ===========================================================================
# Applying the incoherent layer
# ===========================================================================
def _nontrivial(kraus_list):
    """True if the Kraus set is not the identity channel (worth applying)."""
    if len(kraus_list) == 2 and np.allclose(kraus_list[1], 0.0):
        return False
    if len(kraus_list) >= 1 and np.allclose(kraus_list[0], np.eye(kraus_list[0].shape[0])) \
            and all(np.allclose(K, 0.0) for K in kraus_list[1:]):
        return False
    return True


def apply_incoherent_layer(rho_or_state, profile, n, mode="dm", pairs_2q=(), rng=None):
    """Apply one incoherent noise layer.

    Per qubit, in composition order depolarizing -> dephasing -> amplitude damping.
    Two-qubit depolarizing is applied on each pair in `pairs_2q` (1-indexed tuples).

    mode="dm"          : rho_or_state is a 2^n x 2^n density matrix; exact.
    mode="trajectory"  : rho_or_state is a 2^n statevector; one Kraus sampled per
                         channel (uses rng). Returns (state, outcomes) list; else rho.
    """
    if profile.is_noiseless():
        return rho_or_state if mode == "dm" else (rho_or_state, [])

    K_depol1 = depolarizing_kraus_1q(profile) if profile.r_1q > 0 else None
    K_deph = dephasing_kraus(profile) if profile.T2 != inf else None
    K_amp = amplitude_damping_kraus(profile) if profile.T1 != inf else None
    K_depol2 = depolarizing_kraus_2q(profile) if profile.r_2q > 0 else None

    if mode == "dm":
        rho = rho_or_state
        for q in range(1, n + 1):
            for K in (K_depol1, K_deph, K_amp):
                if K is not None and _nontrivial(K):
                    rho = apply_single_qubit_channel_to_dm(rho, K, q, n)
        if K_depol2 is not None and _nontrivial(K_depol2):
            for (q1, q2) in pairs_2q:
                rho = _apply_2q_channel_to_dm(rho, K_depol2, q1, q2, n)
        return rho

    elif mode == "trajectory":
        if rng is None:
            rng = np.random.default_rng()
        state = rho_or_state
        outcomes = []
        for q in range(1, n + 1):
            for K in (K_depol1, K_deph, K_amp):
                if K is not None and _nontrivial(K):
                    state, k = apply_channel_stochastic(state, K, q, n)
                    outcomes.append(k)
        if K_depol2 is not None and _nontrivial(K_depol2):
            for (q1, q2) in pairs_2q:
                state, k = apply_two_qubit_channel_stochastic(state, K_depol2, q1, q2, n)
                outcomes.append(k)
        return state, outcomes

    raise ValueError(f"unknown mode {mode!r}; expected 'dm' or 'trajectory'")


def _apply_2q_channel_to_dm(rho, kraus4, q1, q2, n):
    """Apply a 4x4 two-qubit Kraus channel on qubits (q1,q2) of an n-qubit DM."""
    if n == 2:
        return apply_channel_density_matrix(rho, kraus4)
    embeds = [_embed_two_qubit_op(K, q1, q2, n) for K in kraus4]
    out = np.zeros_like(rho)
    for E in embeds:
        out += E @ rho @ E.conj().T
    return out


def _embed_two_qubit_op(op4, q1, q2, n):
    """Full 2^n operator for a 4x4 op acting on ordered qubits (q1,q2), big-endian."""
    # Build via basis permutation: index bits, apply op on the (q1,q2) 2-bit subspace.
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    b1, b2 = n - q1, n - q2                    # bit positions (0 = LSB)
    for r in range(dim):
        a1 = (r >> b1) & 1
        a2 = (r >> b2) & 1
        row_sub = 2 * a1 + a2
        for col_sub in range(4):
            c1, c2 = col_sub >> 1, col_sub & 1
            cval = op4[row_sub, col_sub]
            if cval == 0:
                continue
            c = r
            c = (c & ~(1 << b1)) | (c1 << b1)
            c = (c & ~(1 << b2)) | (c2 << b2)
            out[r, c] += cval
    return out


# ===========================================================================
# Readout
# ===========================================================================
def apply_readout(probs_or_bits, profile, mode="probs", rng=None):
    """Apply asymmetric single-qubit readout error.

    mode="probs": probs_or_bits is a length-2 array [P(0), P(1)] (true); returns
                  measured [P(0), P(1)] = R @ P_true.
    mode="bits" : probs_or_bits is an array of measured-basis bits in {0,1};
                  flips each 0->1 with prob e01 and 1->0 with prob e10.
    """
    R = readout_confusion(profile)
    if mode == "probs":
        p = np.asarray(probs_or_bits, dtype=float)
        return R @ p
    elif mode == "bits":
        if rng is None:
            rng = np.random.default_rng()
        bits = np.asarray(probs_or_bits).astype(int)
        u = rng.random(bits.shape)
        flip0 = (bits == 0) & (u < profile.e01)
        flip1 = (bits == 1) & (u < profile.e10)
        out = bits.copy()
        out[flip0] = 1
        out[flip1] = 0
        return out
    raise ValueError(f"unknown mode {mode!r}; expected 'probs' or 'bits'")
