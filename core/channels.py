"""Noise channels for the depth-saturation study (CLAUDE.md §4, Phase 5).

Thin, spec-named layer over the existing core/quantum_errors.py Kraus engine.
Every Kraus set satisfies completeness sum_k K_k^dagger K_k = I (trace preserving),
checked by `is_trace_preserving`.

Convention: 1-indexed qubits, big-endian, matching operator_on_state.py.
"""

import numpy as np

from .quantum_errors import (
    _check_completeness,
    apply_channel_stochastic,
    apply_channel_density_matrix,
    apply_single_qubit_channel_to_dm,
    state_to_density_matrix,
)

_I = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def dephasing_kraus(gamma_z, dt):
    """Pure-dephasing (T2 / phase-flip) Kraus set with rate gamma_z over time dt (CLAUDE.md §4).

    Symmetric dephasing rho -> (1-q) rho + q Z rho Z, with
        q = (1 - exp(-2 gamma_z dt)) / 2,
    so the off-diagonal coherence is multiplied by (1-2q) = exp(-2 gamma_z dt) per
    application (T2-type decay of the phase that carries the coherent-error signal).

    Returns [sqrt(1-q) I, sqrt(q) Z].
    """
    assert gamma_z >= 0 and dt > 0
    q = 0.5 * (1.0 - np.exp(-2.0 * gamma_z * dt))
    return [np.sqrt(1.0 - q) * _I, np.sqrt(q) * _Z]


def amplitude_damping_kraus(gamma, dt):
    """Amplitude-damping (T1) Kraus set with rate gamma over time dt (CLAUDE.md §4).

    p = 1 - exp(-gamma dt) is the |1>->|0> decay probability per step.
    K0 = [[1,0],[0,sqrt(1-p)]],  K1 = [[0,sqrt(p)],[0,0]].
    """
    assert gamma >= 0 and dt > 0
    p = 1.0 - np.exp(-gamma * dt)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [K0, K1]


def is_trace_preserving(kraus_list, tol=1e-12):
    """True iff sum_k K_k^dagger K_k = I (completeness / unit trace)."""
    return _check_completeness(kraus_list, tol=tol)


def apply_kraus_stochastic(state, kraus_list, qubit, n_qubits):
    """Quantum-trajectory application of a single-qubit Kraus channel (port of the
    existing engine). Returns (new_state, k_index). Labels/branches are returned,
    never silently averaged."""
    return apply_channel_stochastic(state, kraus_list, qubit, n_qubits)


def apply_kraus_dm(rho, kraus_list, qubit=None, n_qubits=None):
    """Deterministic density-matrix application rho -> sum_k K_k rho K_k^dagger.

    If qubit/n_qubits are given, the 2x2 Kraus operators act on that qubit of an
    n-qubit density matrix; otherwise kraus_list must match rho's dimension.
    """
    if qubit is None:
        return apply_channel_density_matrix(rho, kraus_list)
    return apply_single_qubit_channel_to_dm(rho, kraus_list, qubit, n_qubits)
