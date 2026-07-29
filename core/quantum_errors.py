"""
Quantum error channels for superconducting qubit simulation.

All channels are implemented as Kraus operators (pure numpy, no external libs).
Every Kraus set satisfies the completeness relation sum_k K_k† K_k = I up to
machine precision, which guarantees trace preservation.

Qubit convention: 1-indexed, matching operator_on_state.py.
"""

import numpy as np

from .operator_on_state import single_qubit_op, two_qubit_op

# ---------------------------------------------------------------------------
# Pauli building blocks
# ---------------------------------------------------------------------------
_I  = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_SP = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma_plus  |1><0|
_SM = np.array([[0, 1], [0, 0]], dtype=complex)   # sigma_minus |0><1|


# ---------------------------------------------------------------------------
# Completeness validator (used internally)
# ---------------------------------------------------------------------------
def _check_completeness(kraus_list, tol=1e-10):
    S = sum(K.conj().T @ K for K in kraus_list)
    return np.allclose(S, np.eye(S.shape[0]), atol=tol)


# ===========================================================================
# 1. Pauli error channels
# ===========================================================================

def kraus_bit_flip(p):
    """Bit-flip (X) error channel with probability p."""
    assert 0.0 <= p <= 1.0, "p must be in [0,1]"
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _X]


def kraus_phase_flip(p):
    """Phase-flip (Z) error channel with probability p."""
    assert 0.0 <= p <= 1.0
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _Z]


def kraus_bit_phase_flip(p):
    """Bit-phase-flip (Y) error channel with probability p."""
    assert 0.0 <= p <= 1.0
    return [np.sqrt(1 - p) * _I, np.sqrt(p) * _Y]


def kraus_depolarizing(p):
    """Single-qubit depolarizing channel with total error probability p.

    With prob (1-p) apply I, with prob p/3 each apply X, Y, Z.
    Channel maps any state rho -> (1-p)*rho + p/3*(X rho X + Y rho Y + Z rho Z).
    """
    assert 0.0 <= p <= 1.0
    return [
        np.sqrt(1 - p) * _I,
        np.sqrt(p / 3) * _X,
        np.sqrt(p / 3) * _Y,
        np.sqrt(p / 3) * _Z,
    ]


# ===========================================================================
# 2. Amplitude damping — T1 relaxation (zero temperature)
# ===========================================================================

def kraus_amplitude_damping(T1, dt):
    """T1 energy relaxation: spontaneous |1>→|0> decay.

    gamma = 1 - exp(-dt/T1) is the transition probability per step.
    K0 = [[1,0],[0,sqrt(1-gamma)]],  K1 = [[0,sqrt(gamma)],[0,0]]
    """
    assert T1 > 0 and dt > 0
    gamma = 1.0 - np.exp(-dt / T1)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


# ===========================================================================
# 3. Generalized amplitude damping — finite temperature T
# ===========================================================================

def kraus_generalized_amplitude_damping(T1, dt, n_th):
    """Amplitude damping at finite temperature (thermal bath with n_th excitations).

    n_th = 1/(exp(hbar*omega/kT) - 1) is the mean thermal photon number.
    At n_th=0 reduces to zero-temperature amplitude damping.

    Four Kraus operators preserving completeness exactly.
    """
    assert T1 > 0 and dt > 0 and n_th >= 0
    gamma = 1.0 - np.exp(-dt / T1)
    p = n_th / (2 * n_th + 1)          # excited-state thermal population factor

    K0 = np.sqrt(p)     * np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.sqrt(p)     * np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    K2 = np.sqrt(1 - p) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]], dtype=complex)
    K3 = np.sqrt(1 - p) * np.array([[0, 0], [np.sqrt(gamma), 0]], dtype=complex)
    return [K0, K1, K2, K3]


# ===========================================================================
# 4. Pure dephasing — T_phi
# ===========================================================================

def kraus_phase_damping(T_phi, dt):
    """Pure dephasing: off-diagonal coherences decay without energy loss.

    lambda = 1 - exp(-dt/T_phi).
    K0 = [[1,0],[0,sqrt(1-lambda)]],  K1 = [[0,0],[0,sqrt(lambda)]]
    """
    assert T_phi > 0 and dt > 0
    lam = 1.0 - np.exp(-dt / T_phi)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [K0, K1]


# ===========================================================================
# 5. Thermal relaxation — combined T1 + T2 (the physical channel)
# ===========================================================================

def kraus_thermal_relaxation(T1, T2, dt):
    """Combined T1 (amplitude damping) + T2 (total decoherence) channel.

    T2 constraint: T2 <= 2*T1 (physically enforced).
    T_phi = 1/(1/T2 - 1/(2*T1)) is the pure dephasing time.

    Three Kraus operators:
      K0 — amplitude decay + residual dephasing
      K1 — energy relaxation (|1>→|0>)
      K2 — pure dephasing contribution beyond T1
    """
    assert T1 > 0 and T2 > 0 and dt > 0
    if T2 > 2 * T1:
        raise ValueError(f"T2={T2} violates T2 <= 2*T1={2*T1}; unphysical channel")

    gamma = 1.0 - np.exp(-dt / T1)         # energy decay probability

    if T2 == 2 * T1:
        # Pure amplitude damping limit — no additional dephasing
        p_z = 0.0
    else:
        T_phi = 1.0 / (1.0 / T2 - 1.0 / (2.0 * T1))
        p_z = np.exp(-dt / T1) * (1.0 - np.exp(-dt / T_phi))

    # Clamp tiny negatives from floating-point
    p_z = max(p_z, 0.0)
    val = max(1.0 - gamma - p_z, 0.0)

    K0 = np.array([[1, 0], [0, np.sqrt(val)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    K2 = np.array([[0, 0], [0, np.sqrt(p_z)]], dtype=complex)
    return [K0, K1, K2]


# ===========================================================================
# 6. Coherent (unitary) errors — miscalibration
# ===========================================================================

def kraus_coherent_z_rotation(delta_phi):
    """Coherent Z-rotation error by angle delta_phi (over-/under-rotation on Z axis).
    A single Kraus operator — this is a unitary error, not stochastic.
    """
    return [np.array([[np.exp(-1j * delta_phi / 2), 0],
                      [0, np.exp(1j * delta_phi / 2)]], dtype=complex)]


def kraus_coherent_x_rotation(delta_theta):
    """Coherent X-rotation error by angle delta_theta (pulse amplitude error)."""
    c, s = np.cos(delta_theta / 2), np.sin(delta_theta / 2)
    return [np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)]


# ===========================================================================
# 7. SPAM errors (state preparation and measurement)
# ===========================================================================

def kraus_spam(p_prep, p_meas):
    """SPAM (state preparation and measurement) error channel.

    p_prep: probability of preparing wrong state (bit-flip at start)
    p_meas: probability of flipping measurement outcome (classical bit flip)

    Returns Kraus set for the combined SPAM channel applied to state before
    measurement. The readout bit-flip is classical, so total channel is a
    convex combination of I and X on the state.
    """
    p_total = p_prep + p_meas - 2 * p_prep * p_meas  # combined error prob
    p_total = np.clip(p_total, 0.0, 1.0)
    return kraus_bit_flip(p_total)


# ===========================================================================
# 8. Two-qubit depolarizing channel
# ===========================================================================

def kraus_two_qubit_depolarizing(p):
    """Two-qubit depolarizing channel with total error probability p.

    Acts on the 4-dimensional Hilbert space of two qubits. With prob (1-p)
    apply I⊗I, with prob p/15 apply each of the 15 non-identity two-qubit
    Pauli operators.
    """
    assert 0.0 <= p <= 1.0
    paulis = [_I, _X, _Y, _Z]
    kraus = []
    for P1 in paulis:
        for P2 in paulis:
            op = np.kron(P1, P2)
            if np.allclose(op, np.eye(4)):
                kraus.append(np.sqrt(1 - p) * op)
            else:
                kraus.append(np.sqrt(p / 15) * op)
    return kraus


# ===========================================================================
# 9. Applying channels to state vectors
# ===========================================================================

def apply_channel_stochastic(state, kraus_list, qubit, n_qubits):
    """Apply a single-qubit Kraus channel stochastically (quantum trajectory).

    Selects Kraus operator k with probability p_k = <psi|K_k†K_k|psi>,
    applies it, and renormalizes — giving a pure-state trajectory.

    Args:
        state:      complex state vector, length 2**n_qubits
        kraus_list: list of 2x2 Kraus operators
        qubit:      1-indexed qubit to act on
        n_qubits:   total number of qubits

    Returns:
        (new_state, k_index) — post-error state and which Kraus was applied.
    """
    probs = []
    outcomes = []
    for K in kraus_list:
        psi_k = single_qubit_op(state, K, qubit, n_qubits)
        p_k = np.vdot(psi_k, psi_k).real
        probs.append(max(p_k, 0.0))
        outcomes.append(psi_k)

    total = sum(probs)
    if total < 1e-15:
        return state.copy(), 0
    probs_norm = [p / total for p in probs]

    r = np.random.uniform(0, 1)
    cumulative = 0.0
    for k, (p, psi_k) in enumerate(zip(probs_norm, outcomes)):
        cumulative += p
        if r <= cumulative:
            return (psi_k / np.sqrt(probs[k]) if probs[k] > 1e-15 else psi_k), k

    # Numerical fallback: last operator
    k = len(probs) - 1
    psi_k = outcomes[k]
    return (psi_k / np.sqrt(probs[k]) if probs[k] > 1e-15 else psi_k), k


def apply_two_qubit_channel_stochastic(state, kraus_list, q1, q2, n_qubits):
    """Apply a two-qubit Kraus channel stochastically.

    kraus_list: list of 4x4 Kraus operators.
    q1, q2:     1-indexed qubit positions (must be distinct).
    """
    probs = []
    outcomes = []
    for K in kraus_list:
        psi_k = two_qubit_op(state, K, q1, q2, n_qubits)
        p_k = np.vdot(psi_k, psi_k).real
        probs.append(max(p_k, 0.0))
        outcomes.append(psi_k)

    total = sum(probs)
    if total < 1e-15:
        return state.copy(), 0
    probs_norm = [p / total for p in probs]

    r = np.random.uniform(0, 1)
    cumulative = 0.0
    for k, (p, psi_k) in enumerate(zip(probs_norm, outcomes)):
        cumulative += p
        if r <= cumulative:
            return (psi_k / np.sqrt(probs[k]) if probs[k] > 1e-15 else psi_k), k

    k = len(probs) - 1
    psi_k = outcomes[k]
    return (psi_k / np.sqrt(probs[k]) if probs[k] > 1e-15 else psi_k), k


# ===========================================================================
# 10. Density matrix formalism
# ===========================================================================

def state_to_density_matrix(state):
    """Convert a pure state vector to its density matrix rho = |psi><psi|."""
    return np.outer(state, state.conj())


def apply_channel_density_matrix(rho, kraus_list):
    """Apply a Kraus channel to a density matrix: rho' = sum_k K_k rho K_k†.

    Works for any dimension matching the Kraus operators.
    """
    dim = rho.shape[0]
    rho_new = np.zeros((dim, dim), dtype=complex)
    for K in kraus_list:
        rho_new += K @ rho @ K.conj().T
    return rho_new


def apply_single_qubit_channel_to_dm(rho, kraus_list, qubit, n_qubits):
    """Apply a single-qubit channel to the full n-qubit density matrix
    by embedding each 2x2 Kraus operator as a 2^n x 2^n operator via
    tensor product structure.

    This is exact but O(4^n) — use only for small n.
    """
    dim = 2 ** n_qubits
    rho_new = np.zeros((dim, dim), dtype=complex)
    for K in kraus_list:
        K_full = _embed_single_qubit_op(K, qubit, n_qubits)
        rho_new += K_full @ rho @ K_full.conj().T
    return rho_new


def _embed_single_qubit_op(op_2x2, qubit, n_qubits):
    """Build the 2^n × 2^n matrix for op_2x2 acting on qubit (1-indexed)."""
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit - 1] = op_2x2
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def density_matrix_to_bloch(rho):
    """Extract the Bloch vector (rx, ry, rz) from a single-qubit density matrix."""
    rx = 2.0 * rho[0, 1].real
    ry = 2.0 * rho[0, 1].imag
    rz = (rho[0, 0] - rho[1, 1]).real
    return np.array([rx, ry, rz], dtype=float)


def dm_fidelity(rho, sigma):
    """Fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    For pure sigma = |psi><psi| simplifies to <psi|rho|psi>.
    Uses eigendecomposition of the symmetric form — exact for small matrices.
    """
    # General form via sqrt of rho
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    eigvals_rho = np.maximum(eigvals_rho, 0.0)
    sqrt_rho = eigvecs_rho @ np.diag(np.sqrt(eigvals_rho)) @ eigvecs_rho.conj().T

    M = sqrt_rho @ sigma @ sqrt_rho
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.maximum(eigvals_M, 0.0)
    return float(np.sum(np.sqrt(eigvals_M)) ** 2)


# ===========================================================================
# 11. Multi-step noisy trajectory helpers
# ===========================================================================

def apply_noise_layer(state, n_qubits, kraus_list_per_qubit):
    """Apply independent single-qubit error channels to all qubits.

    kraus_list_per_qubit: list of length n_qubits, each element is a Kraus list
    for that qubit (1-indexed). Use same list for all qubits if homogeneous.

    Returns (state, outcomes_list).
    """
    outcomes = []
    for q in range(1, n_qubits + 1):
        kraus = kraus_list_per_qubit[q - 1]
        state, k = apply_channel_stochastic(state, kraus, q, n_qubits)
        outcomes.append(k)
    return state, outcomes


def run_noisy_circuit(gate_sequence, n_qubits, T1, T2, dt):
    """Run a sequence of (gate_fn, args) operations with thermal relaxation noise
    applied to all qubits after each gate.

    gate_fn(state, *args) must return the new state.
    Returns final state and list of per-step error outcomes.
    """
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    kraus = kraus_thermal_relaxation(T1, T2, dt)
    all_outcomes = []

    for gate_fn, args in gate_sequence:
        state = gate_fn(state, *args)
        state, outcomes = apply_noise_layer(state, n_qubits,
                                            [kraus] * n_qubits)
        all_outcomes.append(outcomes)

    return state, all_outcomes
