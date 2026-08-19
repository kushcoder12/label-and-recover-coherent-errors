"""Trotter circuits carrying twirled coherent errors.

trotter_single_qubit  — single qubit, L identical cycles [Rx(alpha), Rz(s_ell*theta)].
trotter_tfim          — 2-qubit TFIM Trotter step with a twirled Rzz coherent error.

Each builder returns the final state, the ideal pre-error state |chi>, and the
propagated error generators G-tilde_ell (dense operators) used for the Fisher
analysis. An optional per-cycle noise channel supports the noisy-saturation study.
"""

import numpy as np

from .single_qubit_gates import Rx, Rz, X_gate, Y_gate, Z_gate, Hadamard
from .two_qubit_gates import Rzz
from .operator_on_state import single_qubit_op, two_qubit_op
from .pauli import pauli, pauli_string, propagate_generator, expectation_single_qubit
from .noise import apply_incoherent_layer
from .multiqubit_zz import build_zz_hamiltonian, prepare_neighbor_state
from .twirl import apply_signed_z_error
from .drive_crosstalk import n_real_pulses, apply_padding, inject_drive_leakage
from ._compat import quiet

_Z = pauli("Z")


def _embed_1q(op2, qubit, n):
    """Full 2^n operator for a 2x2 op on `qubit` (1-indexed, big-endian)."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[qubit - 1] = op2
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _zero_state(n):
    s = np.zeros(2**n, dtype=complex)
    s[0] = 1.0
    return s


def trotter_single_qubit(L, alpha, theta, signs, noise=None, noise_channel=None, rng=None):
    """Single-qubit depth model.

    Cycle ell (1..L): apply ideal U0 = Rx(alpha), then coherent error Rz(s_ell*theta),
    then (if `noise` is a NoiseProfile) one incoherent noise layer. Input |0>.

    Parameters
    ----------
    L : int              number of Trotter cycles
    alpha : float        ideal rotation angle per cycle
    theta : float        coherent Z-error angle (the recoverable signal)
    signs : sequence     length-L labels s_ell in {+1,-1}
    noise : NoiseProfile or None
        When None (default) the builder is the exact noiseless statevector model and
        reproduces Oracles O1-O5 (N5). When a NoiseProfile is given, the qubit is
        evolved as a density matrix through incoherent layers (amplitude damping,
        dephasing, depolarizing) after each cycle; readout is applied separately at
        the outcome stage (noise.apply_readout). Returns rho instead of a statevector.
    noise_channel : callable or None
        Legacy per-cycle callable noise_channel(state, rng) (statevector path only).
    rng : np.random.Generator or None.

    Returns
    -------
    state_or_rho : ndarray
        length-2 statevector when noise is None; 2x2 density matrix when noise given.
    chi   : ndarray      ideal state |chi> = Rx(L*alpha)|0>  (statevector, noiseless).
    gens  : list[ndarray]  propagated generators G-tilde_ell = Rx((L-ell)a) Z Rx((L-ell)a)^†,
                           ell = 1..L; G^2 = I and <G-tilde_ell>_chi = cos(ell*alpha) (S2, S4).
    """
    assert len(signs) == L, "signs must have length L"
    n = 1
    chi = single_qubit_op(_zero_state(n), Rx(L * alpha), 1, n)
    gens = [propagate_generator(_Z, Rx((L - ell) * alpha)) for ell in range(1, L + 1)]

    if noise is None:
        state = _zero_state(n)
        for ell in range(1, L + 1):
            state = single_qubit_op(state, Rx(alpha), 1, n)
            state = single_qubit_op(state, Rz(signs[ell - 1] * theta), 1, n)
            if noise_channel is not None:
                state = noise_channel(state, rng)
        return state, chi, gens

    # Density-matrix path with a NoiseProfile.
    RxU = _embed_1q(Rx(alpha), 1, n)
    rho = np.zeros((2, 2), dtype=complex)
    rho[0, 0] = 1.0
    for ell in range(1, L + 1):
        rho = RxU @ rho @ RxU.conj().T
        Ez = _embed_1q(Rz(signs[ell - 1] * theta), 1, n)
        rho = Ez @ rho @ Ez.conj().T
        rho = apply_incoherent_layer(rho, noise, n, mode="dm")
    return rho, chi, gens


def trotter_tfim(L, tau, g, delta, signs, n=2, noise=None, rng=None):
    """Two-qubit TFIM Trotter model.

    H = Z1 Z2 + g (X1 + X2). Each Trotter cycle:
        exp(-i tau Z1 Z2) . exp(-i tau g X1) . exp(-i tau g X2)
    The 'hard' gate exp(-i tau Z1 Z2) = Rzz(2 tau) carries the coherent
    over-rotation error: it is applied as Rzz(2 tau + s_ell * delta), i.e. a
    signed coherent error exp(-i s_ell delta Z1 Z2 / 2) with generator Z1 Z2.
    Input |00>.

    noise : NoiseProfile or None
        When None (default) the exact noiseless statevector model (reproduces O5 /
        conservation). When a NoiseProfile is given, evolves the density matrix and
        applies an incoherent layer (single-qubit channels on both qubits + two-qubit
        depolarizing on the entangled pair (1,2)) after each cycle. Returns rho.

    Returns
    -------
    state_or_rho : ndarray   final 2-qubit statevector (noise None) or 4x4 density matrix.
    chi   : ndarray      ideal final state |chi> = U0^L |00> (the output-frame reference).
    gens  : list[ndarray]  propagated Pauli-string generators G-tilde_ell (4x4),
                           ell = 1..L, each squaring to I (S5 generalized). Defined in
                           the output frame: G-tilde_ell = U0^(L-ell) ZZ (U0^(L-ell))^dagger.
    """
    assert len(signs) == L
    ZZ = pauli_string("ZZ")

    def ideal_cycle_unitary():
        # Build the dense 4x4 ideal cycle unitary U0 = Rzz(2tau) . Rx1(2 tau g) . Rx2(2 tau g)
        # Note exp(-i tau g X) = Rx(2 tau g) per R_P(theta)=exp(-i theta P/2).
        Rx1 = np.kron(Rx(2 * tau * g), np.eye(2))
        Rx2 = np.kron(np.eye(2), Rx(2 * tau * g))
        return Rzz(2 * tau) @ Rx1 @ Rx2

    U0 = ideal_cycle_unitary()

    # Ideal (error-free) final state = output-frame reference |chi> = U0^L |00>.
    chi = _zero_state(n)
    for _ in range(L):
        chi = _apply_dense(chi, U0, n)

    if noise is None:
        # Forward evolution with signed coherent error on the ZZ generator each cycle.
        state = _zero_state(n)
        for ell in range(1, L + 1):
            state = _apply_dense(state, U0, n)
            err = _expm_anti(-1j * signs[ell - 1] * delta / 2 * ZZ)
            state = _apply_dense(state, err, n)
        final = state
    else:
        # Density-matrix path with a NoiseProfile.
        rho = np.zeros((2**n, 2**n), dtype=complex)
        rho[0, 0] = 1.0
        for ell in range(1, L + 1):
            rho = U0 @ rho @ U0.conj().T
            err = _expm_anti(-1j * signs[ell - 1] * delta / 2 * ZZ)
            rho = err @ rho @ err.conj().T
            rho = apply_incoherent_layer(rho, noise, n, mode="dm", pairs_2q=[(1, 2)])
        final = rho

    # Propagate the error generator ZZ from each cycle to the output frame.
    # After the error at cycle ell, the remaining ideal unitaries are U0^(L-ell).
    gens = []
    W = np.eye(2**n, dtype=complex)
    for ell in range(L, 0, -1):
        gens.append(propagate_generator(ZZ, W))
        W = W @ U0                  # accumulate one more ideal cycle toward the output
    gens.reverse()                  # order ell = 1..L
    return final, chi, gens


@quiet
def multi_qubit_zz_probe(theta0, kappa_dict, s_1, c_dict, noise=None):
    """Star-topology ZZ-crosstalk probe (MULTIQUBIT_CROSSTALK_SPEC Sec 2, 4).

    Qubit 1 (the target) starts at |+>; each neighbor j in kappa_dict is prepared in
    |0> (c_dict[j]=+1) or |1> (c_dict[j]=-1); qubits 2..Q+1 are the neighbors, coupled
    to qubit 1 only (no neighbor-neighbor coupling). The state then evolves for unit
    time under the FULL (1+Q)-qubit static Hamiltonian

        H = (theta0/2) Z_1 + sum_j (kappa_1j) Z_1 Z_j,

    twirled by s_1 (an X_1-dressing, which flips the sign of every Z_1-bearing term,
    since X_1 anticommutes with Z_1 and with Z_1 Z_j but not with Z_j alone). This is
    the FULL evolution, not the effective-angle formula theta_1 = theta0 + sum kappa*c
    -- that formula is a prediction to be checked against this builder's output
    (see test_multiqubit_zz.test_probe_matches_theory), never hard-coded into it.

    kappa_dict, c_dict : {neighbor qubit index: value}, same key set, neighbors must
                          be qubits 2..Q+1 (contiguous, star topology).
    noise               : NoiseProfile or None.

    Returns <Y_1>, the exact expectation value of Y on qubit 1 (float).
    """
    neighbors = sorted(kappa_dict)
    n = 1 + len(neighbors)
    if neighbors != list(range(2, n + 1)):
        raise ValueError(f"neighbors must be qubits 2..{n} (star topology), got {neighbors}")

    deltas = [theta0] + [0.0] * len(neighbors)
    J = [[0.0] * n for _ in range(n)]
    for j in neighbors:
        J[0][j - 1] = J[j - 1][0] = 2.0 * kappa_dict[j]
    H = build_zz_hamiltonian(deltas, J, n)
    U = _expm_anti(-1j * s_1 * H)

    state = _zero_state(n)
    state = single_qubit_op(state, Hadamard(), 1, n)
    for j in neighbors:
        if c_dict[j] == -1:
            state = single_qubit_op(state, X_gate(), j, n)

    if noise is None:
        state = U @ state
        return expectation_single_qubit(state, Y_gate(), 1, n)

    rho = np.outer(state, state.conj())
    rho = U @ rho @ U.conj().T
    rho = apply_incoherent_layer(rho, noise, n, mode="dm")
    Y1 = _embed_1q(Y_gate(), 1, n)
    return float(np.trace(rho @ Y1).real)


def two_qubit_zz_probe(theta0, kappa_12, s_1, c_2, noise=None):
    """Two-qubit special case of multi_qubit_zz_probe: one neighbor (qubit 2)."""
    return multi_qubit_zz_probe(theta0, {2: kappa_12}, s_1, {2: c_2}, noise=noise)


@quiet
def three_label_probe(theta_1_0, kappa_1j, phi_1j, s_1, c_j, p_j, noise=None):
    """Drive-crosstalk probe: adds the padding label p_j to two_qubit_zz_probe
    (DRIVE_CROSSTALK_SPEC Sec 1-2). Qubit 1 (target) starts at |+>, qubit 2 (neighbor)
    is prepared per c_j, exactly as two_qubit_zz_probe/multi_qubit_zz_probe do -- that
    |+> convention (not the literal |0> shorthand of Sec 2.1) is reused unchanged here,
    since it is what makes the Y-probe sensitive to a Z-generator error in the first
    place.

    On top of that unchanged Part-2 construction, this adds two NEW steps: firing extra
    real pulses on the neighbor per p_j (apply_padding, a no-op on the logical state --
    X_pi twice is identity), and injecting the resulting phenomenological leakage
    Rz(s_1 * phi_1j * n_j) on qubit 1 (inject_drive_leakage), where n_j counts every
    real pulse the neighbor received this shot (n_real_pulses). qubit 1's own error and
    the static-ZZ contribution are reused exactly as in Part 2 -- apply_signed_z_error
    (twirl.py) and build_zz_hamiltonian (multiqubit_zz.py) unchanged.

    All four terms (own error, ZZ coupling, leakage) are diagonal in the Z1 basis, so
    they commute and can be applied in any order; together they reproduce exactly

        <Y_1> = s_1 * sin(theta_1_0 + kappa_1j * c_j + phi_1j * n_j(c_j, p_j))

    checked against this builder in tests/test_drive_crosstalk.py, never hard-coded
    into it (same discipline as multi_qubit_zz_probe).

    Returns <Y_1>, the exact expectation value of Y on qubit 1 (float).
    """
    n = 2
    n_j = n_real_pulses(c_j, p_j)

    state = _zero_state(n)
    state = single_qubit_op(state, Hadamard(), 1, n)
    state = prepare_neighbor_state(state, c_j, 2, n)
    state = apply_padding(state, p_j, 2, n)

    state = apply_signed_z_error(state, theta_1_0, s_1, 1, n)

    J = [[0.0, 2.0 * kappa_1j], [2.0 * kappa_1j, 0.0]]
    H = build_zz_hamiltonian([0.0, 0.0], J, n)
    state = _expm_anti(-1j * s_1 * H) @ state

    state = inject_drive_leakage(state, s_1 * phi_1j, n_j, 1, n)

    if noise is None:
        return expectation_single_qubit(state, Y_gate(), 1, n)

    rho = np.outer(state, state.conj())
    rho = apply_incoherent_layer(rho, noise, n, mode="dm")
    Y1 = _embed_1q(Y_gate(), 1, n)
    return float(np.trace(rho @ Y1).real)


# ---------------------------------------------------------------------------
# small dense helpers (n small only)
# ---------------------------------------------------------------------------
def _apply_dense(state, U, n):
    return U @ state


def _expm_anti(A):
    """exp(A) for anti-Hermitian A via eigendecomposition of the Hermitian iA."""
    w, V = np.linalg.eigh(1j * A)
    return V @ np.diag(np.exp(-1j * w)) @ V.conj().T
