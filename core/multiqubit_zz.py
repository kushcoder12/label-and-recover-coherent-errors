"""Static ZZ crosstalk: recovering a neighbor-coupling vector from RC labels.

Extends the single-qubit sign-flip theorem (core/twirl.py, core/estimator.py) to a
DIFFERENT mechanism than a neighbor's real microwave drive: static ZZ coupling, an
always-on Z_i Z_j term present whether or not either qubit is being driven. Recovery
requires deliberately preparing each neighbor in a known, independently-randomized
computational-basis state every shot and recording that choice as an additional label
c_j alongside the qubit's own RC sign s_1 -- this is NOT a passive read of an
unmodified production circuit, it is a second owned label the caller must inject.

Convention: 1-indexed qubits, big-endian, matching operator_on_state.py. Labels are
always returned to the caller (twirl.py's invariant); this module never averages over
s_1 or any c_j internally.
"""

import numpy as np

from .pauli import pauli
from .operator_on_state import single_qubit_op
from .single_qubit_gates import X_gate
from ._compat import quiet

_X = X_gate()


def _embed_1q(op2, q, n):
    """Full 2^n operator for a 2x2 op on qubit q (1-indexed, big-endian)."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[q - 1] = op2
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


@quiet
def build_zz_hamiltonian(deltas, J_matrix, n):
    """Dense 2^n x 2^n Hamiltonian, Sec 1:

        H = sum_i (deltas[i]/2) Z_i + sum_{i<j} (J_matrix[i][j]/4) Z_i Z_j.

    deltas   : length-n sequence, qubit (i+1)'s own detuning-type coherent error.
    J_matrix : n x n symmetric matrix; J_matrix[i][j] (i<j) is the static ZZ coupling
               between qubits i+1 and j+1. Diagonal and the i>j half are ignored.
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    Zs = [_embed_1q(pauli("Z"), q, n) for q in range(1, n + 1)]
    for i in range(n):
        if deltas[i] != 0.0:
            H += (deltas[i] / 2.0) * Zs[i]
    for i in range(n):
        for j in range(i + 1, n):
            Jij = J_matrix[i][j]
            if Jij != 0.0:
                H += (Jij / 4.0) * (Zs[i] @ Zs[j])
    return H


def sample_neighbor_labels(neighbors, rng=None):
    """Draw one independent, uniform +/-1 label c_j per neighbor qubit (Sec 2.1).

    neighbors : iterable of neighbor qubit indices (1-indexed).
    Returns {j: c_j}. Labels are handed back to the caller and never averaged here,
    matching twirl.sample_sign_pattern's convention.
    """
    if rng is None:
        rng = np.random.default_rng()
    neighbors = list(neighbors)
    draws = rng.random(len(neighbors))
    return {j: (1 if d < 0.5 else -1) for j, d in zip(neighbors, draws)}


def prepare_neighbor_state(state, c_j, qubit_j, n):
    """Prepare neighbor qubit_j in |1> if c_j=-1, leave it at |0> if c_j=+1.

    The caller supplies c_j (from sample_neighbor_labels); this function only applies
    the corresponding real state-preparation gate, exactly as twirl.apply_signed_z_error
    only applies the sign it is handed.
    """
    if c_j == -1:
        return single_qubit_op(state, _X, qubit_j, n)
    return state


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
def _point_estimate(s_arr, c_arrs, b_arr, neighbor_qubits):
    """One pass of the inversion (Sec 2.4, 2.4a) from raw label/outcome arrays.

    Exact two-label inversion for a single neighbor; the leading-order multi-neighbor
    inversion of Sec 2.4b otherwise (their formulas agree to leading order in kappa).
    """
    X = float(np.mean(s_arr * b_arr))
    Y = {j: float(np.mean(s_arr * c_arrs[j] * b_arr)) for j in neighbor_qubits}

    if len(neighbor_qubits) == 1:
        j = neighbor_qubits[0]
        Xp = np.clip(X + Y[j], -1.0, 1.0)
        Xm = np.clip(X - Y[j], -1.0, 1.0)
        theta0_hat = 0.5 * (np.arcsin(Xp) + np.arcsin(Xm))
        kappa_hat = {j: 0.5 * (np.arcsin(Xp) - np.arcsin(Xm))}
        return float(theta0_hat), {j: float(kappa_hat[j])}

    theta0_hat = float(np.arcsin(np.clip(X, -1.0, 1.0)))
    cos_t0 = np.cos(theta0_hat)
    kappa_hat = {}
    for j in neighbor_qubits:
        if abs(cos_t0) < 1e-12:
            kappa_hat[j] = float("nan")
            continue
        kappa_hat[j] = float(np.arcsin(np.clip(Y[j], -1.0, 1.0)) / cos_t0)
    return theta0_hat, kappa_hat


def joint_correlation_estimator(records, target_qubit, neighbor_qubits, B=1000, seed=None):
    """Recover theta_1^(0) and every kappa_1j from one pool of owned-label shots.

    records         : iterable of (s_1, c_dict, b_1), s_1/b_1 in {+1,-1}, c_dict a
                       {neighbor: +/-1} mapping covering every entry of neighbor_qubits.
    target_qubit    : qubit 1's index (carried through into the result for bookkeeping;
                       the records are already specific to this target).
    neighbor_qubits : the neighbor indices to recover kappa_1j for.

    Returns {theta0_hat, kappa_hat: {j: value}, bootstrap_ci: {theta0: (lo,hi),
    kappa: {j: (lo,hi)}}}, resampling over shots (matching hardware/analyze.py's
    bootstrap-over-instances pattern).
    """
    neighbor_qubits = list(neighbor_qubits)
    records = list(records)
    s_arr = np.array([r[0] for r in records], dtype=float)
    b_arr = np.array([r[2] for r in records], dtype=float)
    c_arrs = {j: np.array([r[1][j] for r in records], dtype=float) for j in neighbor_qubits}

    theta0_hat, kappa_hat = _point_estimate(s_arr, c_arrs, b_arr, neighbor_qubits)

    rng = np.random.default_rng(seed)
    m = len(records)
    theta0_bs = np.empty(B)
    kappa_bs = {j: np.empty(B) for j in neighbor_qubits}
    for b in range(B):
        idx = rng.integers(0, m, m)
        th, kh = _point_estimate(s_arr[idx], {j: c_arrs[j][idx] for j in neighbor_qubits},
                                 b_arr[idx], neighbor_qubits)
        theta0_bs[b] = th
        for j in neighbor_qubits:
            kappa_bs[j][b] = kh[j]

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    return {
        "target_qubit": target_qubit,
        "theta0_hat": theta0_hat,
        "kappa_hat": kappa_hat,
        "bootstrap_ci": {
            "theta0": ci(theta0_bs),
            "kappa": {j: ci(kappa_bs[j]) for j in neighbor_qubits},
        },
    }


# ---------------------------------------------------------------------------
# Drive-crosstalk recovery (DRIVE_CROSSTALK_SPEC Sec 2.3-2.4): a third label p_j
# stratifies the same pool into two Part-2 problems, each solved by the ABOVE
# _point_estimate -- no new inversion math, only the stratify/combine bookkeeping.
# ---------------------------------------------------------------------------
def stratified_three_label_estimator(records, target_qubit=1, neighbor_qubit=2,
                                     B=1000, seed=None):
    """Recover theta_1_0, kappa_1j, phi_1j from one pool of (s_1, c_j, p_j, b_1) shots.

    records : iterable of (s_1, c_j, p_j, b_1), all four in {+1,-1}.

    Within a fixed p_j the model reduces exactly to Part 2's two-label form (Sec 2.3):
    stratum p_j=+1 has effective (Theta-Phi, K), stratum p_j=-1 has (Theta+Phi, K),
    where Theta=theta_1_0+1.5*phi_1j, K=kappa_1j-0.5*phi_1j, Phi=phi_1j. Splitting by
    p_j and calling _point_estimate (the same arcsin-sum/difference inversion
    joint_correlation_estimator uses) on each stratum, then combining per Sec 2.3-2.4,
    recovers all three physical parameters exactly (Oracle DC2) -- the two-label
    inversion itself is never reimplemented here.

    Returns {theta_1_0_hat, kappa_1j_hat, phi_1j_hat, K_disagreement, bootstrap_ci}.
    K_disagreement = |K_hat from the p_j=+1 stratum - K_hat from the p_j=-1 stratum|,
    reported unconditionally (Sec 2.3): a large value signals the additive-angle model
    is not holding at the tested parameters, or that padding is not behaving as
    assumed, and must not be silently averaged away.
    """
    records = list(records)
    plus = [(r[0], r[1], r[3]) for r in records if r[2] == 1]
    minus = [(r[0], r[1], r[3]) for r in records if r[2] == -1]
    if not plus or not minus:
        raise ValueError("stratified_three_label_estimator needs shots at both p_j=+1 "
                         "and p_j=-1 -- got %d and %d" % (len(plus), len(minus)))

    def _arrays(stratum):
        s = np.array([r[0] for r in stratum], dtype=float)
        c = np.array([r[1] for r in stratum], dtype=float)
        b = np.array([r[2] for r in stratum], dtype=float)
        return s, c, b

    s_p, c_p, b_p = _arrays(plus)
    s_m, c_m, b_m = _arrays(minus)

    def _combine(s_p, c_p, b_p, s_m, c_m, b_m):
        # stratum p_j=+1 (s_p/c_p/b_p): model reduces to Part 2's form with
        # theta_1_0 -> (Theta-Phi), kappa_1j -> K.
        theta_minus_phi, kappa_from_pplus = _point_estimate(s_p, {neighbor_qubit: c_p}, b_p,
                                                             [neighbor_qubit])
        # stratum p_j=-1 (s_m/c_m/b_m): theta_1_0 -> (Theta+Phi), kappa_1j -> K.
        theta_plus_phi, kappa_from_pminus = _point_estimate(s_m, {neighbor_qubit: c_m}, b_m,
                                                             [neighbor_qubit])
        K_pplus = kappa_from_pplus[neighbor_qubit]
        K_pminus = kappa_from_pminus[neighbor_qubit]
        Theta = 0.5 * (theta_plus_phi + theta_minus_phi)
        Phi = 0.5 * (theta_plus_phi - theta_minus_phi)
        K = 0.5 * (K_pplus + K_pminus)
        return Theta, Phi, K, abs(K_pplus - K_pminus)

    Theta_hat, Phi_hat, K_hat, K_disagreement = _combine(s_p, c_p, b_p, s_m, c_m, b_m)
    theta_1_0_hat = Theta_hat - 1.5 * Phi_hat
    kappa_1j_hat = K_hat + 0.5 * Phi_hat
    phi_1j_hat = Phi_hat

    rng = np.random.default_rng(seed)
    m_p, m_m = len(plus), len(minus)
    theta0_bs = np.empty(B)
    kappa_bs = np.empty(B)
    phi_bs = np.empty(B)
    for b in range(B):
        idx_p = rng.integers(0, m_p, m_p)
        idx_m = rng.integers(0, m_m, m_m)
        Th, Ph, K, _ = _combine(s_p[idx_p], c_p[idx_p], b_p[idx_p],
                                s_m[idx_m], c_m[idx_m], b_m[idx_m])
        theta0_bs[b] = Th - 1.5 * Ph
        kappa_bs[b] = K + 0.5 * Ph
        phi_bs[b] = Ph

    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    return {
        "target_qubit": target_qubit,
        "neighbor_qubit": neighbor_qubit,
        "theta_1_0_hat": float(theta_1_0_hat),
        "kappa_1j_hat": float(kappa_1j_hat),
        "phi_1j_hat": float(phi_1j_hat),
        "K_disagreement": float(K_disagreement),
        "bootstrap_ci": {
            "theta_1_0": ci(theta0_bs),
            "kappa_1j": ci(kappa_bs),
            "phi_1j": ci(phi_bs),
        },
    }
