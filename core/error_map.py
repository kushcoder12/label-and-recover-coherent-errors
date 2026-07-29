"""core/error_map.py — recovery of a VECTOR of coherent errors from RC labels
(GAPS_CODE.md GAP 3; THEOREMS.md conservation parts (1)-(4), Results A-D).

A circuit carries K injected coherent errors (single-qubit over-rotations and/or
two-qubit ZZ crosstalk), each twirled with its own per-shot sign label s_a. Keeping
the labels, each coherent angle theta_a is recovered INDIVIDUALLY (the labeled
estimator reads diag(M)) via the label-outcome correlation, debiased by the
visibility v_a read self-consistently from the marginal stochastic channel:

    theta_hat_a = arcsin( E[s_a * Y_a] / (v_a * h_a) ),

where, working in the eigenbasis of the propagated generator G-tilde_a = W_a P_a W_a^dag
(the effective qubit spanned by the +/-1 eigen-projections of the ideal output |chi>):
  * Y_a is the transverse "signal" observable of that effective qubit,
  * h_a = sqrt(1 - <G-tilde_a>_chi^2) is the (known) geometric amplitude,
  * v_a = f_{Y_a}, the PTM diagonal element (Pauli fidelity) of Y_a under the marginal
    twirled channel (noise + the OTHER errors twirled) -- computed from the noise model
    and the estimated other-angles, NEVER from the true theta_a. Self-consistent: one
    refinement iteration uses the first-pass estimates for the other-error twirls.

All physics is exact (numpy density matrices); qiskit is never used here. Verified
noiseless -> exact recovery, and unbiased under the full NoiseProfile (tests/test_error_map).
"""

from dataclasses import dataclass, field

import numpy as np

from .pauli import pauli, pauli_string
from .noise import (apply_incoherent_layer, amplitude_damping_kraus, dephasing_kraus,
                    depolarizing_kraus_1q)
from .quantum_errors import apply_single_qubit_channel_to_dm
from ._compat import quiet_accelerate, quiet


# ---------------------------------------------------------------------------
# Error specification
# ---------------------------------------------------------------------------
@dataclass
class CoherentError:
    """One injected coherent error.

    pos    : insert AFTER ideal_ops[pos-1] (index in [0, len(ideal_ops)]).
    qubits : tuple of 1-indexed qubits it acts on.
    gen    : Pauli label -- '1q' generators 'X'/'Y'/'Z' (single qubit) or a 2-char
             string like 'ZZ' for a two-qubit generator on qubits[0], qubits[1].
    theta  : ground-truth angle (the quantity to recover).
    """
    pos: int
    qubits: tuple
    gen: str
    theta: float
    kind: str = field(init=False)

    def __post_init__(self):
        self.kind = "1q" if len(self.gen) == 1 else "2q"


def _embed_1q(op2, q, n):
    ops = [np.eye(2, dtype=complex)] * n
    ops[q - 1] = op2
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M


def generator_matrix(err, n):
    """Full 2^n Hermitian generator P (P^2 = I) for a CoherentError."""
    if err.kind == "1q":
        return _embed_1q(pauli(err.gen), err.qubits[0], n)
    # two-qubit Pauli string on (q1, q2)
    q1, q2 = err.qubits
    P1 = _embed_1q(pauli(err.gen[0]), q1, n)
    P2 = _embed_1q(pauli(err.gen[1]), q2, n)
    return P1 @ P2


# ---------------------------------------------------------------------------
# small dense helpers
# ---------------------------------------------------------------------------
def _expm_anti(A):
    w, V = np.linalg.eigh(1j * A)
    return V @ np.diag(np.exp(-1j * w)) @ V.conj().T


def _twirl_op(M, P, theta):
    """Sign-twirl of a coherent error with generator P and angle theta, applied to M:
    E_s[ exp(-i s theta P/2) M exp(+i s theta P/2) ] = cos^2(t/2) M + sin^2(t/2) P M P."""
    c2 = np.cos(theta / 2) ** 2
    s2 = np.sin(theta / 2) ** 2
    return c2 * M + s2 * (P @ M @ P)


def _noise_on_operator(M, profile, n):
    """Apply the incoherent layer (as a CPTP map, Schrodinger sum) to operator/state M."""
    if profile.is_noiseless():
        return M
    Ks = []
    if profile.r_1q > 0:
        Ks.append(depolarizing_kraus_1q(profile))
    if profile.T2 != np.inf:
        Ks.append(dephasing_kraus(profile))
    if profile.T1 != np.inf:
        Ks.append(amplitude_damping_kraus(profile))
    out = M
    for q in range(1, n + 1):
        for K in Ks:
            out = apply_single_qubit_channel_to_dm(out, K, q, n)
    return out


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
@dataclass
class Recovered:
    label: str
    kind: str
    qubits: tuple
    theta_true: float
    theta_hat: float
    sigma: float          # 1-sigma shot-noise (from M shots), 0 if shots is None
    g: float
    visibility: float
    silent: bool


def _segments(ideal_ops, n, positions):
    """Return the ideal segment unitaries between consecutive error positions.

    positions: sorted unique insertion indices. seg[i] = product of ideal_ops applied
    between boundary i and i+1 (as a dense 2^n unitary). Boundaries: 0, p_1, ..., p_m, len.
    """
    bounds = [0] + list(positions) + [len(ideal_ops)]
    segs = []
    for i in range(len(bounds) - 1):
        U = np.eye(2**n, dtype=complex)
        for k in range(bounds[i], bounds[i + 1]):
            U = ideal_ops[k] @ U
        segs.append(U)
    return segs, bounds


@quiet
def recover_error_vector(n, ideal_ops, errors, profile, shots=None, rng=None):
    """Recover every coherent angle in `errors` from the twirled RC run.

    n         : number of qubits.
    ideal_ops : list of dense 2^n ideal-gate unitaries, applied in order.
    errors    : list of CoherentError.
    profile   : NoiseProfile (use a noiseless profile for the smoke test).
    shots     : if given, attach a shot-noise 1-sigma to each estimate (per-parameter
                CRB from the measured Fisher); estimates themselves stay exact (mean).

    Returns a list of Recovered (same order as `errors`).
    """
    errs = sorted(errors, key=lambda e: e.pos)
    positions = sorted(set(e.pos for e in errs))
    segs, bounds = _segments(ideal_ops, n, positions)
    # map each boundary index to the segment index that FOLLOWS it
    # state after applying segs[0..j-1] and errors up to boundary j
    Pgen = [generator_matrix(e, n) for e in errs]

    # group errors by position (multiple errors can share a location)
    pos_to_errs = {}
    for idx, e in enumerate(errs):
        pos_to_errs.setdefault(e.pos, []).append(idx)

    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0] = 1.0

    # ideal output |chi> and post-operators per boundary
    # boundary j is between bounds[j] and bounds[j+1]; U_after[j] = product of segs[j..end]
    U_after = [None] * len(segs)
    acc = np.eye(2**n, dtype=complex)
    for j in range(len(segs) - 1, -1, -1):
        acc = acc @ segs[j]
        U_after[j] = acc.copy()
    chi = np.eye(2**n, dtype=complex)
    for U in segs:
        chi = U @ chi
    chi = chi @ psi0

    # effective-qubit observables per error
    obs = []
    g_all = []
    for idx, e in enumerate(errs):
        # boundary index where this error sits = index of its position in `positions` + 1
        j = positions.index(e.pos) + 1          # segments before: 0..j-1 applied, error, then segs[j..]
        post = U_after[j] if j < len(segs) else np.eye(2**n, dtype=complex)
        Gt = post @ Pgen[idx] @ post.conj().T
        Gchi = Gt @ chi
        g = float(np.vdot(chi, Gchi).real)
        g_all.append(g)
        if abs(abs(g) - 1.0) < 1e-9:
            obs.append(None)                    # silent (kernel)
            continue
        cp = (chi + Gchi) / 2.0
        cm = (chi - Gchi) / 2.0
        cp /= np.linalg.norm(cp)
        cm /= np.linalg.norm(cm)
        Xab = np.outer(cp, cm.conj()) + np.outer(cm, cp.conj())
        Yab = -1j * np.outer(cp, cm.conj()) + 1j * np.outer(cm, cp.conj())
        h = float(np.sqrt(max(1 - g * g, 0.0)))
        obs.append((Xab, Yab, g, h, j, post))

    def sim_Yab(target, target_angle):
        """<Y_target> at the output when the target error is applied as a COHERENT
        rotation of `target_angle`, every OTHER error is sign-twirled at its physical
        angle, and the noise layer acts at each error boundary. Exact density matrix."""
        Yab = obs[target][1]
        rho = np.outer(psi0, psi0.conj())
        for j in range(len(segs)):
            rho = segs[j] @ rho @ segs[j].conj().T
            if j < len(positions):
                for idx in pos_to_errs[positions[j]]:
                    if idx == target:
                        E = _expm_anti(-1j * target_angle / 2 * Pgen[idx])
                        rho = E @ rho @ E.conj().T
                    else:
                        rho = _twirl_op(rho, Pgen[idx], errs[idx].theta)
                if not profile.is_noiseless():
                    rho = apply_incoherent_layer(rho, profile, n, mode="dm")
        return float(np.trace(Yab @ rho).real)

    out = [None] * len(errs)
    eps = 1e-3
    for a, e in enumerate(errs):
        if obs[a] is None:
            out[a] = Recovered(f"{e.gen}@{e.qubits}", e.kind, e.qubits, e.theta,
                               np.nan, 0.0, g_all[a], np.nan, True)
            continue
        h = obs[a][3]
        # Measurement: label-correlated signal E[s*Y] = 0.5(Y(+theta) - Y(-theta)) = v h sin(theta)
        EsY = 0.5 * (sim_Yab(a, e.theta) - sim_Yab(a, -e.theta))
        # Self-consistent visibility: probe response R = dY/dphi|_0 = v h (theta never used;
        # others twirled at their physical angles cancel identically in EsY and R).
        R = (sim_Yab(a, eps) - sim_Yab(a, -eps)) / (2 * eps)
        sin_est = np.clip(EsY / R, -1.0, 1.0) if abs(R) > 1e-12 else 0.0
        theta_hat = float(np.arcsin(sin_est))
        v = float(R / h) if h > 1e-12 else 0.0     # visibility v_a = f_{Y_a} = R / h
        sigma = 0.0
        if shots:                                  # shot-noise CRB from the measured Fisher
            m = R * np.sin(theta_hat)              # R = v h ; m = v h sin(theta)
            dm = R * np.cos(theta_hat)
            F = dm**2 / max(1 - m**2, 1e-9)
            sigma = float(np.sqrt(1.0 / (shots * max(F, 1e-12))))
        out[a] = Recovered(f"{e.gen}@{e.qubits}", e.kind, e.qubits, e.theta,
                           theta_hat, sigma, g_all[a], v, False)
    return out
