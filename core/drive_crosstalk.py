"""Drive-induced leakage from a neighbor's REAL microwave pulse (DRIVE_CROSSTALK_SPEC Sec 1).

A third, physically distinct mechanism on top of core/multiqubit_zz.py's static ZZ term:
each REAL pulse fired on a neighbor leaks a fixed small angle phi_1j onto the target
qubit, additively, to leading order. Present only while a real pulse is firing, absent
for virtual/pulse-free operations -- this is what separates it from the always-on
kappa_1j coupling.

Resolving phi_1j needs a THIRD, independently randomized label p_j (padding: fire the
neighbor's already-calibrated X_pi pulse twice more, net identity on the logical state,
extra real pulses on the wire) on top of s_1 and c_j. Two labels give two numbers
(an intercept and a slope in c_j); there are three physical unknowns -- see
DRIVE_CROSSTALK_SPEC Sec 0 for why no reprocessing of the existing two-label data can
separate them.

Convention: 1-indexed qubits, big-endian, matching operator_on_state.py. Labels are
always returned to the caller (twirl.py's invariant, carried through multiqubit_zz.py);
this module never averages over s_1, c_j, or p_j internally.
"""

import numpy as np

from .operator_on_state import single_qubit_op
from .single_qubit_gates import X_gate, Rz

_X = X_gate()


def sample_padding_label(rng=None):
    """Draw one independent, uniform +/-1 padding label p_j.

    p_j=+1: no padding, the neighbor receives only whatever c_j already required.
    p_j=-1: padding, two extra real X_pi pulses fired on the neighbor after its
            c_j-required preparation.

    Independent of s_1 and c_j by construction (a fresh draw); handed back to the
    caller and never averaged here, matching sample_neighbor_labels's convention.
    """
    if rng is None:
        rng = np.random.default_rng()
    return 1 if rng.random() < 0.5 else -1


def n_real_pulses(c_j, p_j):
    """Total real X_pi pulses fired on the neighbor this shot (Sec 1 table).

    sp_pulses = 0 if c_j == +1 (prepared |0>, no pulse) else 1 (prepared |1>, one pulse)
    pad_pulses = 0 if p_j == +1 (no padding) else 2 (two padding pulses)
    n_j = sp_pulses + pad_pulses, in {0, 1, 2, 3}.
    """
    sp_pulses = 0 if c_j == 1 else 1
    pad_pulses = 0 if p_j == 1 else 2
    return sp_pulses + pad_pulses


def apply_padding(state, p_j, qubit_j, n):
    """Fire the neighbor's existing calibrated X_pi pulse twice more if p_j=-1.

    X_pi . X_pi = -I (identity up to an unobservable global phase), so the neighbor's
    computational-basis state set by c_j is unchanged -- only the real pulse count
    n_real_pulses grows. No new gate is calibrated for this; it is the SAME X_gate
    used by multiqubit_zz.prepare_neighbor_state, applied twice. p_j=+1 is a no-op.
    """
    if p_j == -1:
        state = single_qubit_op(state, _X, qubit_j, n)
        state = single_qubit_op(state, _X, qubit_j, n)
    return state


def inject_drive_leakage(state, phi_1j, n_j, target_qubit, n):
    """Apply the phenomenological per-pulse leakage Rz(phi_1j * n_j) to `target_qubit`.

    Explicit injection of the additive-angle model (Sec 1): each of the n_j real
    pulses fired on the neighbor this shot is assumed to leak a fixed angle phi_1j onto
    the target, additively, to leading order -- the same linear-accumulation assumption
    used for ping-pong amplitude calibration elsewhere in this project. This is a
    phenomenological stand-in for the microscopic leakage mechanism (Sec 4 scope note),
    not a simulation of the pulse envelope. Twirling by s_1 (which this function does
    NOT apply -- the caller passes phi_1j already carrying whatever sign it needs, the
    same division of responsibility as twirl.apply_signed_z_error) makes the leaked
    angle flip sign along with the rest of the target's Z-type error under the RC
    X-dressing, matching the exact outcome model of Sec 2.1.
    """
    if n_j == 0 or phi_1j == 0.0:
        return state
    return single_qubit_op(state, Rz(phi_1j * n_j), target_qubit, n)
