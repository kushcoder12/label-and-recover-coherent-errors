import numpy as np

from .operator_on_state import single_qubit_op
from .single_qubit_gates import Z_gate


def measure_z_on_state(state, p, n):
    """Projective Z-basis measurement on qubit p (1-indexed) of an n-qubit state.
    Returns (outcome, post_measurement_state). outcome is 0 or 1."""
    I = np.eye(2)
    Z = Z_gate()

    P0 = (I + Z) / 2
    P1 = (I - Z) / 2

    state_0 = single_qubit_op(state, P0, p, n)
    state_1 = single_qubit_op(state, P1, p, n)

    p0 = np.vdot(state_0, state_0).real
    p1 = np.vdot(state_1, state_1).real

    r = np.random.uniform(0, 1)

    if r < p0:
        return 0, state_0 / np.sqrt(p0)
    else:
        return 1, state_1 / np.sqrt(p1)
