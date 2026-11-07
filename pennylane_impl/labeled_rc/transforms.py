"""PennyLane transforms for randomized compiling, with and without kept labels.

Two batch transforms are provided:

``pauli_twirl``
    Standard randomized compiling: every CNOT / CZ is dressed with a random
    Pauli frame, ``num_samples`` twirled circuits are executed, and the results
    are averaged. PennyLane (v0.45) has no built-in Pauli-twirling transform.

``sign_label_twirl``
    The construction of arXiv:2607.26756. Every marked error location (by
    default every ``qp.RZ``) is dressed with X on both sides when its label is
    s = -1, since X RZ(theta) X = RZ(-theta). The caller supplies the labels
    (``signs``), so they are owned and never averaged away; the post-processing
    returns one result per twirl instance, in the same order as ``signs``.
"""

import numpy as np
import pennylane as qp
from pennylane.tape import QuantumScriptBatch
from pennylane.typing import PostprocessingFn

from .frames import PAULI_LABELS, pauli_op, twirl_frames


def sample_signs(num_samples, num_layers, p_plus=0.5, seed=None):
    """Draw a (num_samples, num_layers) array of +/-1 labels with P(+1) = p_plus.

    p_plus = 0.5 is the fully twirled case (mean label mu = 0).
    """
    rng = np.random.default_rng(seed)
    return np.where(rng.random((num_samples, num_layers)) < p_plus, 1, -1).astype(int)


def _stack(results):
    """Stack per-tape results into one array with the instance index first."""
    if isinstance(results[0], tuple):
        return tuple(qp.math.stack([r[i] for r in results]) for i in range(len(results[0])))
    return qp.math.stack(results)


@qp.transform
def pauli_twirl(
    tape, num_samples=20, seed=None, gates=("CNOT", "CZ")
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Standard randomized compiling over the two-qubit Clifford gates in ``gates``.

    Returns ``num_samples`` twirled tapes; the post-processing averages their
    results. In the absence of noise every twirled tape is equivalent to the
    original circuit, so the average equals the untwirled result.
    """
    rng = np.random.default_rng(seed)
    tapes = []
    for _ in range(num_samples):
        new_ops = []
        for op in tape.operations:
            if op.name in gates:
                frames = twirl_frames(op.name)
                a, b = (PAULI_LABELS[k] for k in rng.integers(4, size=2))
                c, d = frames[(a, b)]
                w0, w1 = op.wires
                new_ops += [o for o in (pauli_op(a, w0), pauli_op(b, w1)) if o is not None]
                new_ops.append(op)
                new_ops += [o for o in (pauli_op(c, w0), pauli_op(d, w1)) if o is not None]
            else:
                new_ops.append(op)
        tapes.append(tape.copy(operations=new_ops))

    def processing_fn(results):
        stacked = _stack(results)
        if isinstance(stacked, tuple):
            return tuple(qp.math.mean(s, axis=0) for s in stacked)
        return qp.math.mean(stacked, axis=0)

    return tapes, processing_fn


@qp.transform
def sign_label_twirl(tape, signs, error_op="RZ") -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Label-keeping twirl of the coherent Z-error locations of a circuit.

    Args:
        tape: circuit whose ``error_op`` gates (default ``RZ``) are the error locations.
        signs: array of shape (num_samples, L) with entries +/-1, where L is the
            number of ``error_op`` gates in the tape. Row i is the label vector
            of twirl instance i; label -1 dresses that location with X ... X.
        error_op: name of the operation type treated as an error location.

    Returns:
        One tape per row of ``signs``. The post-processing stacks the results
        so that result[i] belongs to the label vector signs[i].
    """
    signs = np.asarray(signs, dtype=int)
    if signs.ndim != 2 or not np.all(np.abs(signs) == 1):
        raise ValueError("signs must be a 2-D array of +/-1")
    locations = [k for k, op in enumerate(tape.operations) if op.name == error_op]
    if len(locations) != signs.shape[1]:
        raise ValueError(
            f"signs has {signs.shape[1]} columns but the tape has {len(locations)} "
            f"{error_op} gates"
        )

    tapes = []
    for row in signs:
        label_at = dict(zip(locations, row))
        new_ops = []
        for k, op in enumerate(tape.operations):
            if label_at.get(k, 1) == -1:
                new_ops += [qp.PauliX(op.wires), op, qp.PauliX(op.wires)]
            else:
                new_ops.append(op)
        tapes.append(tape.copy(operations=new_ops))

    def processing_fn(results):
        return _stack(results)

    return tapes, processing_fn
