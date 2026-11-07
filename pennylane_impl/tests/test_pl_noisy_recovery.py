"""Recovery under noise on default.mixed, mirroring the hardware protocol.

Noise is attached with PennyLane's NoiseModel: phase damping + depolarizing after
every coherent-error layer, and a symmetric readout bit flip through ``meas_map``.
These channels are unital and commute with the X dressing, so they only shrink the
signal by the visibility v(L), which the reference circuit measures.

(Readout error is added through ``meas_map`` rather than ``default.mixed``'s
``readout_prob`` argument: in PennyLane 0.45.1, and on ``main`` as of 2026-11-02,
executing a circuit on ``default.mixed(readout_prob=...)`` raises
``TypeError: 'float' object is not iterable``.)
"""

import numpy as np
import pennylane as qp
import pytest

from labeled_rc import (
    bootstrap_theta,
    fit_theta,
    label_correlation,
    marginal_mean,
    sample_signs,
    sign_label_twirl,
    signal_tape_qnode,
    visibility_qnode,
)


def _layer_noise(op, **kwargs):
    """Noise applied after every coherent-error layer RZ."""
    qp.PhaseDamping(0.02, wires=op.wires)
    qp.DepolarizingChannel(0.005, wires=op.wires)


def _readout_noise(op, **kwargs):
    """Symmetric readout error on the measured wire."""
    qp.BitFlip(0.01, wires=op.wires)


NOISE = qp.NoiseModel(
    {qp.noise.op_eq(qp.RZ): _layer_noise},
    meas_map={qp.noise.meas_eq(qp.expval): _readout_noise},
)


def test_visibility_matches_closed_form():
    dev = qp.device("default.mixed", wires=1)
    for L in (1, 4):
        v = float(_noisy(visibility_qnode(dev, L))())
        expected = (1 - 0.02) ** (L / 2) * (1 - 4 * 0.005 / 3) ** L * (1 - 2 * 0.01)
        assert v == pytest.approx(expected, abs=1e-12)


def _noisy(qnode):
    return qp.add_noise(qnode, NOISE)


def _all_sign_vectors(L):
    idx = np.arange(2**L)
    bits = (idx[:, None] >> np.arange(L)[::-1]) & 1
    return np.where(bits == 1, 1, -1)


@pytest.mark.parametrize("theta_star", [0.0, 0.05, 0.2])
@pytest.mark.parametrize("L", [1, 4])
def test_exact_recovery_and_marginal_loss(theta_star, L):
    dev = qp.device("default.mixed", wires=1)
    signs = _all_sign_vectors(L)  # every label vector once: an exact full twirl
    y = _noisy(sign_label_twirl(signal_tape_qnode(dev, L), signs=signs))(theta_star)
    v = float(_noisy(visibility_qnode(dev, L))())
    m = signs.sum(axis=1)

    assert 0.9 < v < 1.0  # noise really reduced the visibility
    assert np.allclose(y, v * np.sin(theta_star * m), atol=1e-10)
    assert fit_theta(m, y, v) == pytest.approx(theta_star, abs=1e-6)
    assert abs(marginal_mean(y)) < 1e-10  # standard RC: label-averaged signal is gone
    if theta_star > 0:
        assert label_correlation(m, y) > 0  # ...but it is still in the labels


def test_finite_shots_recovery_with_confidence_interval():
    theta_star, L, instances = 0.1, 4, 50
    dev = qp.device("default.mixed", wires=1, seed=2026)
    signs = sample_signs(instances, L, seed=11)
    signal = qp.set_shots(_noisy(sign_label_twirl(signal_tape_qnode(dev, L), signs=signs)), 3000)
    reference = qp.set_shots(_noisy(visibility_qnode(dev, L)), 3000)

    y = np.asarray(signal(theta_star))
    v = float(np.mean([reference() for _ in range(20)]))
    theta_hat, (lo, hi) = bootstrap_theta(signs.sum(axis=1), y, v, num_boot=400, seed=3)

    assert lo <= theta_star <= hi
    assert abs(theta_hat - theta_star) < 0.01
